import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report, multilabel_confusion_matrix, confusion_matrix
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

label_list = ["Racism", "Region", "LGBTQ", "Sexism", "others", "non-hate"]
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

def normalize_labels(raw_labels):
    """
    如果标签不在前五类，则强制归入 non-hate
    raw_labels 可以是字符串或列表
    返回多标签列表
    """
    if isinstance(raw_labels, str):
        raw_labels = [raw_labels]
    filtered = []
    for l in raw_labels:
        if l in label2id and l != "non-hate":
            filtered.append(l)
    if not filtered:
        filtered = ["non-hate"]
    return filtered

class HateGroupDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128, mask_prob=0.0):
        """
        mask_prob: 训练时随机遮挡token的概率，用于数据增强
        """
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                text = item["target"] + "。 " + item["argument"]
                labels = normalize_labels(item["label"])
                label_vec = [0] * len(label2id)
                for label in labels:
                    label_vec[label2id[label]] = 1
                self.samples.append({
                    "text": text,
                    "label": torch.tensor(label_vec, dtype=torch.float)
                })
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.samples)

    def mask_tokens(self, input_ids):
        """
        随机遮挡token，mask token id用tokenizer.mask_token_id替代
        """
        input_ids = input_ids.clone()
        for i in range(len(input_ids)):
            if input_ids[i] not in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                if random.random() < self.mask_prob:
                    input_ids[i] = self.tokenizer.mask_token_id
        return input_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(item["text"], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        if self.mask_prob > 0:
            input_ids = self.mask_tokens(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": item["label"]
        }

class AttentionHead(nn.Module):
    def __init__(self, hidden_dim, head_count=4):
        super().__init__()
        self.head_count = head_count
        self.hidden_dim = hidden_dim
        assert hidden_dim % head_count == 0, "hidden_dim must be divisible by head_count"
        self.head_dim = hidden_dim // head_count

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        Q = self.query(x).view(batch_size, seq_len, self.head_count, self.head_dim).transpose(1,2)  # (b, heads, seq_len, head_dim)
        K = self.key(x).view(batch_size, seq_len, self.head_count, self.head_dim).transpose(1,2)
        V = self.value(x).view(batch_size, seq_len, self.head_count, self.head_dim).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)  # (b, heads, seq_len, head_dim)
        context = context.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        out = self.out(context)
        return out
        
class TextCNN(nn.Module):
    def __init__(self, hidden_size, filter_sizes=[2,3,4], num_filters=64):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        x = x.transpose(1, 2)  # -> (batch, hidden_size, seq_len)
        conv_outputs = []
        for conv in self.convs:
            c = conv(x)  # (batch, num_filters, seq_len - fs + 1)
            c = torch.relu(c)
            c = torch.max(c, dim=2)[0]  # max pooling over time dimension (seq_len)
            conv_outputs.append(c)
        out = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(filter_sizes))
        out = self.dropout(out)
        return out

class MultiLabelBertTextCNNClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout_prob=0.2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.attention = AttentionHead(hidden_size, head_count=4)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

        self.text_cnn = TextCNN(hidden_size, filter_sizes=[2,3,4], num_filters=64)

        # CLS向量大小 + TextCNN输出大小
        self.classifier = nn.Linear(hidden_size + 64 * 3, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        attn_output = self.attention(sequence_output)  # (batch, seq_len, hidden_size)
        combined = self.layernorm(sequence_output + attn_output)  # (batch, seq_len, hidden_size)

        cls_output = combined[:, 0, :]  # (batch, hidden_size)
        cls_output = self.dropout(cls_output)

        cnn_output = self.text_cnn(combined)  # (batch, num_filters * len(filter_sizes))

        concat_output = torch.cat([cls_output, cnn_output], dim=1)  # (batch, hidden_size + ...)

        logits = self.classifier(concat_output)
        return logits


def train(model, dataloader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask)
        loss = criterion(torch.sigmoid(logits), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)

best_macro_f1 = 0
best_model_path = "/root/nlp-new/group/best_group_model.pth"

def evaluate(model, dataloader, device, threshold=0.5, save_cm=False):
    global best_macro_f1
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    print("\n🔍 Classification Report (per label):")
    print(classification_report(all_labels, all_preds, target_names=label_list, digits=4))

    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"\n🎯 Micro-F1: {micro_f1:.4f} | Macro-F1: {macro_f1:.4f}")

    # 多标签混淆矩阵
    print("\n📊 Multilabel Confusion Matrix (TP, FP, FN per class):")
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    for idx, label in enumerate(label_list):
        tn, fp, fn, tp = mcm[idx].ravel()
        print(f"{label}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # 可视化每类 2x2 混淆矩阵
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for idx, (label, cm) in enumerate(zip(label_list, mcm)):
        tn, fp, fn, tp = cm.ravel()
        matrix = np.array([[tp, fp], [fn, tn]])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Pred:1", "Pred:0"], yticklabels=["True:1", "True:0"],
                    ax=axes[idx])
        axes[idx].set_title(f"{label}")

    plt.tight_layout()
    plt.suptitle("🔍 Confusion Matrix per Label", fontsize=16, y=1.03)
    plt.savefig("confusion_matrix_per_label.png")
    plt.show()

    # 保存最佳 macro-F1 时的 6x6 多类混淆矩阵
    if macro_f1 > best_macro_f1 and save_cm:
        best_macro_f1 = macro_f1
        y_pred_labels = all_preds.argmax(axis=1)
        y_true_labels = all_labels.argmax(axis=1)
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=list(range(len(label_list))))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=label_list, yticklabels=label_list)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Best Epoch Multi-class Confusion Matrix (via argmax)")
        plt.tight_layout()
        plt.savefig("best_epoch_confusion_matrix_6x6.png")
        plt.show()

    # 保存权重
    if macro_f1 > best_macro_f1 and save_cm:
        print(f"✨ Macro-F1 improved from {best_macro_f1:.4f} to {macro_f1:.4f}, saving model...")
        torch.save(model.state_dict(), best_model_path)

    return macro_f1, all_preds, all_labels

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "/root/nlp-new/chinese-roberta-wwm-ext"
    train_path = "/root/nlp-new/group/group_cls/train.jsonl"
    dev_path = "/root/nlp-new/group/group_cls/dev.jsonl"

    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset = HateGroupDataset(train_path, tokenizer, mask_prob=0.15)  # mask 15%词做数据增强
    dev_dataset = HateGroupDataset(dev_path, tokenizer, mask_prob=0.0)  # 验证集不mask

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16)

    model = MultiLabelBertTextCNNClassifier(model_name, len(label2id)).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 10  # 10轮
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    criterion = nn.BCELoss()

    early_stop_patience = 5
    no_improve_epochs = 0
    best_score = 0

    for epoch in range(20):
        print(f"\nEpoch {epoch+1}")
        train_loss = train(model, train_loader, optimizer, scheduler, device, criterion)
        print(f"Train Loss: {train_loss:.4f}")
        macro_f1, _, _ = evaluate(model, dev_loader, device, save_cm=False)

        # 判断是否是最佳模型
        if macro_f1 > best_score:
            print(f"✨ Macro-F1 improved from {best_score:.4f} to {macro_f1:.4f}, saving model...")
            best_score = macro_f1
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve_epochs += 1
            print(f"⚠️ No improvement for {no_improve_epochs} epochs")
            if no_improve_epochs >= early_stop_patience:
                print(f"🚨 Early stopping triggered at epoch {epoch+1}")
                break

    print(f"训练结束，最佳模型保存在 {best_model_path}")

