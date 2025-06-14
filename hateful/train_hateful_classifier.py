import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

class BinaryHateDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128, mask_prob=0.15):
        self.data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = item["target"] + "。 " + item["argument"]
                label = item["binary_label"]
                self.data.append({
                    "text": text,
                    "label": label
                })
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.data)

    def mask_tokens(self, input_ids):
        input_ids = input_ids.clone()
        for i in range(len(input_ids)):
            if input_ids[i] not in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                if random.random() < self.mask_prob:
                    input_ids[i] = self.tokenizer.mask_token_id
        return input_ids

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(item["text"], padding="max_length", truncation=True,
                                 max_length=self.max_len, return_tensors="pt")
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        if self.mask_prob > 0:
            input_ids = self.mask_tokens(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(item["label"], dtype=torch.float)
        }

class BinaryClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        out = self.fc(self.dropout(cls))
        return out.squeeze(1)

class EarlyStopping:
    def __init__(self, patience=3, mode="max", delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.delta = delta

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            if score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    return True
            else:
                self.best_score = score
                self.counter = 0
        else:  # for "min"
            if score > self.best_score - self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    return True
            else:
                self.best_score = score
                self.counter = 0
        return False
        
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()
            logits = model(input_ids, attention_mask)
            preds = torch.sigmoid(logits).cpu().numpy()
            preds = (preds > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    print(f"Accuracy: {acc:.4f} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["non-hate", "hate"], yticklabels=["non-hate", "hate"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Binary Confusion Matrix")
    plt.tight_layout()
    plt.savefig("binary_confusion_matrix.png")
    plt.show()

    return f1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "/root/nlp-new/chinese-roberta-wwm-ext"
    train_path = "/root/nlp-new/hateful/binary_cls/train.jsonl"
    dev_path = "/root/nlp-new/hateful/binary_cls/dev.jsonl"

    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset = BinaryHateDataset(train_path, tokenizer)
    dev_dataset = BinaryHateDataset(dev_path, tokenizer, mask_prob=0.0)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16)

    model = BinaryClassifier(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_loader) * 5
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()
    
    early_stopper = EarlyStopping(patience=3, mode="max")
    
    best_f1 = 0
    
    for epoch in range(20):
        print(f"\nEpoch {epoch+1}")
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Train Loss: {total_loss/len(train_loader):.4f}")
        f1 = evaluate(model, dev_loader, device)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "/root/nlp-new/hateful/best_binary_model.pth")
            print("✅ New best model saved.")
        
        if early_stopper(f1):
            print("⏹️ 早停触发，停止训练。")
            break

    print("🎉 训练完成，最佳模型已保存。")
