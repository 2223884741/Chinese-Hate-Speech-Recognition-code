import torch
import json
import os
import difflib
from collections import namedtuple
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report, f1_score
from transformers import AutoTokenizer
from dataset import TargetNERDataset
from model import BERTCRFForNER
from tqdm import tqdm

model_name = "/root/nlp-new/chinese-roberta-wwm-ext"
label2id = {"O": 0, "B-TAR": 1, "I-TAR": 2}
id2label = {v: k for k, v in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = TargetNERDataset("ner_data/target_ner_train.jsonl", label2id, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = BERTCRFForNER(model_name, num_labels=len(label2id)).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

val_dataset = TargetNERDataset("ner_data/target_dev.jsonl", label2id, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=16)

# 定义实体结构体方便管理
Entity = namedtuple("Entity", ["text", "start", "end"])

def get_entities_from_labels(labels, tokens):
    """
    根据BIO标签序列和token序列提取实体列表，实体为Entity(text, start_idx, end_idx)
    """
    entities = []
    entity_tokens = []
    entity_start = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if entity_tokens:
                entities.append(Entity("".join(entity_tokens), entity_start, i-1))
            entity_tokens = [tokens[i]]
            entity_start = i
        elif label.startswith("I-") and entity_tokens:
            entity_tokens.append(tokens[i])
        else:
            if entity_tokens:
                entities.append(Entity("".join(entity_tokens), entity_start, i-1))
                entity_tokens = []
            entity_start = None
    # 结尾收尾
    if entity_tokens:
        entities.append(Entity("".join(entity_tokens), entity_start, len(labels)-1))
    return entities

def hard_match(pred_entities, gold_entities):
    """
    硬匹配：两个实体列表所有元素完全匹配（文本完全相等，且数量也相等）
    """
    if len(pred_entities) != len(gold_entities):
        return False
    for p, g in zip(pred_entities, gold_entities):
        if p.text != g.text:
            return False
    return True

def soft_match(pred_entities, gold_entities):
    """
    软匹配：这里简化为每个实体的文本相似度超过0.5才匹配成功
    对应你的需求，后续你可以细化到target和argument。
    """
    if len(pred_entities) != len(gold_entities):
        return False
    for p, g in zip(pred_entities, gold_entities):
        # 用 difflib.SequenceMatcher 计算相似度
        sim = difflib.SequenceMatcher(None, p.text, g.text).ratio()
        if sim < 0.5:
            return False
    return True
    
def save_best_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Best model saved to {save_path}")
    
def evaluate(model, val_loader, tokenizer):
    model.eval()
    all_preds_labels = []
    all_gold_labels = []
    all_preds_texts = []
    all_gold_texts = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cpu().tolist()
            # 这里假设你的Dataset返回原始文本，batch["text"] 是 List[str]
            texts = batch["text"]

            preds = model(input_ids, attention_mask)
            
            for pred_seq, label_seq, mask, text in zip(preds, labels, batch["attention_mask"], texts):
                true_len = min(sum(mask).item(), len(label_seq), len(tokenizer.tokenize(text)))
                pred_labels = []
                true_labels = []
                tokens = tokenizer.tokenize(text)[:true_len]  # 分词

                for p, l in zip(pred_seq[:true_len], label_seq[:true_len]):
                    if l != -100:
                        pred_labels.append(id2label[p])
                        true_labels.append(id2label[l])
                all_preds_labels.append(pred_labels)
                all_gold_labels.append(true_labels)
                all_preds_texts.append(tokens)
                all_gold_texts.append(tokens)  # 文本一样，实体从labels提取

    # 用BIO标签转换实体文本
    all_pred_entities = [get_entities_from_labels(p, t) for p, t in zip(all_preds_labels, all_preds_texts)]
    all_gold_entities = [get_entities_from_labels(g, t) for g, t in zip(all_gold_labels, all_gold_texts)]

    # 计算硬匹配和软匹配的TP, FP, FN用于计算F1
    hard_TP = hard_FP = hard_FN = 0
    soft_TP = soft_FP = soft_FN = 0

    for pred_e, gold_e in zip(all_pred_entities, all_gold_entities):
        if hard_match(pred_e, gold_e):
            hard_TP += 1
        else:
            hard_FP += 1
            hard_FN += 1

        if soft_match(pred_e, gold_e):
            soft_TP += 1
        else:
            soft_FP += 1
            soft_FN += 1

    def calc_f1(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    hard_p, hard_r, hard_f1 = calc_f1(hard_TP, hard_FP, hard_FN)
    soft_p, soft_r, soft_f1 = calc_f1(soft_TP, soft_FP, soft_FN)
    avg_f1 = (hard_f1 + soft_f1) / 2

    print(f"硬匹配F1: {hard_f1:.4f}，软匹配F1: {soft_f1:.4f}，平均F1: {avg_f1:.4f}")

    # 你依然可以输出seqeval的F1和报告辅助分析
    seqeval_f1 = f1_score(all_gold_labels, all_preds_labels, average="macro")
    print(f"Seqeval F1: {seqeval_f1:.4f}")
    print(classification_report(all_gold_labels, all_preds_labels))

    return avg_f1

best_f1 = 0.0
patience = 5
early_stop_counter = 0
save_path = "./saved_models/best_model.pth"

for epoch in range(30):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()

        loss = model(input_ids, attention_mask, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")

    f1 = evaluate(model, val_loader, tokenizer)  # 传入tokenizer

    if f1 is not None and f1 > best_f1:
        best_f1 = f1
        early_stop_counter = 0
        save_best_model(model, save_path)
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break
