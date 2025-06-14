import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TargetNERDataset(Dataset):
    def __init__(self, jsonl_file, label2id, tokenizer=None, max_length=128):
        self.samples = []
        self.label2id = label2id
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("/root/nlp-new/chinese-roberta-wwm-ext")
        self.max_length = max_length

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = item["tokens"]
        labels = item["labels"]
        text = item["content"]
        
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        word_ids = encoding.word_ids(batch_index=0)

        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id >= len(labels):
                aligned_labels.append(-100)  # 而不是 0，表示忽略这个 token
            else:
                aligned_labels.append(self.label2id.get(labels[word_id], 0))

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels),
            "text": text
        }
