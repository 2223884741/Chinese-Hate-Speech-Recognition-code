import json
import os
from pathlib import Path
from transformers import AutoTokenizer

model_name = "/root/nlp-new/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_label(example):
    text = example["content"]
    target = example["target"]

    start_idx = text.find(target)
    end_idx = start_idx + len(target)

    tokens = tokenizer.tokenize(text)
    char2tok = [-1] * len(text)

    # 映射字符到token索引（粗略对齐）
    tokenized_ids = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    for idx, (start, end) in enumerate(tokenized_ids["offset_mapping"]):
        for i in range(start, end):
            char2tok[i] = idx

    labels = ["O"] * len(tokens)
    if start_idx != -1 and end_idx <= len(text):
        token_start = char2tok[start_idx]
        token_end = char2tok[end_idx - 1]
        if token_start != -1 and token_end != -1:
            labels[token_start] = "B-TAR"
            for i in range(token_start + 1, token_end + 1):
                labels[i] = "I-TAR"

    return {
        "content": text,    # 保留原始文本
        "tokens": tokens,
        "labels": labels
    }

def convert(input_path, output_path):
    data = json.load(open(input_path, encoding="utf-8"))
    new_data = []

    for item in data:
        result = tokenize_and_label(item)
        if "B-TAR" in result["labels"]:  # 跳过找不到 target 的样本
            new_data.append(result)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in new_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    convert("/root/nlp-new/data/split_data/target_data.json", "ner_data/target_ner_train.jsonl")
