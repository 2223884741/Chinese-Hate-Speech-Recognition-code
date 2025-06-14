import json
import os
from sklearn.model_selection import train_test_split

input_path = "/root/nlp-new/data/split_data/group_data.json"
output_dir = "/root/nlp-new/hateful/binary_cls"
os.makedirs(output_dir, exist_ok=True)

def convert_to_binary_label(data):
    binary_data = []
    for item in data:
        labels = item["label"]
        if isinstance(labels, str):
            labels = [labels]
        # 非 non-hate 则归为 hate 类
        is_hate = any(l != "non-hate" for l in labels)
        item["binary_label"] = 1 if is_hate else 0
        binary_data.append(item)
    return binary_data

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    with open(input_path, encoding="utf-8") as f:
        all_data = json.load(f)
    all_data = convert_to_binary_label(all_data)
    train, dev = train_test_split(all_data, test_size=0.2, random_state=42)
    save_jsonl(train, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(dev, os.path.join(output_dir, "dev.jsonl"))
    print(f"✅ Binary 数据处理完毕，路径：{output_dir}，总数: {len(all_data)}")
