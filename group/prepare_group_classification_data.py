import json
import os
from sklearn.model_selection import train_test_split

input_path = "/root/nlp-new/data/split_data/group_data.json"
output_dir = "/root/nlp-new/group/group_cls"
os.makedirs(output_dir, exist_ok=True)

label_list = ["Racism", "Region", "LGBTQ", "Sexism", "others", "non-hate"]
label2id = {label: i for i, label in enumerate(label_list)}
hate_labels = set(label_list) - {"non-hate"}

def load_and_process_data(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    processed = []
    for item in data:
        labels = item["label"]
        if isinstance(labels, str):
            labels = [labels]

        # 如果 label 中没有 hate 类别标签，就归为 non-hate
        labels = [label for label in labels if label in label2id]
        if not any(label in hate_labels for label in labels):
            labels = ["non-hate"]

        label_vector = [0] * len(label_list)
        for label in labels:
            label_vector[label2id[label]] = 1

        item["labels"] = label_vector
        processed.append(item)

    return processed

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    all_data = load_and_process_data(input_path)
    train, dev = train_test_split(all_data, test_size=0.2, random_state=42)

    save_jsonl(train, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(dev, os.path.join(output_dir, "dev.jsonl"))

    print(f"✅ 数据已处理完毕并保存到 {output_dir}，总数: {len(all_data)}")

