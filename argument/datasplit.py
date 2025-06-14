import random
import json

def split_ner_data(input_path, train_path, dev_path, ratio=0.9):
    data = [json.loads(line) for line in open(input_path, 'r', encoding='utf-8')]
    random.shuffle(data)
    split_idx = int(len(data) * ratio)
    train_data = data[:split_idx]
    dev_data = data[split_idx:]

    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(dev_path, 'w', encoding='utf-8') as f:
        for item in dev_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    split_ner_data("ner_data/argument_ner_train.jsonl",
                   "ner_data/argument_train.jsonl",
                   "ner_data/argument_dev.jsonl")
