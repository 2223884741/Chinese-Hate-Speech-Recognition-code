import torch
import json
from transformers import AutoTokenizer
from model import BERTCRFForNER
from tqdm import tqdm

model_name = "/root/nlp-new/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)

label2id = {"O": 0, "B-ARG": 1, "I-ARG": 2}
id2label = {v: k for k, v in label2id.items()}

model = BERTCRFForNER(model_name, num_labels=len(label2id))
model.load_state_dict(torch.load("./saved_models/best_model.pth"))  # 与训练保存路径一致
model.eval()
model.cuda()

def extract_argument_spans(tokens, pred_labels):
    spans = []
    curr = []
    for token, label in zip(tokens, pred_labels):
        if label == "B-ARG":
            if curr:
                spans.append(tokenizer.convert_tokens_to_string(curr))
                curr = []
            curr = [token]
        elif label == "I-ARG":
            if curr:
                curr.append(token)
        else:
            if curr:
                spans.append(tokenizer.convert_tokens_to_string(curr))
                curr = []
    if curr:
        spans.append(tokenizer.convert_tokens_to_string(curr))
    return spans

def predict(contents):
    results = []
    for content in tqdm(contents):
        tokens = tokenizer(content, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        input_ids = tokens["input_ids"].cuda()
        attention_mask = tokens["attention_mask"].cuda()

        with torch.no_grad():
            pred_ids = model(input_ids, attention_mask)  # 返回形如 List[List[int]] 或 tensor，batch=1

        # 处理输出，batch=1
        pred = model(input_ids, attention_mask)  # 返回 list[list[int]]
        pred_ids = pred[0]  # 直接拿 list，不需要 .cpu().tolist()
        pred_labels = [id2label[p] for p in pred_ids]
        tokens_list = tokenizer.convert_ids_to_tokens(input_ids[0])
        seq_len = attention_mask[0].sum().item()
        tokens_list = tokens_list[:seq_len]
        pred_labels = [id2label[i] for i in pred_ids[:seq_len]]

        arguments = extract_argument_spans(tokens_list, pred_labels)
        results.append({"content": content, "pred_arguments": arguments})

    return results

if __name__ == "__main__":
    with open("ner_data/argument_dev.jsonl", encoding="utf-8") as f:
        test_inputs = ["".join(json.loads(line)["tokens"]) for line in open("ner_data/argument_dev.jsonl", encoding='utf-8')]

    results = predict(test_inputs)
    with open("predict_argument_results.json", "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)
