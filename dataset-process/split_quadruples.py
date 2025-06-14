import json
from pathlib import Path

def parse_quadruples(output_str):
    """解析 output 字符串为四元组列表"""
    quads = []
    output_str = output_str.strip()
    parts = output_str.split('[SEP]')
    for part in parts:
        part = part.strip().replace('[END]', '')
        if part.count('|') == 3:
            target, argument, group, hate = [x.strip() for x in part.split('|')]
            quads.append((target, argument, group, hate))
    return quads

def split_data(input_file, out_dir):
    data = json.load(open(input_file, 'r', encoding='utf-8'))

    target_samples = []
    argument_samples = []
    group_samples = []
    hateful_samples = []

    for item in data:
        content = item['content']
        item_id = item['id']
        quadruples = parse_quadruples(item['output'])

        for quad_index, quad in enumerate(quadruples):
            target, argument, group, hate = quad

            if target and target.lower() != 'null':
                target_samples.append({
                    "id": item_id,
                    "quad_index": quad_index,
                    "content": content,
                    "target": target
                })

            if argument and argument.lower() != 'null':
                argument_samples.append({
                    "id": item_id,
                    "quad_index": quad_index,
                    "content": content,
                    "argument": argument
                })

            group_samples.append({
                "id": item_id,
                "quad_index": quad_index,
                "target": target,
                "argument": argument,
                "label": group
            })

            hateful_samples.append({
                "id": item_id,
                "quad_index": quad_index,
                "target": target,
                "argument": argument,
                "label": hate
            })

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    json.dump(target_samples, open(f"{out_dir}/target_data.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    json.dump(argument_samples, open(f"{out_dir}/argument_data.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    json.dump(group_samples, open(f"{out_dir}/group_data.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    json.dump(hateful_samples, open(f"{out_dir}/hateful_data.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

if __name__ == '__main__':
    split_data(input_file='train.json', out_dir='split_data')

