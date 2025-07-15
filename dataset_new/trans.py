import json

# 假设你的文件名是 data.json
with open("/home/wangao/Code_clone/c4/C4_wang/dataset_new/pair_test.json", "r", encoding="utf-8") as f:
    objs = json.load(f)

with open("/home/wangao/Code_clone/c4/C4_wang/dataset_new/pair_test.jsonl", "w", encoding="utf-8") as fout:
    for obj in objs:
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")