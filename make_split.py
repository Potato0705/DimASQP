import json
import random

random.seed(42)

src = "./data/eng_laptop_train_alltasks.jsonl"
train_out = "./output/train_gold_task3.jsonl"
valid_out = "./output/valid_gold_task3.jsonl"

val_ratio = 0.1

rows = []
with open(src, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

random.shuffle(rows)
n_val = max(1, int(len(rows) * val_ratio))
val_rows = rows[:n_val]
train_rows = rows[n_val:]

def dump(path, data):
    with open(path, "w", encoding="utf-8") as w:
        for x in data:
            w.write(json.dumps(x, ensure_ascii=False) + "\n")

dump(train_out, train_rows)
dump(valid_out, val_rows)

print("Total:", len(rows))
print("Train:", len(train_rows), "->", train_out)
print("Valid:", len(val_rows), "->", valid_out)
