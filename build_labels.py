import json

def collect_laptop_categories(train_path, dev_path):
    cats = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            for quad in data["Quadruplet"]:
                cats.add(quad["Category"])

    # dev 中可能存在 train 没有的 Category
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # dev 里无 label，但为了完整性可跳过
            if "Quadruplet" in data:
                for quad in data["Quadruplet"]:
                    cats.add(quad["Category"])

    return sorted(list(cats))


if __name__ == "__main__":
    train_path = "./data/eng_laptop_train_alltasks.jsonl"
    dev_path = "./data/eng_laptop_dev_task3.jsonl"

    cats = collect_laptop_categories(train_path, dev_path)

    print(f"Total Laptop Categories: {len(cats)}")
    print(cats)

    with open("./configs/laptop_categories.txt", "w", encoding="utf-8") as fw:
        fw.write("\n".join(cats))

    print("\n✔ Laptop category labels saved to ./configs/laptop_categories.txt")
