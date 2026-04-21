"""
Split DimASQP official training JSONL files into train/dev sets.

Usage:
    python tools/split_dataset.py
    python tools/split_dataset.py --ratio 0.85 --seed 42
"""
import json
import os
import random
import argparse
from collections import defaultdict

# All language-domain combinations
DATASETS = [
    ("eng", "eng_restaurant"),
    ("eng", "eng_laptop"),
    ("zho", "zho_restaurant"),
    ("zho", "zho_laptop"),
    ("jpn", "jpn_hotel"),
    ("rus", "rus_restaurant"),
    ("tat", "tat_restaurant"),
    ("ukr", "ukr_restaurant"),
]

# Low-resource languages get a 90/10 split to preserve more training data
LOW_RESOURCE_LANGS = {"rus", "tat", "ukr"}


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def detect_id_split_marker(data):
    """Check if IDs contain train/dev/test split markers.

    E.g. "rest16_quad_dev_1" has "_dev_", "laptop_quad_train_2" has "_train_".
    Returns True if markers are found and can be used for splitting.
    """
    has_dev = any("_dev_" in entry["ID"] for entry in data)
    has_train = any("_train_" in entry["ID"] for entry in data)
    return has_dev and has_train


def split_by_id_marker(data):
    """Split data using train/dev/test markers in IDs.

    IDs with "_dev_" or "_test_" go to dev set (as local validation).
    IDs with "_train_" go to train set.
    """
    train_data = []
    dev_data = []
    for entry in data:
        eid = entry["ID"]
        if "_dev_" in eid or "_test_" in eid:
            dev_data.append(entry)
        else:
            train_data.append(entry)
    return train_data, dev_data


def group_by_doc(data):
    """Group entries by document-level ID prefix.

    For IDs with colon (e.g. "R0283:S003", "225:3_0"), group by the part
    before the first colon to keep same-document entries together.
    For IDs without colon, each entry is its own group.
    """
    groups = defaultdict(list)
    for entry in data:
        eid = entry["ID"]
        if ":" in eid:
            # Document-level grouping: "R0283:S003" -> "R0283"
            doc_id = eid.split(":", 1)[0]
        else:
            # Sentence-level: each entry is its own group
            doc_id = eid
        groups[doc_id].append(entry)
    return groups


def split_data_random(data, train_ratio, seed):
    """Split data at document level to avoid data leakage.

    For datasets where each entry is independent (no colon in IDs),
    this effectively does a random sample-level split.
    """
    groups = group_by_doc(data)
    doc_ids = sorted(groups.keys())

    random.seed(seed)
    random.shuffle(doc_ids)

    n_train = int(len(doc_ids) * train_ratio)
    train_doc_ids = set(doc_ids[:n_train])

    train_data = []
    dev_data = []
    for doc_id in doc_ids:
        entries = groups[doc_id]
        if doc_id in train_doc_ids:
            train_data.extend(entries)
        else:
            dev_data.extend(entries)

    return train_data, dev_data


def split_data(data, train_ratio, seed):
    """Smart split: use ID markers if available, otherwise random split."""
    if detect_id_split_marker(data):
        train_data, dev_data = split_by_id_marker(data)
        method = "id_marker"
    else:
        train_data, dev_data = split_data_random(data, train_ratio, seed)
        method = "random"
    return train_data, dev_data, method


def main():
    parser = argparse.ArgumentParser(description="Split DimASQP training data into train/dev")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root data directory")
    parser.add_argument("--ratio", type=float, default=0.85,
                        help="Train ratio for high-resource languages (default: 0.85)")
    parser.add_argument("--low_ratio", type=float, default=0.90,
                        help="Train ratio for low-resource languages (default: 0.90)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"Seed: {args.seed}")
    print(f"High-resource train ratio: {args.ratio}")
    print(f"Low-resource train ratio: {args.low_ratio}")
    print(f"Low-resource languages: {LOW_RESOURCE_LANGS}")
    print("=" * 60)

    for lang, name in DATASETS:
        src_path = os.path.join(args.data_dir, lang, f"{name}_train_alltasks.jsonl")
        if not os.path.exists(src_path):
            print(f"[SKIP] {src_path} not found")
            continue

        data = load_jsonl(src_path)
        ratio = args.low_ratio if lang in LOW_RESOURCE_LANGS else args.ratio
        train_data, dev_data, method = split_data(data, ratio, args.seed)

        train_path = os.path.join(args.data_dir, lang, f"{name}_train.jsonl")
        dev_path = os.path.join(args.data_dir, lang, f"{name}_dev.jsonl")
        save_jsonl(train_data, train_path)
        save_jsonl(dev_data, dev_path)

        print(f"[{name}] total={len(data)}, train={len(train_data)}, dev={len(dev_data)} "
              f"(method={method})")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
