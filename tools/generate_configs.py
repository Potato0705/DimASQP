"""
Generate category config JSON files for each language-domain combination.

Scans training data to extract all unique ENTITY#ATTRIBUTE categories,
then writes configs/{lang}_{domain}.json with category->id mappings.

Usage:
    python tools/generate_configs.py
    python tools/generate_configs.py --data_dir ./data --config_dir ./configs
"""
import json
import os
import argparse

DATASETS = [
    ("eng", "restaurant"),
    ("eng", "laptop"),
    ("zho", "restaurant"),
    ("zho", "laptop"),
    ("jpn", "hotel"),
    ("rus", "restaurant"),
    ("tat", "restaurant"),
    ("ukr", "restaurant"),
]


def extract_categories(jsonl_path):
    """Extract all unique categories from a JSONL file."""
    categories = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            for quad in entry.get("Quadruplet", []):
                categories.add(quad["Category"])
    return sorted(categories)


def main():
    parser = argparse.ArgumentParser(description="Generate category config files for DimASQP")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--config_dir", type=str, default="./configs")
    args = parser.parse_args()

    os.makedirs(args.config_dir, exist_ok=True)

    print("=" * 60)
    for lang, domain in DATASETS:
        name = f"{lang}_{domain}"
        # Use the original full training file for config generation
        # to ensure all categories are captured
        src_path = os.path.join(args.data_dir, lang, f"{name}_train_alltasks.jsonl")
        if not os.path.exists(src_path):
            print(f"[SKIP] {src_path} not found")
            continue

        categories = extract_categories(src_path)
        cat2id = {cat: idx for idx, cat in enumerate(categories)}

        config_path = os.path.join(args.config_dir, f"{name}.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cat2id, f, ensure_ascii=False, indent=2)

        print(f"[{name}] {len(categories)} categories -> {config_path}")
        for cat, idx in cat2id.items():
            print(f"    {idx}: {cat}")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
