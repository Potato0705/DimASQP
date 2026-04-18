"""Entity×Attribute category distribution analysis for DimASQP.

Reads train/dev/test JSONL files for a given task_domain, builds E×A heatmaps,
identifies zero-shot and rare categories, and writes a machine-readable gap
report that drives the CCA (Compositional Category Augmentation) pipeline.

Outputs:
    category_heatmap_{task_domain}.csv      — E×A count matrix per split
    category_gap_report_{task_domain}.json   — prioritised list of (E, A) pairs
                                               for augmentation

Usage:
    python tools/category_analysis.py --task_domain eng_restaurant
    python tools/category_analysis.py --task_domain eng_restaurant --out_dir analysis_output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                entries.append(json.loads(ln))
    return entries


def _extract_categories(entries: List[Dict]) -> Counter:
    counter: Counter = Counter()
    for e in entries:
        for q in e.get("Quadruplet", []):
            cat = q.get("Category", "")
            if cat:
                counter[cat] += 1
    return counter


def _split_category(cat: str) -> Tuple[str, str]:
    parts = cat.split("#", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return cat, ""


def _discover_axes(all_cats: Set[str]) -> Tuple[List[str], List[str]]:
    entities: Set[str] = set()
    attributes: Set[str] = set()
    for cat in all_cats:
        e, a = _split_category(cat)
        entities.add(e)
        attributes.add(a)
    entity_order = ["FOOD", "SERVICE", "RESTAURANT", "AMBIENCE", "DRINKS", "LOCATION"]
    attr_order = ["GENERAL", "PRICES", "QUALITY", "STYLE_OPTIONS", "MISCELLANEOUS"]
    entities_sorted = [e for e in entity_order if e in entities]
    entities_sorted += sorted(entities - set(entity_order))
    attrs_sorted = [a for a in attr_order if a in attributes]
    attrs_sorted += sorted(attributes - set(attr_order))
    return entities_sorted, attrs_sorted


def _build_heatmap(
    counter: Counter, entities: List[str], attributes: List[str]
) -> List[List[int]]:
    matrix = []
    for e in entities:
        row = []
        for a in attributes:
            row.append(counter.get(f"{e}#{a}", 0))
        matrix.append(row)
    return matrix


def _print_heatmap(
    name: str,
    matrix: List[List[int]],
    entities: List[str],
    attributes: List[str],
) -> None:
    col_w = max(len(a) for a in attributes)
    col_w = max(col_w, 6)
    ent_w = max(len(e) for e in entities)
    header = " " * (ent_w + 2) + "".join(a.rjust(col_w + 1) for a in attributes)
    print(f"\n  [{name}]")
    print(header)
    for i, e in enumerate(entities):
        cells = []
        for v in matrix[i]:
            cells.append(str(v).rjust(col_w) if v > 0 else "✗".rjust(col_w))
        print(f"  {e.ljust(ent_w)}  {''.join(c + ' ' for c in cells)}")


def _write_csv(
    path: str,
    split_matrices: Dict[str, List[List[int]]],
    entities: List[str],
    attributes: List[str],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for split_name, matrix in split_matrices.items():
            f.write(f"# {split_name}\n")
            f.write("Entity," + ",".join(attributes) + "\n")
            for i, e in enumerate(entities):
                f.write(e + "," + ",".join(str(v) for v in matrix[i]) + "\n")
            f.write("\n")


def _build_gap_report(
    train_counter: Counter,
    dev_counter: Counter,
    test_counter: Counter,
    entities: List[str],
    attributes: List[str],
) -> List[Dict]:
    gaps = []
    for e in entities:
        for a in attributes:
            cat = f"{e}#{a}"
            train_n = train_counter.get(cat, 0)
            dev_n = dev_counter.get(cat, 0)
            test_n = test_counter.get(cat, 0)
            total_eval = dev_n + test_n
            if train_n == 0:
                status = "zero_shot"
                priority = 100 + total_eval
            elif train_n <= 5:
                status = "extremely_rare"
                priority = 50 + total_eval
            elif train_n <= 20:
                status = "rare"
                priority = 20 + total_eval
            else:
                status = "adequate"
                priority = 0
            gaps.append({
                "entity": e,
                "attribute": a,
                "category": cat,
                "train_count": train_n,
                "dev_count": dev_n,
                "test_count": test_n,
                "status": status,
                "priority": priority,
            })
    gaps.sort(key=lambda g: -g["priority"])
    return gaps


def main():
    parser = argparse.ArgumentParser(
        description="Analyse Entity×Attribute category distribution for DimASQP."
    )
    parser.add_argument("--task_domain", required=True,
                        help="e.g., eng_restaurant, zho_restaurant, jpn_hotel")
    parser.add_argument("--data_dir", default=None,
                        help="Base directory for JSONL files. Default: data/v2/{lang}/")
    parser.add_argument("--out_dir", default=".",
                        help="Directory for output CSV and JSON. Default: current dir.")
    args = parser.parse_args()

    lang = args.task_domain.split("_", 1)[0]
    data_dir = args.data_dir or os.path.join(_REPO_ROOT, "data", "v2", lang)

    train_path = os.path.join(data_dir, f"{args.task_domain}_train.jsonl")
    dev_path = os.path.join(data_dir, f"{args.task_domain}_dev.jsonl")
    test_path = os.path.join(data_dir, f"{args.task_domain}_test.jsonl")

    train_entries = _load_jsonl(train_path)
    dev_entries = _load_jsonl(dev_path)
    test_entries = _load_jsonl(test_path)

    if not train_entries:
        parser.error(f"No train data found at {train_path}. Run tools/convert_v2_all_languages.py first.")

    train_cats = _extract_categories(train_entries)
    dev_cats = _extract_categories(dev_entries)
    test_cats = _extract_categories(test_entries)

    all_cats = set(train_cats.keys()) | set(dev_cats.keys()) | set(test_cats.keys())
    entities, attributes = _discover_axes(all_cats)

    train_matrix = _build_heatmap(train_cats, entities, attributes)
    dev_matrix = _build_heatmap(dev_cats, entities, attributes)
    test_matrix = _build_heatmap(test_cats, entities, attributes)

    print(f"\n{'='*60}")
    print(f"  Category Analysis: {args.task_domain}")
    print(f"  Train: {len(train_entries)} sentences, {sum(train_cats.values())} quadruplets")
    print(f"  Dev:   {len(dev_entries)} sentences, {sum(dev_cats.values())} quadruplets")
    print(f"  Test:  {len(test_entries)} sentences, {sum(test_cats.values())} quadruplets")
    print(f"{'='*60}")

    _print_heatmap("Train", train_matrix, entities, attributes)
    _print_heatmap("Dev", dev_matrix, entities, attributes)
    _print_heatmap("Test", test_matrix, entities, attributes)

    gap_report = _build_gap_report(train_cats, dev_cats, test_cats, entities, attributes)

    zero_shot = [g for g in gap_report if g["status"] == "zero_shot"]
    rare = [g for g in gap_report if g["status"] in ("extremely_rare", "rare")]
    need_aug = [g for g in gap_report if g["status"] != "adequate"]

    print(f"\n  --- Summary ---")
    print(f"  Total E×A combinations: {len(entities)}×{len(attributes)} = {len(entities)*len(attributes)}")
    print(f"  Covered in train: {sum(1 for g in gap_report if g['train_count'] > 0)}")
    print(f"  Zero-shot (train=0): {len(zero_shot)}")
    for g in zero_shot:
        eval_note = ""
        if g["dev_count"] > 0 or g["test_count"] > 0:
            eval_note = f"  ← appears in eval (dev={g['dev_count']}, test={g['test_count']})"
        print(f"    - {g['category']} {eval_note}")
    if rare:
        print(f"  Rare (train≤20): {len(rare)}")
        for g in rare:
            print(f"    - {g['category']} (train={g['train_count']})")
    print(f"  Categories needing augmentation: {len(need_aug)}")

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"category_heatmap_{args.task_domain}.csv")
    _write_csv(csv_path, {"train": train_matrix, "dev": dev_matrix, "test": test_matrix},
               entities, attributes)
    print(f"\n  CSV  → {csv_path}")

    report_path = os.path.join(args.out_dir, f"category_gap_report_{args.task_domain}.json")
    report_obj = {
        "task_domain": args.task_domain,
        "entities": entities,
        "attributes": attributes,
        "n_train_sentences": len(train_entries),
        "n_dev_sentences": len(dev_entries),
        "n_test_sentences": len(test_entries),
        "gaps": gap_report,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)
    print(f"  JSON → {report_path}")


if __name__ == "__main__":
    main()
