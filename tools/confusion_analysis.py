"""Category confusion matrix analysis for DimASQP.

Analyses prediction errors to identify high-confusion category pairs.
This feeds into CCA Innovation 3 (Confusion-Guided Contrastive Generation):
pairs of categories that are frequently confused can be targeted with
contrastive training examples.

Inputs:
    --pred   prediction JSONL
    --gold   gold JSONL

Outputs:
    confusion_matrix_{task_domain}.json  — full confusion data
    Terminal summary of top confused pairs

Usage:
    python tools/confusion_analysis.py \
        --pred output/.../predictions.jsonl \
        --gold data/v2/eng/eng_restaurant_test.jsonl \
        --task_domain eng_restaurant
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple


def _load_jsonl(path: str) -> List[Dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                entries.append(json.loads(ln))
    return entries


def _build_confusion(
    pred_data: List[Dict],
    gold_data: List[Dict],
) -> Tuple[Dict[str, Dict[str, int]], Counter, Counter]:
    """Build a confusion matrix: confusion[gold_cat][pred_cat] = count.

    Also tracks unmatched gold (FN) and unmatched pred (FP) per category.
    Matching is done on (Aspect, Opinion) — the category is what may differ.
    """
    gold_dict = {e["ID"]: e.get("Quadruplet", []) for e in gold_data}
    pred_dict = {e["ID"]: e.get("Quadruplet", []) for e in pred_data}

    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    fn_counter: Counter = Counter()
    fp_counter: Counter = Counter()

    all_ids = set(gold_dict.keys()) | set(pred_dict.keys())
    for id_ in all_ids:
        gold_quads = gold_dict.get(id_, [])
        pred_quads = pred_dict.get(id_, [])

        pred_used = [False] * len(pred_quads)

        for gq in gold_quads:
            g_asp = gq.get("Aspect", "")
            g_opi = gq.get("Opinion", "")
            g_cat = gq.get("Category", "")
            matched = False

            for pi, pq in enumerate(pred_quads):
                if pred_used[pi]:
                    continue
                if pq.get("Aspect", "") == g_asp and pq.get("Opinion", "") == g_opi:
                    p_cat = pq.get("Category", "")
                    confusion[g_cat][p_cat] += 1
                    pred_used[pi] = True
                    matched = True
                    break

            if not matched:
                fn_counter[g_cat] += 1

        for pi, pq in enumerate(pred_quads):
            if not pred_used[pi]:
                fp_counter[pq.get("Category", "")] += 1

    return dict(confusion), fn_counter, fp_counter


def main():
    parser = argparse.ArgumentParser(
        description="Category confusion matrix analysis for DimASQP."
    )
    parser.add_argument("--pred", required=True, help="Prediction JSONL")
    parser.add_argument("--gold", required=True, help="Gold JSONL")
    parser.add_argument("--task_domain", default="unknown")
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--top_k", type=int, default=15,
                        help="Show top K confused pairs")
    args = parser.parse_args()

    pred_data = _load_jsonl(args.pred)
    gold_data = _load_jsonl(args.gold)

    confusion, fn_counter, fp_counter = _build_confusion(pred_data, gold_data)

    all_cats = sorted(set(confusion.keys()) | set(
        c for row in confusion.values() for c in row.keys()
    ))

    confused_pairs: List[Tuple[str, str, int]] = []
    for g_cat, preds in confusion.items():
        for p_cat, count in preds.items():
            if g_cat != p_cat:
                confused_pairs.append((g_cat, p_cat, count))
    confused_pairs.sort(key=lambda x: -x[2])

    correct = sum(confusion.get(c, {}).get(c, 0) for c in all_cats)
    total_matched = sum(v for row in confusion.values() for v in row.values())
    misclassified = total_matched - correct

    print(f"\n{'='*65}")
    print(f"  Category Confusion Analysis: {args.task_domain}")
    print(f"  Matched quads: {total_matched}  "
          f"(correct={correct}, misclassified={misclassified})")
    print(f"  Unmatched FN: {sum(fn_counter.values())}  "
          f"Unmatched FP: {sum(fp_counter.values())}")
    print(f"{'='*65}")

    print(f"\n  Top-{args.top_k} confused pairs (gold → predicted):")
    print(f"  {'Gold Category':<30} {'Pred Category':<30} {'Count':>5}")
    print(f"  {'-'*67}")
    for g, p, n in confused_pairs[:args.top_k]:
        print(f"  {g:<30} {p:<30} {n:>5}")

    bidirectional: Dict[Tuple[str, str], int] = defaultdict(int)
    for g, p, n in confused_pairs:
        key = tuple(sorted([g, p]))
        bidirectional[key] += n
    bidi_sorted = sorted(bidirectional.items(), key=lambda x: -x[1])

    print(f"\n  Top bidirectional confusion pairs:")
    print(f"  {'Category A':<30} {'Category B':<30} {'Total':>5}")
    print(f"  {'-'*67}")
    for (a, b), n in bidi_sorted[:args.top_k]:
        print(f"  {a:<30} {b:<30} {n:>5}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"confusion_matrix_{args.task_domain}.json")
    out_obj = {
        "task_domain": args.task_domain,
        "categories": all_cats,
        "confusion_matrix": {g: dict(p) for g, p in confusion.items()},
        "fn_by_category": dict(fn_counter),
        "fp_by_category": dict(fp_counter),
        "top_confused_pairs": [
            {"gold": g, "predicted": p, "count": n}
            for g, p, n in confused_pairs[:50]
        ],
        "top_bidirectional_pairs": [
            {"category_a": a, "category_b": b, "total": n}
            for (a, b), n in bidi_sorted[:50]
        ],
        "summary": {
            "total_matched": total_matched,
            "correct": correct,
            "misclassified": misclassified,
            "accuracy": round(correct / total_matched, 4) if total_matched > 0 else 0,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"\n  Output → {out_path}")


if __name__ == "__main__":
    main()
