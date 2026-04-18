"""Category Coverage Rate (CCR) and Zero-shot Category Recall (ZCR) evaluator.

Novel evaluation metrics for DimASQP that expose per-category performance
hidden by aggregate cF1:

    CCR@k  = fraction of E×A categories (with ≥k test samples) where cF1 > 0
    ZCR    = recall on categories never seen in training
    Seen/Unseen cF1 split = aggregate cF1 computed separately on seen vs unseen categories

Inputs:
    --pred     prediction JSONL (model output)
    --gold     gold JSONL (ground truth)
    --train    train JSONL (to determine which categories are "seen")

Usage:
    python tools/eval_category_coverage.py \
        --pred output/.../predictions.jsonl \
        --gold data/v2/eng/eng_restaurant_test.jsonl \
        --train data/v2/eng/eng_restaurant_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

D_MAX = math.sqrt(128)


def _load_jsonl(path: str) -> List[Dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                entries.append(json.loads(ln))
    return entries


def _extract_train_categories(train_entries: List[Dict]) -> Set[str]:
    cats = set()
    for e in train_entries:
        for q in e.get("Quadruplet", []):
            c = q.get("Category", "")
            if c:
                cats.add(c)
    return cats


def _compute_per_category_scores(
    pred_data: List[Dict],
    gold_data: List[Dict],
) -> Dict[str, Dict]:
    gold_dict = {e["ID"]: e.get("Quadruplet", []) for e in gold_data}
    pred_dict = {e["ID"]: e.get("Quadruplet", []) for e in pred_data}

    cat_stats: Dict[str, Dict] = defaultdict(lambda: {
        "TP": 0, "cTP": 0.0, "FP": 0, "FN": 0,
        "gold_count": 0, "pred_count": 0,
    })

    all_ids = set(gold_dict.keys()) | set(pred_dict.keys())
    for id_ in all_ids:
        gold_quads = gold_dict.get(id_, [])
        pred_quads = pred_dict.get(id_, [])

        pred_matched = [False] * len(pred_quads)

        for gq in gold_quads:
            g_key = (gq.get("Aspect", ""), gq.get("Opinion", ""), gq.get("Category", ""))
            cat = gq.get("Category", "UNKNOWN")
            cat_stats[cat]["gold_count"] += 1
            best_score = -1.0
            best_idx = -1

            for pi, pq in enumerate(pred_quads):
                if pred_matched[pi]:
                    continue
                p_key = (pq.get("Aspect", ""), pq.get("Opinion", ""), pq.get("Category", ""))
                if g_key == p_key:
                    try:
                        gv, ga = map(float, gq["VA"].split("#"))
                        pv, pa = map(float, pq["VA"].split("#"))
                    except (ValueError, KeyError):
                        continue
                    if not (1.0 <= pv <= 9.0 and 1.0 <= pa <= 9.0):
                        continue
                    dist = math.sqrt((pv - gv) ** 2 + (pa - ga) ** 2)
                    score = max(0.0, 1.0 - dist / D_MAX)
                    if score > best_score:
                        best_score = score
                        best_idx = pi

            if best_idx >= 0:
                pred_matched[best_idx] = True
                cat_stats[cat]["TP"] += 1
                cat_stats[cat]["cTP"] += best_score
            else:
                cat_stats[cat]["FN"] += 1

        for pi, pq in enumerate(pred_quads):
            cat = pq.get("Category", "UNKNOWN")
            cat_stats[cat]["pred_count"] += 1
            if not pred_matched[pi]:
                cat_stats[cat]["FP"] += 1

    results = {}
    for cat, s in cat_stats.items():
        tp_fp = s["TP"] + s["FP"]
        tp_fn = s["TP"] + s["FN"]
        cprec = s["cTP"] / tp_fp if tp_fp > 0 else 0.0
        crecall = s["cTP"] / tp_fn if tp_fn > 0 else 0.0
        cf1 = 2 * cprec * crecall / (cprec + crecall) if (cprec + crecall) > 0 else 0.0
        results[cat] = {
            "TP": s["TP"],
            "cTP": round(s["cTP"], 4),
            "FP": s["FP"],
            "FN": s["FN"],
            "gold_count": s["gold_count"],
            "pred_count": s["pred_count"],
            "cPrecision": round(cprec, 4),
            "cRecall": round(crecall, 4),
            "cF1": round(cf1, 4),
        }
    return results


def _aggregate_cf1(per_cat: Dict[str, Dict], categories: Set[str]) -> Dict:
    tp = sum(per_cat[c]["TP"] for c in categories if c in per_cat)
    ctp = sum(per_cat[c]["cTP"] for c in categories if c in per_cat)
    fp = sum(per_cat[c]["FP"] for c in categories if c in per_cat)
    fn = sum(per_cat[c]["FN"] for c in categories if c in per_cat)
    tp_fp = tp + fp
    tp_fn = tp + fn
    cprec = ctp / tp_fp if tp_fp > 0 else 0.0
    crecall = ctp / tp_fn if tp_fn > 0 else 0.0
    cf1 = 2 * cprec * crecall / (cprec + crecall) if (cprec + crecall) > 0 else 0.0
    return {"cPrecision": round(cprec, 4), "cRecall": round(crecall, 4), "cF1": round(cf1, 4)}


def main():
    parser = argparse.ArgumentParser(
        description="Compute CCR, ZCR, and per-category cF1 for DimASQP."
    )
    parser.add_argument("--pred", required=True, help="Prediction JSONL")
    parser.add_argument("--gold", required=True, help="Gold test/dev JSONL")
    parser.add_argument("--train", required=True, help="Gold train JSONL (to identify seen categories)")
    parser.add_argument("--out", default=None, help="Output JSON path (optional)")
    args = parser.parse_args()

    pred_data = _load_jsonl(args.pred)
    gold_data = _load_jsonl(args.gold)
    train_data = _load_jsonl(args.train)

    seen_cats = _extract_train_categories(train_data)
    per_cat = _compute_per_category_scores(pred_data, gold_data)

    gold_cat_counts: Counter = Counter()
    for e in gold_data:
        for q in e.get("Quadruplet", []):
            c = q.get("Category", "")
            if c:
                gold_cat_counts[c] += 1

    all_eval_cats = set(gold_cat_counts.keys())
    unseen_cats = all_eval_cats - seen_cats
    seen_eval_cats = all_eval_cats & seen_cats

    ccr_at = {}
    for k in [1, 5, 10]:
        eligible = {c for c, n in gold_cat_counts.items() if n >= k}
        if not eligible:
            ccr_at[k] = None
            continue
        covered = sum(1 for c in eligible if c in per_cat and per_cat[c]["cF1"] > 0)
        ccr_at[k] = round(covered / len(eligible), 4)

    zcr_num = sum(per_cat.get(c, {}).get("TP", 0) for c in unseen_cats)
    zcr_den = sum(gold_cat_counts[c] for c in unseen_cats)
    zcr = round(zcr_num / zcr_den, 4) if zcr_den > 0 else None

    overall = _aggregate_cf1(per_cat, all_eval_cats)
    seen_agg = _aggregate_cf1(per_cat, seen_eval_cats)
    unseen_agg = _aggregate_cf1(per_cat, unseen_cats)

    print(f"\n{'='*65}")
    print(f"  Category Coverage Evaluation")
    print(f"  Pred: {args.pred}")
    print(f"  Gold: {args.gold}")
    print(f"  Train categories (seen): {len(seen_cats)}")
    print(f"  Eval categories: {len(all_eval_cats)} "
          f"(seen={len(seen_eval_cats)}, unseen={len(unseen_cats)})")
    print(f"{'='*65}")

    print(f"\n  Overall cF1:   {overall['cF1']:.4f}  "
          f"(cPrec={overall['cPrecision']:.4f}, cRecall={overall['cRecall']:.4f})")
    print(f"  Seen cF1:      {seen_agg['cF1']:.4f}  "
          f"(cPrec={seen_agg['cPrecision']:.4f}, cRecall={seen_agg['cRecall']:.4f})")
    print(f"  Unseen cF1:    {unseen_agg['cF1']:.4f}  "
          f"(cPrec={unseen_agg['cPrecision']:.4f}, cRecall={unseen_agg['cRecall']:.4f})")

    print(f"\n  CCR@1:  {ccr_at.get(1, 'N/A')}")
    print(f"  CCR@5:  {ccr_at.get(5, 'N/A')}")
    print(f"  CCR@10: {ccr_at.get(10, 'N/A')}")
    print(f"  ZCR:    {zcr if zcr is not None else 'N/A'} "
          f"({zcr_num}/{zcr_den} zero-shot quads recalled)")

    print(f"\n  --- Per-Category Breakdown ---")
    print(f"  {'Category':<30} {'Gold':>5} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'cF1':>7} {'cPrec':>7} {'cRecall':>7}  Status")
    sorted_cats = sorted(all_eval_cats, key=lambda c: gold_cat_counts[c], reverse=True)
    for cat in sorted_cats:
        s = per_cat.get(cat, {"TP": 0, "FP": 0, "FN": 0, "cF1": 0, "cPrecision": 0, "cRecall": 0})
        status = "seen" if cat in seen_cats else "UNSEEN"
        print(f"  {cat:<30} {gold_cat_counts[cat]:>5} {s['TP']:>4} {s['FP']:>4} {s['FN']:>4} "
              f"{s['cF1']:>7.4f} {s['cPrecision']:>7.4f} {s['cRecall']:>7.4f}  {status}")

    if args.out:
        out_obj = {
            "overall": overall,
            "seen": seen_agg,
            "unseen": unseen_agg,
            "ccr": ccr_at,
            "zcr": zcr,
            "zcr_detail": {"recalled": zcr_num, "total": zcr_den},
            "per_category": per_cat,
            "seen_categories": sorted(seen_cats),
            "unseen_categories": sorted(unseen_cats),
        }
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"\n  Results → {args.out}")


if __name__ == "__main__":
    main()
