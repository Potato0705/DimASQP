"""
Local evaluation wrapper that calls the official DimABSA2026 evaluation script.

Provides a convenient interface for evaluating predictions against gold data
using the official cF1 metric for Task 3 (DimASQP).

Usage:
    python tools/evaluate_local.py \
        --pred submission/eng_restaurant_dev_pred.jsonl \
        --gold data/eng/eng_restaurant_dev.jsonl \
        --task 3
"""
import sys
import os
import json
import math
import argparse


def read_jsonl_file(file_path, task=3, data_type='pred'):
    """Read JSONL file (adapted from official evaluation script)."""
    key_name = {1: "Aspect_VA", 2: "Triplet", 3: 'Quadruplet'}
    output_key = key_name[task]
    input_key = key_name[3] if (data_type == 'gold' and task == 2) else key_name[task]

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            json_data = json.loads(line)
            entry = {
                'ID': json_data.get('ID'),
                'Text': json_data.get('Text', ''),
            }
            quadruplets = json_data.get(input_key, [])
            if data_type == 'gold' and len(quadruplets) == 0:
                quadruplets = json_data.get(output_key, [])

            parsed = []
            for quad in quadruplets:
                if not isinstance(quad, dict):
                    continue
                parsed.append({
                    'Aspect': quad.get('Aspect', 'Unknown_Aspect').lower(),
                    'Category': quad.get('Category', 'Unknown_Category').lower(),
                    'Opinion': quad.get('Opinion', 'Unknown_Opinion').lower(),
                    'VA': quad.get('VA', '5.00#5.00')
                })
            entry[output_key] = parsed
            data.append(entry)
    return data


def evaluate_predictions(gold_data, pred_data, task=3):
    """Calculate cF1 metrics (from official evaluation script)."""
    key_name = {1: "Aspect_VA", 2: "Triplet", 3: 'Quadruplet'}
    key = key_name[task]

    key_fields = ['Aspect', 'Opinion'] if task == 2 else ['Aspect', 'Opinion', 'Category']

    gold_dict = {entry['ID']: entry[key] for entry in gold_data}
    pred_dict = {entry['ID']: entry[key] for entry in pred_data}

    cTP_total = 0.0
    TP_cat = 0
    FP_cat = 0
    FN_cat = 0

    all_ids = set(gold_dict.keys()).union(set(pred_dict.keys()))
    for id_ in all_ids:
        gold_quads = gold_dict.get(id_, [])
        pred_quads = pred_dict.get(id_, [])
        matched_pred_num = 0

        for gold_quad in gold_quads:
            gold_key = tuple(gold_quad.get(f, '') for f in key_fields)
            all_cTP_scores = []

            for pred_quad in pred_quads:
                pred_key = tuple(pred_quad.get(f, '') for f in key_fields)
                if gold_key == pred_key:
                    try:
                        gold_v, gold_a = map(float, gold_quad['VA'].split('#'))
                        pred_v, pred_a = map(float, pred_quad['VA'].split('#'))
                    except ValueError:
                        continue

                    if pred_v < 1.0 or pred_v > 9.0 or pred_a < 1.0 or pred_a > 9.0:
                        all_cTP_scores.append(0)
                        continue

                    va_euclid = math.sqrt((pred_v - gold_v) ** 2 + (pred_a - gold_a) ** 2)
                    D_max = math.sqrt(128)
                    cTP_t = max(0.0, 1.0 - (va_euclid / D_max))
                    all_cTP_scores.append(cTP_t)

            if len(all_cTP_scores) > 1:
                FN_cat += 1
            elif len(all_cTP_scores) == 1:
                matched_pred_num += 1
                TP_cat += 1
                cTP_total += all_cTP_scores[0]
            else:
                FN_cat += 1

        FP_cat += (len(pred_quads) - matched_pred_num)

    cPrecision = cTP_total / (TP_cat + FP_cat) if (TP_cat + FP_cat) > 0 else 0.0
    cRecall = cTP_total / (TP_cat + FN_cat) if (TP_cat + FN_cat) > 0 else 0.0
    cF1 = 2 * cPrecision * cRecall / (cPrecision + cRecall) if (cPrecision + cRecall) > 0 else 0.0

    return {
        'TP': TP_cat,
        'cTP': cTP_total,
        'FP': FP_cat,
        'FN': FN_cat,
        'cPrecision': cPrecision,
        'cRecall': cRecall,
        'cF1': cF1,
        'total_gold': TP_cat + FN_cat,
        'total_pred': TP_cat + FP_cat,
    }


def main():
    parser = argparse.ArgumentParser(description="Local DimASQP evaluation")
    parser.add_argument("--pred", required=True, type=str, help="Prediction JSONL file")
    parser.add_argument("--gold", required=True, type=str, help="Gold JSONL file")
    parser.add_argument("--task", type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()

    gold_data = read_jsonl_file(args.gold, task=args.task, data_type='gold')
    pred_data = read_jsonl_file(args.pred, task=args.task, data_type='pred')

    print(f"Gold entries: {len(gold_data)}")
    print(f"Pred entries: {len(pred_data)}")

    results = evaluate_predictions(gold_data, pred_data, task=args.task)

    print(f"\n{'=' * 40}")
    print(f"Task {args.task} Evaluation Results")
    print(f"{'=' * 40}")
    print(f"TP (categorical): {results['TP']}")
    print(f"cTP (continuous):  {results['cTP']:.4f}")
    print(f"FP:                {results['FP']}")
    print(f"FN:                {results['FN']}")
    print(f"Total gold quads:  {results['total_gold']}")
    print(f"Total pred quads:  {results['total_pred']}")
    print(f"cPrecision:        {results['cPrecision']:.4f}")
    print(f"cRecall:           {results['cRecall']:.4f}")
    print(f"cF1:               {results['cF1']:.4f}")

    # Also compute triplet-only F1 (ignoring VA)
    triplet_precision = results['TP'] / results['total_pred'] if results['total_pred'] > 0 else 0
    triplet_recall = results['TP'] / results['total_gold'] if results['total_gold'] > 0 else 0
    triplet_f1 = 2 * triplet_precision * triplet_recall / (triplet_precision + triplet_recall) if (triplet_precision + triplet_recall) > 0 else 0
    print(f"\nTriplet-only (no VA):")
    print(f"  Precision: {triplet_precision:.4f}")
    print(f"  Recall:    {triplet_recall:.4f}")
    print(f"  F1:        {triplet_f1:.4f}")

    if results['TP'] > 0:
        avg_ctp = results['cTP'] / results['TP']
        print(f"\nAvg cTP per matched quad: {avg_ctp:.4f}")
        print(f"  (VA discount factor: {avg_ctp:.1%} of perfect)")

    return results


if __name__ == "__main__":
    main()
