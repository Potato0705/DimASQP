"""
VA-specific metrics analysis: MAE_V, MAE_A, VA Euclidean, extraction-only F1.

Runs each model at its optimal threshold, matches pred↔gold quads,
and computes detailed VA metrics separated from extraction performance.

Usage:
    python tools/va_analysis.py \
        --model_paths dir1 dir2 dir3 ... \
        --test_data data/eng/eng_restaurant_dev.txt \
        --sidecar data/eng/eng_restaurant_dev_sidecar.json \
        --gold data/eng/eng_restaurant_dev.jsonl
"""
import json
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.threshold_sweep import extract_raw_logits, decode_at_threshold, preds_to_submission
from tools.evaluate_local import read_jsonl_file
from tools.generate_submission import load_sidecar


def compute_va_metrics(gold_data_list, pred_data_list):
    """Match pred to gold quads and compute detailed VA metrics.

    Returns dict with: TP, FP, FN, F1 (extraction-only),
    cF1, VA%, MAE_V, MAE_A, VA_euclid_mean, and per-quad details.
    """
    key_fields = ['Aspect', 'Opinion', 'Category']
    quad_key = 'Quadruplet'
    D_max = math.sqrt(128)

    # Convert list format to dict keyed by ID
    gold_data = {e['ID']: e[quad_key] for e in gold_data_list}
    pred_data = {e['ID']: e[quad_key] for e in pred_data_list}

    TP = FP_total = FN_total = 0
    cTP_total = 0.0
    v_errors = []
    a_errors = []
    euclid_errors = []

    all_ids = set(list(gold_data.keys()) + list(pred_data.keys()))
    for id_ in all_ids:
        gold_quads = gold_data.get(id_, [])
        pred_quads = pred_data.get(id_, [])
        matched_pred = 0

        for gq in gold_quads:
            gk = tuple(gq.get(f, '') for f in key_fields)
            best_ctp = None
            best_va_err = None

            for pq in pred_quads:
                pk = tuple(pq.get(f, '') for f in key_fields)
                if gk != pk:
                    continue
                try:
                    gv, ga = [float(x) for x in gq['VA'].split('#')]
                    pv, pa = [float(x) for x in pq['VA'].split('#')]
                except (ValueError, KeyError):
                    continue
                if pv < 1 or pv > 9 or pa < 1 or pa > 9:
                    continue
                dist = math.sqrt((pv - gv)**2 + (pa - ga)**2)
                ctp = max(0.0, 1.0 - dist / D_max)
                if best_ctp is None or ctp > best_ctp:
                    best_ctp = ctp
                    best_va_err = (abs(pv - gv), abs(pa - ga), dist)

            if best_ctp is not None:
                matched_pred += 1
                TP += 1
                cTP_total += best_ctp
                v_errors.append(best_va_err[0])
                a_errors.append(best_va_err[1])
                euclid_errors.append(best_va_err[2])
            else:
                FN_total += 1

        FP_total += len(pred_quads) - matched_pred

    # Extraction-only F1 (ignoring VA)
    prec = TP / (TP + FP_total) if (TP + FP_total) > 0 else 0
    rec = TP / (TP + FN_total) if (TP + FN_total) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    # cF1 (VA-weighted)
    cprec = cTP_total / (TP + FP_total) if (TP + FP_total) > 0 else 0
    crec = cTP_total / (TP + FN_total) if (TP + FN_total) > 0 else 0
    cf1 = 2 * cprec * crec / (cprec + crec) if (cprec + crec) > 0 else 0

    va_pct = (cTP_total / TP * 100) if TP > 0 else 0
    mae_v = np.mean(v_errors) if v_errors else 0
    mae_a = np.mean(a_errors) if a_errors else 0
    euclid_mean = np.mean(euclid_errors) if euclid_errors else 0

    return {
        'TP': TP, 'FP': FP_total, 'FN': FN_total,
        'Prec': prec, 'Rec': rec, 'F1': f1,
        'cF1': cf1, 'cPrec': cprec, 'cRec': crec,
        'VA%': va_pct, 'MAE_V': mae_v, 'MAE_A': mae_a,
        'Euclid': euclid_mean,
    }


def find_best_threshold(model_path, test_data, sidecar_path, gold_path):
    """Run threshold sweep and return (best_threshold, best_cF1, all needed objects)."""
    dataset, raw_matrices, va_preds_list, hidden_states_list, training_args, model = \
        extract_raw_logits(model_path, test_data)
    sidecar_data = load_sidecar(sidecar_path)
    gold_data = read_jsonl_file(gold_path, task=3, data_type='gold')

    thresholds = [-2.0, -1.5, -1.0, -0.5, -0.3, -0.1, 0.0, 0.3, 0.5, 1.0]
    best_cf1 = -1
    best_th = 0.0

    for th in thresholds:
        all_preds = decode_at_threshold(dataset, raw_matrices, va_preds_list,
                                        hidden_states_list, th, training_args, model)
        submissions = preds_to_submission(all_preds, sidecar_data)
        tmp_path = "submission/_tmp_va_analysis.jsonl"
        os.makedirs("submission", exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            for entry in submissions:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        pred_data = read_jsonl_file(tmp_path, task=3, data_type='pred')
        m = compute_va_metrics(gold_data, pred_data)
        if m['cF1'] > best_cf1:
            best_cf1 = m['cF1']
            best_th = th

    return best_th, best_cf1, dataset, raw_matrices, va_preds_list, hidden_states_list, \
           training_args, model, sidecar_data, gold_data


def analyze_model(model_path, test_data, sidecar_path, gold_path):
    """Full analysis for one model: find best threshold, compute all VA metrics."""
    best_th, _, dataset, raw_matrices, va_preds_list, hidden_states_list, \
        training_args, model, sidecar_data, gold_data = \
        find_best_threshold(model_path, test_data, sidecar_path, gold_path)

    # Decode at best threshold
    all_preds = decode_at_threshold(dataset, raw_matrices, va_preds_list,
                                    hidden_states_list, best_th, training_args, model)
    submissions = preds_to_submission(all_preds, sidecar_data)
    tmp_path = "submission/_tmp_va_analysis.jsonl"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for entry in submissions:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    pred_data = read_jsonl_file(tmp_path, task=3, data_type='pred')
    metrics = compute_va_metrics(gold_data, pred_data)
    metrics['threshold'] = best_th
    metrics['va_mode'] = training_args.get('va_mode', 'position')
    metrics['seed'] = training_args.get('seed', '?')

    # Free GPU
    del model
    import torch
    torch.cuda.empty_cache()

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs="+", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--gold", required=True)
    args = parser.parse_args()

    all_results = []
    for mp in args.model_paths:
        print(f"\n{'='*60}")
        print(f"Analyzing: {os.path.basename(mp)}")
        print(f"{'='*60}")
        m = analyze_model(mp, args.test_data, args.sidecar, args.gold)
        all_results.append(m)

    # Group by va_mode
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        groups[r['va_mode']].append(r)

    # Print detailed table
    print(f"\n{'='*80}")
    print("VA-SPECIFIC METRICS ANALYSIS")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Seed':>4} {'cF1':>7} {'F1':>7} {'VA%':>6} {'MAE_V':>6} {'MAE_A':>6} {'Euclid':>7} {'TP':>5} {'Th':>5}")
    print(f"{'-'*80}")

    for va_mode in ['position', 'span_pair', 'opinion_guided']:
        if va_mode not in groups:
            continue
        seeds = sorted(groups[va_mode], key=lambda x: x['seed'])
        for r in seeds:
            print(f"{va_mode:<20} {r['seed']:>4} {r['cF1']:>7.4f} {r['F1']:>7.4f} "
                  f"{r['VA%']:>5.1f}% {r['MAE_V']:>6.3f} {r['MAE_A']:>6.3f} "
                  f"{r['Euclid']:>7.3f} {r['TP']:>5} {r['threshold']:>5.1f}")

        # Averages
        avg_cf1 = np.mean([r['cF1'] for r in seeds])
        avg_f1 = np.mean([r['F1'] for r in seeds])
        avg_va = np.mean([r['VA%'] for r in seeds])
        avg_mae_v = np.mean([r['MAE_V'] for r in seeds])
        avg_mae_a = np.mean([r['MAE_A'] for r in seeds])
        avg_euclid = np.mean([r['Euclid'] for r in seeds])
        std_cf1 = np.std([r['cF1'] for r in seeds])
        print(f"  {'>> AVG':<18} {'':>4} {avg_cf1:>7.4f} {avg_f1:>7.4f} "
              f"{avg_va:>5.1f}% {avg_mae_v:>6.3f} {avg_mae_a:>6.3f} "
              f"{avg_euclid:>7.3f}   (±{std_cf1:.4f})")
        print()

    # Summary comparison table
    print(f"\n{'='*80}")
    print("SUMMARY: VA Quality Comparison")
    print(f"{'='*80}")
    print(f"{'Method':<22} {'Avg cF1':>10} {'Avg F1':>10} {'Avg VA%':>8} "
          f"{'MAE_V':>7} {'MAE_A':>7} {'Euclid':>8}")
    print(f"{'-'*80}")
    for va_mode in ['position', 'span_pair', 'opinion_guided']:
        if va_mode not in groups:
            continue
        seeds = groups[va_mode]
        avg_cf1 = np.mean([r['cF1'] for r in seeds])
        std_cf1 = np.std([r['cF1'] for r in seeds])
        avg_f1 = np.mean([r['F1'] for r in seeds])
        avg_va = np.mean([r['VA%'] for r in seeds])
        avg_mae_v = np.mean([r['MAE_V'] for r in seeds])
        avg_mae_a = np.mean([r['MAE_A'] for r in seeds])
        avg_euclid = np.mean([r['Euclid'] for r in seeds])
        label = {'position': 'Position VA', 'span_pair': 'Span-Pair VA',
                 'opinion_guided': 'Opinion-Guided VA'}[va_mode]
        print(f"{label:<22} {avg_cf1:>.4f}±{std_cf1:.4f} {avg_f1:>10.4f} {avg_va:>7.1f}% "
              f"{avg_mae_v:>7.3f} {avg_mae_a:>7.3f} {avg_euclid:>8.3f}")

    # Delta table
    if 'position' in groups:
        base = groups['position']
        base_cf1 = np.mean([r['cF1'] for r in base])
        base_f1 = np.mean([r['F1'] for r in base])
        base_va = np.mean([r['VA%'] for r in base])
        base_mae_v = np.mean([r['MAE_V'] for r in base])
        base_mae_a = np.mean([r['MAE_A'] for r in base])
        base_euclid = np.mean([r['Euclid'] for r in base])
        print(f"\n{'Δ vs Position VA':<22} {'cF1':>10} {'F1':>10} {'VA%':>8} "
              f"{'MAE_V':>7} {'MAE_A':>7} {'Euclid':>8}")
        print(f"{'-'*80}")
        for va_mode in ['span_pair', 'opinion_guided']:
            if va_mode not in groups:
                continue
            seeds = groups[va_mode]
            d_cf1 = np.mean([r['cF1'] for r in seeds]) - base_cf1
            d_f1 = np.mean([r['F1'] for r in seeds]) - base_f1
            d_va = np.mean([r['VA%'] for r in seeds]) - base_va
            d_mae_v = np.mean([r['MAE_V'] for r in seeds]) - base_mae_v
            d_mae_a = np.mean([r['MAE_A'] for r in seeds]) - base_mae_a
            d_euclid = np.mean([r['Euclid'] for r in seeds]) - base_euclid
            label = {'span_pair': 'Span-Pair VA', 'opinion_guided': 'Opinion-Guided VA'}[va_mode]
            print(f"{label:<22} {d_cf1:>+10.4f} {d_f1:>+10.4f} {d_va:>+7.1f}% "
                  f"{d_mae_v:>+7.3f} {d_mae_a:>+7.3f} {d_euclid:>+8.3f}")


if __name__ == "__main__":
    main()
