"""
Extended VA evaluation: continuous metrics (MAE, PCC, L2) and discrete F1 metrics.

Two evaluation modes:
  matched  - compute VA metrics only on structurally-matched (Aspect/Category/Opinion) quads
  (oracle mode is not supported without retraining; matched mode is used throughout)

Usage (single model):
  python tools/eval_va_extended.py \
      --model_path output/eng_restaurant_..._seed42_... \
      --test_data data/eng/eng_restaurant_dev.txt \
      --sidecar data/eng/eng_restaurant_dev_sidecar.json \
      --gold data/eng/eng_restaurant_dev.jsonl

Usage (batch, multiple models, outputs CSV):
  python tools/eval_va_extended.py \
      --model_paths output/dir1 output/dir2 output/dir3 \
      --test_data data/eng/eng_restaurant_dev.txt \
      --sidecar data/eng/eng_restaurant_dev_sidecar.json \
      --gold data/eng/eng_restaurant_dev.jsonl \
      --output results/va_extended_restaurant.csv
"""

import argparse
import csv
import json
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.threshold_sweep import (
    extract_raw_logits,
    decode_at_threshold,
    preds_to_submission,
)
from tools.evaluate_local import read_jsonl_file, evaluate_predictions
from tools.generate_submission import load_sidecar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_va(va_str):
    """Parse 'V.VV#A.AA' string → (float v, float a), clipped to [1, 9]."""
    try:
        parts = str(va_str).strip().split("#")
        v = float(parts[0])
        a = float(parts[1])
        v = max(1.0, min(9.0, v))
        a = max(1.0, min(9.0, a))
        return v, a
    except Exception:
        return None, None


def _safe_pcc(xs, ys):
    """Pearson correlation with safety guards (constant array / < 2 samples)."""
    if len(xs) < 2:
        return float("nan")
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    if np.std(xs) < 1e-9 or np.std(ys) < 1e-9:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _round_va(v, a):
    """Round V/A to nearest integer, clipped to [1, 9]."""
    rv = int(max(1, min(9, round(v))))
    ra = int(max(1, min(9, round(a))))
    return rv, ra


# ---------------------------------------------------------------------------
# Core: build matched pairs from prediction JSONL and gold JSONL
# ---------------------------------------------------------------------------

def build_matched_pairs(pred_data, gold_data):
    """
    For each sentence, match pred quads to gold quads on (Aspect, Category, Opinion)
    — case-insensitive, consistent with the official scorer.

    Returns list of dicts:
        {pred_v, pred_a, gold_v, gold_a}
    """
    gold_by_id = {d["ID"]: d for d in gold_data}
    pairs = []

    for pred_entry in pred_data:
        eid = pred_entry["ID"]
        gold_entry = gold_by_id.get(eid)
        if gold_entry is None:
            continue

        # Build gold lookup: (aspect, category, opinion) -> (v, a)
        gold_lookup = {}
        for gq in gold_entry.get("Quadruplet", []):
            key = (
                str(gq.get("Aspect", "")).lower().strip(),
                str(gq.get("Category", "")).lower().strip(),
                str(gq.get("Opinion", "")).lower().strip(),
            )
            gv, ga = _parse_va(gq.get("VA", "5#5"))
            if gv is not None:
                gold_lookup[key] = (gv, ga)

        for pq in pred_entry.get("Quadruplet", []):
            key = (
                str(pq.get("Aspect", "")).lower().strip(),
                str(pq.get("Category", "")).lower().strip(),
                str(pq.get("Opinion", "")).lower().strip(),
            )
            if key not in gold_lookup:
                continue
            pv, pa = _parse_va(pq.get("VA", "5#5"))
            gv, ga = gold_lookup[key]
            if pv is None or gv is None:
                continue
            pairs.append({"pred_v": pv, "pred_a": pa, "gold_v": gv, "gold_a": ga})

    return pairs


# ---------------------------------------------------------------------------
# A. Continuous metrics
# ---------------------------------------------------------------------------

def compute_continuous_metrics(pairs):
    """
    Compute V-MAE, A-MAE, V-PCC, A-PCC, Joint-L2 on matched pairs.
    Returns dict of metrics, or all-nan if no pairs.
    """
    if not pairs:
        return {
            "n_matched": 0,
            "V_MAE": float("nan"),
            "A_MAE": float("nan"),
            "V_PCC": float("nan"),
            "A_PCC": float("nan"),
            "Joint_L2": float("nan"),
        }

    pv = [p["pred_v"] for p in pairs]
    pa = [p["pred_a"] for p in pairs]
    gv = [p["gold_v"] for p in pairs]
    ga = [p["gold_a"] for p in pairs]

    mae_v = float(np.mean(np.abs(np.array(pv) - np.array(gv))))
    mae_a = float(np.mean(np.abs(np.array(pa) - np.array(ga))))
    pcc_v = _safe_pcc(pv, gv)
    pcc_a = _safe_pcc(pa, ga)
    l2 = float(np.mean(np.sqrt((np.array(pv) - np.array(gv)) ** 2 +
                               (np.array(pa) - np.array(ga)) ** 2)))

    return {
        "n_matched": len(pairs),
        "V_MAE": round(mae_v, 4),
        "A_MAE": round(mae_a, 4),
        "V_PCC": round(pcc_v, 4) if not math.isnan(pcc_v) else float("nan"),
        "A_PCC": round(pcc_a, 4) if not math.isnan(pcc_a) else float("nan"),
        "Joint_L2": round(l2, 4),
    }


# ---------------------------------------------------------------------------
# B. Discrete F1 metrics
# ---------------------------------------------------------------------------

def compute_discrete_f1(pred_data, gold_data):
    """
    Compute V-Quad-F1, A-Quad-F1, VA-Quad-F1.

    Match rule (for each sentence):
      V-Quad: (Aspect, Category, Opinion, rounded_V) match — ignoring A
      A-Quad: (Aspect, Category, Opinion, rounded_A) match — ignoring V
      VA-Quad: (Aspect, Category, Opinion, rounded_V, rounded_A) match

    Rounding: round() to nearest int, clipped to [1, 9].
    """
    gold_by_id = {d["ID"]: d for d in gold_data}

    tp_v = fp_v = fn_v = 0
    tp_a = fp_a = fn_a = 0
    tp_va = fp_va = fn_va = 0

    for pred_entry in pred_data:
        eid = pred_entry["ID"]
        gold_entry = gold_by_id.get(eid)
        if gold_entry is None:
            continue

        def make_keys(quads, include_v, include_a):
            keys = set()
            for q in quads:
                asp = str(q.get("Aspect", "")).lower().strip()
                cat = str(q.get("Category", "")).lower().strip()
                opi = str(q.get("Opinion", "")).lower().strip()
                pv, pa = _parse_va(q.get("VA", "5#5"))
                if pv is None:
                    continue
                rv, ra = _round_va(pv, pa)
                key = [asp, cat, opi]
                if include_v:
                    key.append(rv)
                if include_a:
                    key.append(ra)
                keys.add(tuple(key))
            return keys

        gold_quads = gold_entry.get("Quadruplet", [])
        pred_quads = pred_entry.get("Quadruplet", [])

        for inc_v, inc_a, tp_ref in [
            (True, False, "v"),
            (False, True, "a"),
            (True, True, "va"),
        ]:
            g_keys = make_keys(gold_quads, inc_v, inc_a)
            p_keys = make_keys(pred_quads, inc_v, inc_a)
            matched = len(g_keys & p_keys)
            if tp_ref == "v":
                tp_v += matched
                fp_v += len(p_keys) - matched
                fn_v += len(g_keys) - matched
            elif tp_ref == "a":
                tp_a += matched
                fp_a += len(p_keys) - matched
                fn_a += len(g_keys) - matched
            else:
                tp_va += matched
                fp_va += len(p_keys) - matched
                fn_va += len(g_keys) - matched

    def f1(tp, fp, fn):
        if tp + fp == 0 or tp + fn == 0:
            return float("nan")
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if p + r == 0:
            return 0.0
        return round(2 * p * r / (p + r), 4)

    return {
        "V_Quad_F1":  f1(tp_v,  fp_v,  fn_v),
        "A_Quad_F1":  f1(tp_a,  fp_a,  fn_a),
        "VA_Quad_F1": f1(tp_va, fp_va, fn_va),
    }


# ---------------------------------------------------------------------------
# Single model evaluation
# ---------------------------------------------------------------------------

def eval_single_model(model_path, test_data, sidecar_path, gold_path,
                      thresholds=None, verbose=True):
    """
    Run threshold sweep, find best threshold, compute all extended metrics.
    Returns dict with all metrics.
    """
    if thresholds is None:
        thresholds = [-2.0, -1.5, -1.0, -0.5, -0.3, -0.1, 0.0, 0.3, 0.5, 1.0]

    import json as _json

    if verbose:
        print(f"\n[eval] {os.path.basename(model_path)}")

    dataset, raw_matrices, va_preds, hs_list, training_args, model = \
        extract_raw_logits(model_path, test_data)
    sidecar_data = load_sidecar(sidecar_path)
    gold_data = read_jsonl_file(gold_path, task=3, data_type="gold")

    va_mode = training_args.get("va_mode", "position")
    seed = training_args.get("seed", "?")
    task_domain = training_args.get("task_domain", "?")

    # Sweep to find best threshold
    best_thresh = None
    best_cf1 = -1.0
    best_metrics_main = None
    best_submissions = None

    os.makedirs("submission", exist_ok=True)

    for thr in thresholds:
        all_preds = decode_at_threshold(dataset, raw_matrices, va_preds, hs_list,
                                        thr, training_args, model)
        submissions = preds_to_submission(all_preds, sidecar_data)

        import tempfile as _tmp
        with _tmp.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False,
                                     encoding="utf-8", dir="submission") as tf:
            tmp_path = tf.name
            for entry in submissions:
                tf.write(_json.dumps(entry, ensure_ascii=False) + "\n")

        pred_data_tmp = read_jsonl_file(tmp_path, task=3, data_type="pred")
        metrics_tmp = evaluate_predictions(gold_data, pred_data_tmp, task=3)
        try:
            os.remove(tmp_path)
        except OSError:
            pass  # Windows file lock; file will be cleaned up later

        if metrics_tmp["cF1"] > best_cf1:
            best_cf1 = metrics_tmp["cF1"]
            best_thresh = thr
            best_metrics_main = metrics_tmp
            best_submissions = submissions

    # Save best-threshold predictions permanently for inspection
    best_pred_path = os.path.join(model_path, "best_thresh_preds.jsonl")
    with open(best_pred_path, "w", encoding="utf-8") as f:
        for entry in best_submissions:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")

    pred_data_best = read_jsonl_file(best_pred_path, task=3, data_type="pred")

    # Extended metrics
    pairs = build_matched_pairs(pred_data_best, gold_data)
    cont_metrics = compute_continuous_metrics(pairs)
    disc_metrics = compute_discrete_f1(pred_data_best, gold_data)

    va_pct = (best_metrics_main["cTP"] / best_metrics_main["TP"] * 100) \
        if best_metrics_main["TP"] > 0 else 0.0

    result = {
        "domain":     task_domain,
        "va_mode":    va_mode,
        "seed":       seed,
        "best_thresh": best_thresh,
        "cF1":        round(best_cf1, 4),
        "cPrec":      round(best_metrics_main["cPrecision"], 4),
        "cRec":       round(best_metrics_main["cRecall"], 4),
        "VA_pct":     round(va_pct, 2),
        **cont_metrics,
        **disc_metrics,
    }

    if verbose:
        print(f"  va_mode={va_mode}  seed={seed}  thresh={best_thresh}  cF1={best_cf1:.4f}")
        print(f"  V-MAE={cont_metrics['V_MAE']:.4f}  A-MAE={cont_metrics['A_MAE']:.4f}  "
              f"L2={cont_metrics['Joint_L2']:.4f}")
        print(f"  V-PCC={cont_metrics['V_PCC']:.4f}  A-PCC={cont_metrics['A_PCC']:.4f}")
        print(f"  V-Quad-F1={disc_metrics['V_Quad_F1']:.4f}  "
              f"A-Quad-F1={disc_metrics['A_Quad_F1']:.4f}  "
              f"VA-Quad-F1={disc_metrics['VA_Quad_F1']:.4f}")

    return result


# ---------------------------------------------------------------------------
# Batch summary
# ---------------------------------------------------------------------------

def summarize_results(results, output_csv=None):
    """Print structured summary table and optionally save CSV."""
    import statistics

    # Group by (domain, va_mode)
    groups = {}
    for r in results:
        k = (r["domain"], r["va_mode"])
        groups.setdefault(k, []).append(r)

    cont_cols = ["V_MAE", "A_MAE", "Joint_L2", "V_PCC", "A_PCC"]
    disc_cols = ["V_Quad_F1", "A_Quad_F1", "VA_Quad_F1"]
    main_cols = ["cF1", "cPrec", "cRec", "VA_pct"]

    def fmt(vs):
        vs = [v for v in vs if not (isinstance(v, float) and math.isnan(v))]
        if not vs:
            return "  —  "
        if len(vs) == 1:
            return f"{vs[0]:.4f}"
        return f"{statistics.mean(vs):.4f}±{statistics.stdev(vs):.4f}"

    print("\n" + "=" * 100)
    print("  EXTENDED VA EVALUATION SUMMARY")
    print("=" * 100)

    for domain in sorted(set(r["domain"] for r in results)):
        print(f"\n  Domain: {domain}")
        print(f"  {'Method':<18} {'Seeds':<8} {'cF1':>7} {'V-MAE':>7} {'A-MAE':>7} "
              f"{'L2':>7} {'V-PCC':>7} {'A-PCC':>7} {'V-QF1':>7} {'A-QF1':>7} {'VA-QF1':>7}")
        print("  " + "-" * 98)

        for va_mode in ["position", "span_pair", "opinion_guided"]:
            key = (domain, va_mode)
            if key not in groups:
                continue
            grp = groups[key]
            seeds = [str(r["seed"]) for r in grp]
            row = [
                fmt([r[c] for r in grp])
                for c in ["cF1", "V_MAE", "A_MAE", "Joint_L2", "V_PCC", "A_PCC",
                           "V_Quad_F1", "A_Quad_F1", "VA_Quad_F1"]
            ]
            label = va_mode.replace("opinion_guided", "OG").replace(
                "span_pair", "SP").replace("position", "Pos")
            print(f"  {label:<18} {','.join(seeds):<8} " +
                  "  ".join(f"{v:>9}" for v in row))

    print("=" * 100)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
        fieldnames = ["domain", "va_mode", "seed", "best_thresh",
                      "cF1", "cPrec", "cRec", "VA_pct",
                      "n_matched", "V_MAE", "A_MAE", "Joint_L2", "V_PCC", "A_PCC",
                      "V_Quad_F1", "A_Quad_F1", "VA_Quad_F1"]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"\n  CSV saved → {output_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extended VA evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_path", type=str,
                       help="Single model directory")
    group.add_argument("--model_paths", nargs="+",
                       help="Multiple model directories (batch mode)")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", default=None,
                        help="Output CSV path (optional, batch mode)")
    parser.add_argument("--thresholds", default="-2.0,-1.5,-1.0,-0.5,-0.3,-0.1,0.0,0.3,0.5,1.0")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]

    model_paths = [args.model_path] if args.model_path else args.model_paths

    results = []
    for mp in model_paths:
        r = eval_single_model(mp, args.test_data, args.sidecar, args.gold,
                              thresholds=thresholds, verbose=True)
        results.append(r)

    summarize_results(results, output_csv=args.output)


if __name__ == "__main__":
    main()
