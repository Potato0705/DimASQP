# -*- coding: utf-8 -*-
"""
Sweep wrapper for DimASQP / SemEval Task3.

What it does
- Runs predict.py with different decoding/threshold hyperparameters (post-processing).
- Runs official metrics_subtask_1_2_3.py on each prediction file.
- Logs cF1 and saves a CSV + best_params.json (+ best_pred.jsonl).

Why you may need this
- On Windows, subprocess output is often NOT UTF-8 (e.g., CP936/GBK). This script captures bytes and
  decodes safely (utf-8 -> gbk fallback), preventing UnicodeDecodeError during long sweeps.

Typical usage (one-liner):
python sweep_predict.py --predict_py predict.py --metrics_py metrics_subtask_1_2_3.py --input ./output/valid_gold_task3.jsonl --gold ./output/valid_gold_task3.jsonl --train_stats ./output/train_gold_task3.jsonl --ckpt ./output/best_model.pt --model_name ./deberta-v3-base --out_dir ./output/sweep_small --max_len 256 --batch 8 --thr_rel_list 0.10,0.12,0.14 --null_thr_o_list 0.08,0.10,0.12 --max_quads_list 1,2 --min_score_list 0.8,1.0 --refine_span_list 0,1 --max_runs 0
"""

import os
import sys
import json
import time
import random
import argparse
import subprocess
import itertools
import re
import ast
from typing import Any, Dict, List, Tuple


FINAL_RE = re.compile(r"^Final Results:\s*(\{.*\})\s*$")


def _safe_decode(b: bytes) -> str:
    """
    Robust decode for Windows console output.
    Prefer UTF-8; fallback to GBK; always replace invalid bytes.
    """
    if b is None:
        return ""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return b.decode("gbk", errors="replace")
        except Exception:
            return b.decode("utf-8", errors="replace")


def run_cmd(args: List[str], cwd: str = None) -> Tuple[int, str]:
    """
    Run a command and return (returncode, combined_output).
    Capture BYTES and decode safely to avoid UnicodeDecodeError.
    """
    p = subprocess.run(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False
    )
    out = _safe_decode(p.stdout)
    return p.returncode, out


def parse_final_results(output: str) -> Dict[str, Any]:
    """
    Parse metrics_subtask_1_2_3.py output.
    It prints a line like: Final Results: {'TP':..., 'FP':..., 'FN':..., 'cF1':...}
    We take the LAST occurrence in case something prints twice.
    """
    m = None
    for line in output.splitlines():
        mm = FINAL_RE.search(line.strip())
        if mm:
            m = mm
    if not m:
        raise RuntimeError("Cannot find 'Final Results: {...}' in metrics output.")
    d = ast.literal_eval(m.group(1))
    if not isinstance(d, dict):
        raise RuntimeError("Parsed Final Results is not a dict.")
    return d


def parse_csv_list(s: str, cast=float) -> List[Any]:
    """
    Parse comma-separated list. Accepts single value too.
    Example: '0.10,0.12' -> [0.10, 0.12]
    """
    s = str(s).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [cast(p) for p in parts]


def parse_int_list(s: str) -> List[int]:
    return parse_csv_list(s, cast=int)


def parse_float_list(s: str) -> List[float]:
    return parse_csv_list(s, cast=float)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_csv(path: str, header: List[str], row: Dict[str, Any], write_header_if_new: bool = True) -> None:
    is_new = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    with open(path, "a", encoding="utf-8") as f:
        if is_new and write_header_if_new:
            f.write(",".join(header) + "\n")
        vals = []
        for k in header:
            v = row.get(k, "")
            if isinstance(v, str):
                v = v.replace("\n", " ").replace("\r", " ").replace(",", " ")
            vals.append(str(v))
        f.write(",".join(vals) + "\n")


def build_grid(args) -> List[Dict[str, Any]]:
    grid = []
    for (thr_aux, thr_rel, null_thr_o, topk_aux, topk_rel, max_span_len,
         max_pair_dist, max_quads, min_score, refine_span) in itertools.product(
        args.thr_aux_list,
        args.thr_rel_list,
        args.null_thr_o_list,
        args.topk_aux_list,
        args.topk_rel_list,
        args.max_span_len_list,
        args.max_pair_dist_list,
        args.max_quads_list,
        args.min_score_list,
        args.refine_span_list,
    ):
        grid.append({
            "thr_aux": float(thr_aux),
            "thr_rel": float(thr_rel),
            "null_thr_o": float(null_thr_o),
            "topk_aux": int(topk_aux),
            "topk_rel": int(topk_rel),
            "max_span_len": int(max_span_len),
            "max_pair_dist": int(max_pair_dist),
            "max_quads": int(max_quads),
            "min_score": float(min_score),
            "refine_span": int(refine_span),
        })
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict_py", default="predict.py", help="path to predict.py")
    ap.add_argument("--metrics_py", default="metrics_subtask_1_2_3.py", help="path to official metrics script")

    ap.add_argument("--input", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--train_stats", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", required=True)

    ap.add_argument("--label_pattern", type=str, default="sentiment_dim", choices=["raw", "sentiment", "sentiment_dim"])
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)

    ap.add_argument("--out_dir", default="./output/sweep")
    ap.add_argument("--seed", type=int, default=42)

    # Lists (comma-separated)
    ap.add_argument("--thr_aux_list", type=str, default="0.05")
    ap.add_argument("--thr_rel_list", type=str, default="0.12")
    ap.add_argument("--null_thr_o_list", type=str, default="0.10")
    ap.add_argument("--topk_aux_list", type=str, default="80")
    ap.add_argument("--topk_rel_list", type=str, default="800")
    ap.add_argument("--max_span_len_list", type=str, default="12")
    ap.add_argument("--max_pair_dist_list", type=str, default="120")
    ap.add_argument("--max_quads_list", type=str, default="2")
    ap.add_argument("--min_score_list", type=str, default="1.0")

    # IMPORTANT: list version for refine_span (0/1).
    ap.add_argument("--refine_span_list", type=str, default="0")

    # Fixed predict flags (not swept)
    ap.add_argument("--apostrophe_norm", action="store_true")
    ap.add_argument("--cat_case", type=str, default="upper", choices=["upper", "lower"])
    ap.add_argument("--va_stat", type=str, default="median", choices=["mode", "median", "mean"])

    # Control sweep size
    ap.add_argument("--max_runs", type=int, default=0,
                    help="0 means run full grid; >0 means randomly sample max_runs configs from the grid")

    # Optional: print predict/metrics output
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    # Parse lists
    args.thr_aux_list = parse_float_list(args.thr_aux_list)
    args.thr_rel_list = parse_float_list(args.thr_rel_list)
    args.null_thr_o_list = parse_float_list(args.null_thr_o_list)
    args.topk_aux_list = parse_int_list(args.topk_aux_list)
    args.topk_rel_list = parse_int_list(args.topk_rel_list)
    args.max_span_len_list = parse_int_list(args.max_span_len_list)
    args.max_pair_dist_list = parse_int_list(args.max_pair_dist_list)
    args.max_quads_list = parse_int_list(args.max_quads_list)
    args.min_score_list = parse_float_list(args.min_score_list)
    args.refine_span_list = parse_int_list(args.refine_span_list)

    ensure_dir(args.out_dir)

    grid = build_grid(args)
    total = len(grid)

    rnd = random.Random(args.seed)
    if args.max_runs and args.max_runs > 0 and args.max_runs < total:
        rnd.shuffle(grid)
        grid = grid[:args.max_runs]
    runs = len(grid)

    results_csv = os.path.join(args.out_dir, "sweep_results.csv")
    best_json = os.path.join(args.out_dir, "best_params.json")
    best_pred = os.path.join(args.out_dir, "best_pred.jsonl")
    tmp_pred = os.path.join(args.out_dir, "_tmp_pred.jsonl")

    header = [
        "idx", "cF1", "cPrecision", "cRecall", "TP", "FP", "FN", "time_sec",
        "thr_aux", "thr_rel", "null_thr_o", "topk_aux", "topk_rel",
        "max_span_len", "max_pair_dist", "max_quads", "min_score", "refine_span"
    ]

    best = {"cF1": -1.0, "params": None, "metrics": None}

    print(f"[Sweep] grid_size={total} runs={runs} out_dir={args.out_dir}")

    for i, params in enumerate(grid, start=1):
        t0 = time.time()

        cmd_pred = [
            sys.executable, args.predict_py,
            "--input", args.input,
            "--train_stats", args.train_stats,
            "--ckpt", args.ckpt,
            "--model_name", args.model_name,
            "--max_len", str(args.max_len),
            "--batch", str(args.batch),
            "--label_pattern", args.label_pattern,
            "--thr_aux", str(params["thr_aux"]),
            "--topk_aux", str(params["topk_aux"]),
            "--max_span_len", str(params["max_span_len"]),
            "--thr_rel", str(params["thr_rel"]),
            "--topk_rel", str(params["topk_rel"]),
            "--max_pair_dist", str(params["max_pair_dist"]),
            "--max_quads", str(params["max_quads"]),
            "--min_score", str(params["min_score"]),
            "--null_thr_o", str(params["null_thr_o"]),
            "--va_stat", args.va_stat,
            "--cat_case", args.cat_case,
            "--output", tmp_pred,
        ]
        if params["refine_span"] == 1:
            cmd_pred.append("--refine_span")
        if args.apostrophe_norm:
            cmd_pred.append("--apostrophe_norm")

        rc, out_pred = run_cmd(cmd_pred)
        if args.verbose:
            print(out_pred)
        if rc != 0:
            row = {"idx": i, "cF1": -1.0, "time_sec": round(time.time() - t0, 2), **params}
            append_csv(results_csv, header, row)
            print(f"[{i}/{runs}] FAILED predict rc={rc} params={params}")
            continue

        cmd_met = [
            sys.executable, args.metrics_py,
            "-g", args.gold,
            "-p", tmp_pred,
            "-t", "3"
        ]
        rc2, out_met = run_cmd(cmd_met)
        if args.verbose:
            print(out_met)
        if rc2 != 0:
            row = {"idx": i, "cF1": -1.0, "time_sec": round(time.time() - t0, 2), **params}
            append_csv(results_csv, header, row)
            print(f"[{i}/{runs}] FAILED metrics rc={rc2} params={params}")
            continue

        met = parse_final_results(out_met)
        cF1 = float(met.get("cF1", -1.0))
        row = {
            "idx": i,
            "cF1": cF1,
            "cPrecision": met.get("cPrecision", ""),
            "cRecall": met.get("cRecall", ""),
            "TP": met.get("TP", ""),
            "FP": met.get("FP", ""),
            "FN": met.get("FN", ""),
            "time_sec": round(time.time() - t0, 2),
            **params
        }
        append_csv(results_csv, header, row)

        if cF1 > best["cF1"]:
            best["cF1"] = cF1
            best["params"] = params
            best["metrics"] = met
            try:
                import shutil
                shutil.copyfile(tmp_pred, best_pred)
            except Exception:
                pass
            save_json(best_json, best)

        print(f"[{i}/{runs}] cF1={cF1:.4f} params={params} time={row['time_sec']}s")

    print(f"[Done] best_cF1={best['cF1']:.4f} best_params={best['params']}")
    print(f"[Saved] {results_csv}")
    if best["params"] is not None:
        print(f"[Saved] {best_json}")
        print(f"[Saved] {best_pred}")


if __name__ == "__main__":
    main()
