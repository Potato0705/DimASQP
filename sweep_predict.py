# -*- coding: utf-8 -*-
"""
Sweep wrapper for DimASQP / SemEval Task3.

Key improvements:
- Robust list parsing: comma-separated, space-separated, or mixed tokens.
- Resume with stronger signature (ckpt/model/label/max_len/batch + params).
- Predict args passthrough: --extra_predict_args forwards any predict.py flags.
- Unique tmp pred per run (safe, no clobbering), best_pred.jsonl saved in jobs=1.
- Metrics invocation aligned with your actual usage:
  metrics_subtask_1_2_3.py -t 3 --pred ... --gold ...

IMPORTANT COMPATIBILITY PATCH (for your latest predict.py):
- DO NOT define/forward a local flag named '--apostrophe_norm' here, because your new predict.py
  uses '--apostrophe_norm' as a boolean default (no CLI flag). Forwarding it will crash.
- Keep '--apostrophe_norm' removed from this sweep wrapper completely.
- Any other predict flags can still be forwarded through --extra_predict_args.

Recommended:
- Single GPU: keep --jobs 1.
"""

from __future__ import annotations

import os
import sys
import csv
import json
import time
import shlex
import random
import argparse
import subprocess
import itertools
import re
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

FINAL_RE = re.compile(r"^Final Results:\s*(\{.*\})\s*$")


# -------------------------
# IO / Utils
# -------------------------
def _safe_decode(b: Optional[bytes]) -> str:
    if b is None:
        return ""
    for enc in ("utf-8", "gbk"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    return b.decode("utf-8", errors="replace")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_final_results(output: str) -> Dict[str, Any]:
    """Parse official metrics output: last 'Final Results: {...}' line."""
    m = None
    for line in output.splitlines():
        mm = FINAL_RE.search(line.strip())
        if mm:
            m = mm
    if not m:
        tail = "\n".join(output.splitlines()[-30:])
        raise RuntimeError("Cannot find 'Final Results: {...}' in metrics output.\n--- tail ---\n" + tail)
    d = ast.literal_eval(m.group(1))
    if not isinstance(d, dict):
        raise RuntimeError("Parsed Final Results is not a dict.")
    return d


def run_cmd(args: List[str], quiet: bool = True, err_tail: int = 2000) -> Tuple[int, str]:
    p = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False
    )
    out = _safe_decode(p.stdout)
    if not quiet and out:
        print(out)
    if p.returncode != 0 and quiet and len(out) > err_tail:
        out = out[-err_tail:]
    return p.returncode, out


def _split_list_tokens(tokens: Optional[Sequence[str]]) -> List[str]:
    """
    Accept tokens list that may be:
      - ["0.1,0.2,0.3"]
      - ["0.1", "0.2", "0.3"]
      - ["0.1,0.2", "0.3"]
    """
    if not tokens:
        return []
    parts: List[str] = []
    for t in tokens:
        s = str(t).strip()
        if not s:
            continue
        s = s.replace(",", " ")
        parts.extend([x for x in s.split() if x])
    return parts


def parse_float_list(tokens: Optional[Sequence[str]]) -> List[float]:
    parts = _split_list_tokens(tokens)
    return [float(x) for x in parts] if parts else []


def parse_int_list(tokens: Optional[Sequence[str]]) -> List[int]:
    parts = _split_list_tokens(tokens)
    return [int(float(x)) for x in parts] if parts else []


def parse_extra_args(s: str) -> List[str]:
    """
    Parse a single string into argv-like list.
    Example: '--categories_path ./a.txt --cat_source mix --cat_head_min_conf 0.55'
    """
    s = (s or "").strip()
    if not s:
        return []
    return shlex.split(s, posix=False)


# -------------------------
# Signatures / Resume
# -------------------------
def params_signature(params: Dict[str, Any]) -> str:
    keys = [
        "thr_aux", "thr_rel", "null_thr_o", "topk_aux", "topk_rel",
        "max_span_len", "max_pair_dist", "max_quads", "min_score", "refine_span"
    ]
    return "|".join(f"{k}={params[k]}" for k in keys)


def run_signature(params: Dict[str, Any], meta: Dict[str, Any]) -> str:
    meta_keys = ["ckpt", "model_name", "label_pattern", "max_len", "batch"]
    meta_part = "|".join(f"{k}={meta.get(k,'')}" for k in meta_keys)
    return meta_part + "||" + params_signature(params)


def load_done_signatures(csv_path: Path, meta: Dict[str, Any]) -> set:
    done = set()
    if (not csv_path.exists()) or csv_path.stat().st_size == 0:
        return done

    needed = [
        "thr_aux", "thr_rel", "null_thr_o", "topk_aux", "topk_rel",
        "max_span_len", "max_pair_dist", "max_quads", "min_score", "refine_span"
    ]

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or any(k not in reader.fieldnames for k in needed):
            return done

        for row in reader:
            try:
                p = {
                    "thr_aux": float(row["thr_aux"]),
                    "thr_rel": float(row["thr_rel"]),
                    "null_thr_o": float(row["null_thr_o"]),
                    "topk_aux": int(float(row["topk_aux"])),
                    "topk_rel": int(float(row["topk_rel"])),
                    "max_span_len": int(float(row["max_span_len"])),
                    "max_pair_dist": int(float(row["max_pair_dist"])),
                    "max_quads": int(float(row["max_quads"])),
                    "min_score": float(row["min_score"]),
                    "refine_span": int(float(row["refine_span"])),
                }
                done.add(run_signature(p, meta))
            except Exception:
                continue
    return done


# -------------------------
# Grid generator
# -------------------------
def grid_iter(args) -> Iterator[Dict[str, Any]]:
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
        yield {
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
        }


def grid_size(args) -> int:
    lens = [
        len(args.thr_aux_list), len(args.thr_rel_list), len(args.null_thr_o_list),
        len(args.topk_aux_list), len(args.topk_rel_list), len(args.max_span_len_list),
        len(args.max_pair_dist_list), len(args.max_quads_list), len(args.min_score_list),
        len(args.refine_span_list),
    ]
    total = 1
    for x in lens:
        total *= max(1, x)
    return total


# -------------------------
# Worker
# -------------------------
@dataclass(frozen=True)
class RunContext:
    predict_py: str
    metrics_py: str
    input_path: str
    gold_path: str
    train_stats: str
    ckpt: str
    model_name: str
    label_pattern: str
    max_len: int
    batch: int
    cat_case: str
    va_stat: str
    extra_predict_args: Tuple[str, ...]
    quiet_subprocess: bool
    err_tail: int
    out_dir: str


def build_predict_cmd(ctx: RunContext, params: Dict[str, Any], pred_out: str) -> List[str]:
    cmd = [
        sys.executable, ctx.predict_py,
        "--input", ctx.input_path,
        "--train_stats", ctx.train_stats,
        "--ckpt", ctx.ckpt,
        "--model_name", ctx.model_name,
        "--max_len", str(ctx.max_len),
        "--batch", str(ctx.batch),
        "--label_pattern", ctx.label_pattern,
        "--thr_aux", str(params["thr_aux"]),
        "--topk_aux", str(params["topk_aux"]),
        "--max_span_len", str(params["max_span_len"]),
        "--thr_rel", str(params["thr_rel"]),
        "--topk_rel", str(params["topk_rel"]),
        "--max_pair_dist", str(params["max_pair_dist"]),
        "--max_quads", str(params["max_quads"]),
        "--min_score", str(params["min_score"]),
        "--null_thr_o", str(params["null_thr_o"]),
        "--va_stat", ctx.va_stat,
        "--cat_case", ctx.cat_case,
        "--output", pred_out,
    ]
    if params["refine_span"] == 1:
        cmd.append("--refine_span")
    if ctx.extra_predict_args:
        cmd.extend(list(ctx.extra_predict_args))
    return cmd


def build_metrics_cmd(ctx: RunContext, pred_out: str) -> List[str]:
    return [
        sys.executable, ctx.metrics_py,
        "-t", "3",
        "--pred", pred_out,
        "--gold", ctx.gold_path,
    ]


def run_one(run_id: int, params: Dict[str, Any], ctx: RunContext) -> Tuple[bool, Dict[str, Any], Optional[Dict[str, Any]], str]:
    t0 = time.time()
    pred_out = str(Path(ctx.out_dir) / f"_tmp_pred_{os.getpid()}_{run_id}.jsonl")

    rc, out_pred = run_cmd(build_predict_cmd(ctx, params, pred_out), quiet=ctx.quiet_subprocess, err_tail=ctx.err_tail)
    if rc != 0:
        row = {"idx": run_id, "cF1": -1.0, "time_sec": round(time.time() - t0, 2), **params}
        try:
            if os.path.exists(pred_out):
                os.remove(pred_out)
        except Exception:
            pass
        return False, row, None, f"[{run_id}] FAILED predict rc={rc}\n{out_pred}"

    rc2, out_met = run_cmd(build_metrics_cmd(ctx, pred_out), quiet=ctx.quiet_subprocess, err_tail=ctx.err_tail)
    if rc2 != 0:
        row = {"idx": run_id, "cF1": -1.0, "time_sec": round(time.time() - t0, 2), **params}
        try:
            if os.path.exists(pred_out):
                os.remove(pred_out)
        except Exception:
            pass
        return False, row, None, f"[{run_id}] FAILED metrics rc={rc2}\n{out_met}"

    met = parse_final_results(out_met)
    cF1 = float(met.get("cF1", -1.0))
    row = {
        "idx": run_id,
        "cF1": cF1,
        "cPrecision": met.get("cPrecision", ""),
        "cRecall": met.get("cRecall", ""),
        "TP": met.get("TP", ""),
        "FP": met.get("FP", ""),
        "FN": met.get("FN", ""),
        "time_sec": round(time.time() - t0, 2),
        **params,
    }
    met["_pred_path"] = pred_out
    return True, row, met, ""


def append_csv_row(csv_path: Path, header: List[str], row: Dict[str, Any]) -> None:
    is_new = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if is_new:
            w.writeheader()
        safe_row = {}
        for k in header:
            v = row.get(k, "")
            if isinstance(v, str):
                v = v.replace("\n", " ").replace("\r", " ").replace(",", " ")
            safe_row[k] = v
        w.writerow(safe_row)


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--predict_py", default="predict.py")
    ap.add_argument("--metrics_py", default="metrics_subtask_1_2_3.py")

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

    # Lists: allow either comma-separated string OR multiple tokens.
    ap.add_argument("--thr_aux_list", nargs="*", default=["0.05"])
    ap.add_argument("--thr_rel_list", nargs="*", default=["0.12"])
    ap.add_argument("--null_thr_o_list", nargs="*", default=["0.10"])
    ap.add_argument("--topk_aux_list", nargs="*", default=["80"])
    ap.add_argument("--topk_rel_list", nargs="*", default=["800"])
    ap.add_argument("--max_span_len_list", nargs="*", default=["12"])
    ap.add_argument("--max_pair_dist_list", nargs="*", default=["120"])
    ap.add_argument("--max_quads_list", nargs="*", default=["2"])
    ap.add_argument("--min_score_list", nargs="*", default=["1.0"])
    ap.add_argument("--refine_span_list", nargs="*", default=["0"])  # 0/1

    # Fixed predict flags (not swept)
    ap.add_argument("--cat_case", type=str, default="upper", choices=["upper", "lower"])
    ap.add_argument("--va_stat", type=str, default="median", choices=["mode", "median", "mean"])

    ap.add_argument("--extra_predict_args", type=str, default="",
                    help='Quoted string forwarded to predict.py, e.g. "--categories_path ./x.txt --cat_source mix --cat_head_min_conf 0.55"')

    ap.add_argument("--resume", action="store_true", help="skip configs already in sweep_results.csv")
    ap.add_argument("--jobs", type=int, default=1, help="number of worker processes (single GPU -> keep 1)")

    ap.add_argument("--quiet", action="store_true", help="suppress subprocess outputs (default quiet unless --verbose)")
    ap.add_argument("--verbose", action="store_true", help="print subprocess outputs (predict/metrics)")
    ap.add_argument("--print_every", type=int, default=0,
                    help="0 means only print when best improves/failures; >0 prints progress every N ok runs.")
    ap.add_argument("--err_tail", type=int, default=2000, help="show last N chars of subprocess output on failure")

    args = ap.parse_args()

    # default quiet unless verbose
    if args.verbose:
        quiet_subprocess = False
    else:
        quiet_subprocess = True

    # Parse lists
    args.thr_aux_list = parse_float_list(args.thr_aux_list) or [0.05]
    args.thr_rel_list = parse_float_list(args.thr_rel_list) or [0.12]
    args.null_thr_o_list = parse_float_list(args.null_thr_o_list) or [0.10]
    args.topk_aux_list = parse_int_list(args.topk_aux_list) or [80]
    args.topk_rel_list = parse_int_list(args.topk_rel_list) or [800]
    args.max_span_len_list = parse_int_list(args.max_span_len_list) or [12]
    args.max_pair_dist_list = parse_int_list(args.max_pair_dist_list) or [120]
    args.max_quads_list = parse_int_list(args.max_quads_list) or [2]
    args.min_score_list = parse_float_list(args.min_score_list) or [1.0]
    args.refine_span_list = parse_int_list(args.refine_span_list) or [0]

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    results_csv = out_dir / "sweep_results.csv"
    best_json = out_dir / "best_params.json"
    best_pred = out_dir / "best_pred.jsonl"

    header = [
        "idx", "cF1", "cPrecision", "cRecall", "TP", "FP", "FN", "time_sec",
        "thr_aux", "thr_rel", "null_thr_o", "topk_aux", "topk_rel",
        "max_span_len", "max_pair_dist", "max_quads", "min_score", "refine_span"
    ]

    meta_for_resume = {
        "ckpt": str(args.ckpt),
        "model_name": str(args.model_name),
        "label_pattern": str(args.label_pattern),
        "max_len": int(args.max_len),
        "batch": int(args.batch),
    }

    rnd = random.Random(args.seed)

    done = set()
    if args.resume:
        done = load_done_signatures(results_csv, meta_for_resume)

    full_size = grid_size(args)

    extra_predict = tuple(parse_extra_args(args.extra_predict_args))

    ctx = RunContext(
        predict_py=args.predict_py,
        metrics_py=args.metrics_py,
        input_path=args.input,
        gold_path=args.gold,
        train_stats=args.train_stats,
        ckpt=args.ckpt,
        model_name=args.model_name,
        label_pattern=args.label_pattern,
        max_len=args.max_len,
        batch=args.batch,
        cat_case=args.cat_case,
        va_stat=args.va_stat,
        extra_predict_args=extra_predict,
        quiet_subprocess=quiet_subprocess,
        err_tail=args.err_tail,
        out_dir=str(out_dir),
    )

    print("[Sweep] " + json.dumps({
        "grid_size_total": full_size,
        "resume": bool(args.resume),
        "jobs": int(args.jobs),
        "quiet_subprocess": bool(ctx.quiet_subprocess),
        "out_dir": str(out_dir),
        "ckpt": str(args.ckpt),
        "model_name": str(args.model_name),
        "label_pattern": str(args.label_pattern),
        "max_len": int(args.max_len),
        "batch": int(args.batch),
        "extra_predict_args": " ".join(extra_predict),
    }, ensure_ascii=False))

    # best state
    best = {"cF1": -1.0, "params": None, "metrics": None}
    if best_json.exists():
        try:
            old = json.loads(best_json.read_text(encoding="utf-8"))
            if isinstance(old, dict) and "cF1" in old:
                best = old
        except Exception:
            pass

    # Run (recommend jobs=1)
    if args.jobs and args.jobs > 1:
        # For single GPU, not recommended.
        from concurrent.futures import ProcessPoolExecutor, as_completed

        def _worker(task: Tuple[int, Dict[str, Any], RunContext]) -> Tuple[bool, Dict[str, Any], Optional[Dict[str, Any]], str]:
            rid, p, c = task
            return run_one(rid, p, c)

        run_list: List[Dict[str, Any]] = []
        for p in grid_iter(args):
            if args.resume:
                sig = run_signature(p, meta_for_resume)
                if sig in done:
                    continue
            run_list.append(p)

        tasks: List[Tuple[int, Dict[str, Any], RunContext]] = [(i, p, ctx) for i, p in enumerate(run_list, start=1)]
        done_cnt = 0

        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_worker, t) for t in tasks]
            for fut in as_completed(futs):
                ok, row, met, err = fut.result()
                done_cnt += 1
                append_csv_row(results_csv, header, row)

                if not ok:
                    print(err)
                    continue

                cF1 = float(row["cF1"])
                if cF1 > float(best.get("cF1", -1.0)):
                    best["cF1"] = cF1
                    best["params"] = {k: row[k] for k in ["thr_aux", "thr_rel", "null_thr_o", "topk_aux", "topk_rel",
                                                         "max_span_len", "max_pair_dist", "max_quads", "min_score", "refine_span"]}
                    best["metrics"] = {k: row.get(k) for k in ["cPrecision", "cRecall", "TP", "FP", "FN"]}
                    save_json(best_json, best)
                    print(f"[BEST] cF1={cF1:.6f} params={best['params']}")

    else:
        import shutil

        run_id = 0
        ok_cnt = 0

        for p in grid_iter(args):
            if args.resume:
                sig = run_signature(p, meta_for_resume)
                if sig in done:
                    continue

            run_id += 1
            ok, row, met, err = run_one(run_id, p, ctx)
            append_csv_row(results_csv, header, row)

            if not ok:
                print(err)
                continue

            ok_cnt += 1
            cF1 = float(row["cF1"])
            if cF1 > float(best.get("cF1", -1.0)):
                best["cF1"] = cF1
                best["params"] = {k: row[k] for k in ["thr_aux", "thr_rel", "null_thr_o", "topk_aux", "topk_rel",
                                                     "max_span_len", "max_pair_dist", "max_quads", "min_score", "refine_span"]}
                best["metrics"] = {k: row.get(k) for k in ["cPrecision", "cRecall", "TP", "FP", "FN"]}
                save_json(best_json, best)

                pred_path = (met or {}).get("_pred_path", "")
                if pred_path and os.path.exists(pred_path):
                    try:
                        shutil.copyfile(pred_path, str(best_pred))
                    except Exception:
                        pass

                print(f"[BEST] cF1={cF1:.6f} params={best['params']} time={row.get('time_sec','')}s")

            if args.print_every and args.print_every > 0 and (ok_cnt % args.print_every == 0):
                print(f"[Progress] ok={ok_cnt} last_idx={run_id} best_cF1={float(best.get('cF1', -1.0)):.6f}")

    print(f"[Done] best_cF1={float(best.get('cF1', -1.0)):.6f} best_params={best.get('params')}")
    print(f"[Saved] {results_csv}")
    if best.get("params") is not None:
        print(f"[Saved] {best_json}")
        if (args.jobs or 1) <= 1 and best_pred.exists():
            print(f"[Saved] {best_pred}")


if __name__ == "__main__":
    main()
