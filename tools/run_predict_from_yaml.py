# -*- coding: utf-8 -*-
"""
Run predict from configs/best_params.yaml + target data.yaml

Usage:
  python tools/run_predict_from_yaml.py --target eng-Laptop
"""

import os
import sys
import yaml
import argparse
import subprocess
from copy import deepcopy


def _read_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: dict, upd: dict) -> dict:
    out = deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _absnorm(p: str) -> str:
    return os.path.normpath(p)


def _read_data_yaml(path: str) -> dict:
    d = _read_yaml(path)
    for key in ["categories"]:
        if key not in d:
            raise ValueError(f"data.yaml missing '{key}': {path}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="e.g., eng-Laptop / eng-Restaurant")
    ap.add_argument("--best_yaml", default="configs/best_params.yaml")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "valid", "train"],
                    help="which split to run predict on (default dev if exists)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = _read_yaml(args.best_yaml)
    defaults = cfg.get("defaults", {})
    targets = cfg.get("targets", {})
    if args.target not in targets:
        raise KeyError(f"target not found in {args.best_yaml}: {args.target}")

    tcfg = targets[args.target] or {}
    paths = tcfg.get("paths", {}) or {}
    best = tcfg.get("best", {}) or {}

    data_yaml = paths.get("data_yaml")
    if not data_yaml:
        raise ValueError(f"targets.{args.target}.paths.data_yaml is required")
    dy = _read_data_yaml(data_yaml)

    eff_pred = _deep_update(defaults.get("predict", {}) or {}, best.get("predict_overrides", {}) or {})
    eff_paths = _deep_update(defaults.get("paths", {}) or {}, paths or {})

    model_name = eff_paths.get("model_name", "./deberta-v3-base")
    max_len = int(eff_paths.get("max_len", 256))

    ckpt = eff_paths.get("ckpt")
    if not ckpt:
        raise ValueError(f"targets.{args.target}.paths.ckpt is required (train first or set it)")

    # pick input by split
    split_key = args.split
    if split_key == "dev" and ("dev" not in dy):
        # fallback
        split_key = "test" if "test" in dy else ("valid" if "valid" in dy else "train")
    input_path = dy.get(split_key)
    if not input_path:
        raise ValueError(f"data.yaml has no key '{split_key}'. Found keys: {list(dy.keys())}")

    train_stats = dy.get("train_all") or dy.get("train")  # priors/statistics
    if not train_stats:
        raise ValueError("data.yaml must have train_all or train for --train_stats")

    categories_path = dy["categories"]

    out_path = eff_paths.get("pred_output")
    if not out_path:
        out_path = os.path.join("output", "preds", f"{args.target}_{split_key}.jsonl")

    cmd = [
        sys.executable, "predict.py",
        "--input", _absnorm(input_path),
        "--train_stats", _absnorm(train_stats),
        "--ckpt", _absnorm(ckpt),
        "--model_name", str(model_name),
        "--max_len", str(max_len),
        "--categories_path", _absnorm(categories_path),
        "--output", _absnorm(out_path),
    ]

    # bool flags (predict.py uses action="store_true")
    bool_flags = [
        ("apostrophe_norm", "--apostrophe_norm"),
        ("no_pair2cat_when_op_null", "--no_pair2cat_when_op_null"),
        ("dedup_by_aspect", "--dedup_by_aspect"),
        ("refine_span", "--refine_span"),
        ("diag", "--diag"),
    ]
    for k, flag in bool_flags:
        if bool(eff_pred.get(k, False)):
            cmd.append(flag)

    # scalar args
    scalar_keys = [
        "batch", "label_pattern",
        "thr_aux", "topk_aux", "max_span_len",
        "thr_rel", "topk_rel", "max_pair_dist",
        "max_quads", "min_score", "null_thr_o",
        "va_stat", "cat_case",
        "span_expand", "span_min_gain",
        "cat_source", "cat_head_min_conf", "cat_head_batch",
    ]
    for k in scalar_keys:
        if k in eff_pred and eff_pred[k] is not None:
            cmd.extend([f"--{k}", str(eff_pred[k])])

    print("[run_predict_from_yaml] target =", args.target)
    print("[run_predict_from_yaml] data_yaml =", _absnorm(data_yaml))
    print("[run_predict_from_yaml] split =", split_key)
    print("[run_predict_from_yaml] output =", _absnorm(out_path))
    print("[run_predict_from_yaml] cmd:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"predict.py failed with returncode={r.returncode}")

    print("[run_predict_from_yaml] done.")


if __name__ == "__main__":
    main()
