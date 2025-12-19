# -*- coding: utf-8 -*-
"""
Run training from configs/best_params.yaml + target data.yaml

Usage:
  python tools/run_train_from_yaml.py --target eng-Laptop --update_yaml
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from copy import deepcopy
from datetime import datetime, timezone, timedelta


TZ_SG = timezone(timedelta(hours=8))


def _read_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def _deep_update(base: dict, upd: dict) -> dict:
    out = deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _dict_diff(full: dict, base: dict) -> dict:
    """
    Return only keys in 'full' whose value differs from 'base'
    (recursive for dicts).
    """
    diff = {}
    for k, v in (full or {}).items():
        if k not in (base or {}):
            diff[k] = v
            continue
        bv = base[k]
        if isinstance(v, dict) and isinstance(bv, dict):
            sub = _dict_diff(v, bv)
            if sub:
                diff[k] = sub
        else:
            if v != bv:
                diff[k] = v
    return diff


def _read_data_yaml(path: str) -> dict:
    d = _read_yaml(path)
    # minimal sanity
    for key in ["train", "valid", "categories"]:
        if key not in d:
            raise ValueError(f"data.yaml missing '{key}': {path}")
    return d


def _absnorm(p: str) -> str:
    return os.path.normpath(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="e.g., eng-Laptop / eng-Restaurant")
    ap.add_argument("--best_yaml", default="configs/best_params.yaml")
    ap.add_argument("--update_yaml", action="store_true", help="write back best metrics + overrides")
    ap.add_argument("--dry_run", action="store_true", help="print cmd only, do not execute")
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

    # ----- resolve effective train config: defaults.train + best.train_overrides -----
    eff_train = _deep_update(defaults.get("train", {}) or {}, best.get("train_overrides", {}) or {})
    eff_paths = _deep_update(defaults.get("paths", {}) or {}, paths or {})

    model_name = eff_paths.get("model_name", "./deberta-v3-base")
    max_len = int(eff_paths.get("max_len", 256))

    output_dir = eff_paths.get("output_dir")
    if not output_dir:
        # safe fallback
        output_dir = os.path.join("output", "runs", args.target)

    train_path = dy["train"]
    valid_path = dy["valid"]
    categories_path = dy["categories"]

    # train.py args mapping (keys must match train.py argparse)
    cmd = [
        sys.executable, "train.py",
        "--train", _absnorm(train_path),
        "--valid", _absnorm(valid_path),
        "--model_name", str(model_name),
        "--output_dir", _absnorm(output_dir),
        "--max_len", str(max_len),
        "--categories_path", _absnorm(categories_path),
    ]

    # boolean flags
    def _add_flag(k: str, flag: str):
        if bool(eff_train.get(k, False)):
            cmd.append(flag)

    _add_flag("fp16", "--fp16")
    _add_flag("neg_include_cross", "--neg_include_cross")
    _add_flag("neg_include_random", "--neg_include_random")
    _add_flag("early_stop", "--early_stop")

    # scalar args
    scalar_keys = [
        "batch", "epochs", "lr", "weight_decay", "warmup_ratio", "seed",
        "label_pattern",
        "w_ent", "w_rel", "w_dim", "w_dim_seq", "w_sen_seq", "w_cat",
        "neg_ratio", "neg_shift", "neg_max_per_sample",
        "patience", "min_delta", "min_epochs",
        "select_by",
        "pred_batch",
        "thr_aux", "topk_aux", "max_span_len",
        "thr_rel", "topk_rel", "max_pair_dist",
        "max_quads", "min_score", "null_thr_o",
        "va_stat", "cat_case", "cat_source", "cat_head_min_conf",
    ]

    for k in scalar_keys:
        if k in eff_train and eff_train[k] is not None:
            cmd.extend([f"--{k}", str(eff_train[k])])

    print("[run_train_from_yaml] target =", args.target)
    print("[run_train_from_yaml] data_yaml =", _absnorm(data_yaml))
    print("[run_train_from_yaml] output_dir =", _absnorm(output_dir))
    print("[run_train_from_yaml] cmd:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"train.py failed with returncode={r.returncode}")

    # ----- read best metrics emitted by train.py patch -----
    best_metrics_path = os.path.join(output_dir, "best_metrics.json")
    if not os.path.exists(best_metrics_path):
        raise FileNotFoundError(
            f"best_metrics.json not found: {best_metrics_path}. "
            f"Please apply the train.py patch that writes this file."
        )

    with open(best_metrics_path, "r", encoding="utf-8") as f:
        bm = json.load(f)

    best_value = bm.get("best_cf1", None) if bm.get("select_by") == "cf1" else bm.get("best_loss", None)
    best_metric = "cf1" if bm.get("select_by") == "cf1" else "loss"

    if args.update_yaml:
        # compute overrides relative to defaults.train
        train_overrides = _dict_diff(eff_train, defaults.get("train", {}) or {})

        cfg["targets"][args.target]["best"] = cfg["targets"][args.target].get("best", {}) or {}
        cfg["targets"][args.target]["best"]["metric"] = best_metric
        cfg["targets"][args.target]["best"]["value"] = float(best_value) if best_value is not None else None
        cfg["targets"][args.target]["best"]["updated_at"] = datetime.now(TZ_SG).isoformat()
        cfg["targets"][args.target]["best"]["source"] = "train.select_by=" + str(bm.get("select_by"))
        cfg["targets"][args.target]["best"]["train_overrides"] = train_overrides

        # 写回 ckpt（以训练产出的 best_model.pt 为准）
        cfg["targets"][args.target]["paths"] = cfg["targets"][args.target].get("paths", {}) or {}
        cfg["targets"][args.target]["paths"]["output_dir"] = _absnorm(output_dir)
        cfg["targets"][args.target]["paths"]["ckpt"] = _absnorm(bm.get("best_path", os.path.join(output_dir, "best_model.pt")))

        _write_yaml(args.best_yaml, cfg)
        print(f"[run_train_from_yaml] YAML updated -> {args.best_yaml}")
        print(f"[run_train_from_yaml] best.{best_metric} = {cfg['targets'][args.target]['best']['value']}")

    print("[run_train_from_yaml] done.")


if __name__ == "__main__":
    main()
