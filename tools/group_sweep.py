from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path
from statistics import mean, pstdev

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_groups import (
    DEFAULT_SEEDS,
    GROUP_DESCRIPTIONS,
    load_best_score,
    normalize_group_name,
    select_latest_runs,
)
from tools import threshold_sweep as sweep_mod
from tools.evaluate_local import evaluate_predictions, read_jsonl_file
from tools.generate_submission import load_sidecar
from utils.utils import load_train_args


DEFAULT_THRESHOLDS = [-3.0, -2.0, -1.5, -1.0, -0.5, -0.3, -0.1, 0.0, 0.3, 0.5, 1.0]


def _evaluate_submissions(submissions: list[dict], gold_data: list[dict], work_dir: Path) -> dict:
    fd, temp_path = tempfile.mkstemp(prefix="sweep_", suffix=".jsonl", dir=str(work_dir))
    os.close(fd)
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            for entry in submissions:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        pred_data = read_jsonl_file(temp_path, task=3, data_type="pred")
        return evaluate_predictions(gold_data, pred_data, task=3)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_single_sweep(
    model_path: Path,
    test_data: Path,
    sidecar: Path,
    gold: Path,
    thresholds: list[float] | None = None,
    batch_size: int = 0,
    work_dir: Path | None = None,
) -> dict:
    threshold_list = list(thresholds or DEFAULT_THRESHOLDS)
    sidecar_data = load_sidecar(str(sidecar))
    gold_data = read_jsonl_file(str(gold), task=3, data_type="gold")
    train_args = load_train_args(str(model_path))
    eval_batch = batch_size or train_args.get("per_gpu_eval_batch_size", 16)
    temp_dir = work_dir or model_path

    dataset, raw_matrices, va_preds, hidden_states, training_args, model = sweep_mod.extract_raw_logits(
        str(model_path), str(test_data), batch_size=eval_batch
    )

    try:
        rows = []
        for threshold in threshold_list:
            all_preds = sweep_mod.decode_at_threshold(
                dataset,
                raw_matrices,
                va_preds,
                hidden_states,
                threshold,
                training_args,
                model,
            )
            submissions = sweep_mod.preds_to_submission(all_preds, sidecar_data)
            metrics = _evaluate_submissions(submissions, gold_data, temp_dir)
            total_preds = sum(len(item["Quadruplet"]) for item in submissions)
            va_pct = (metrics["cTP"] / metrics["TP"] * 100.0) if metrics["TP"] > 0 else 0.0
            rows.append(
                {
                    "threshold": threshold,
                    "cF1": metrics["cF1"],
                    "cPrecision": metrics["cPrecision"],
                    "cRecall": metrics["cRecall"],
                    "TP": metrics["TP"],
                    "FP": metrics["FP"],
                    "FN": metrics["FN"],
                    "cTP": metrics["cTP"],
                    "total_preds": total_preds,
                    "va_pct": va_pct,
                }
            )

        best = max(rows, key=lambda row: row["cF1"])
        return {
            "run_dir": str(model_path),
            "seed": int(training_args["seed"]),
            "va_mode": training_args.get("va_mode", "position"),
            "weight_va_prior": float(training_args.get("weight_va_prior", 0.0)),
            "use_va_prior_aux": bool(training_args.get("use_va_prior_aux", False)),
            "train_best_score": load_best_score(model_path),
            "best_threshold": best["threshold"],
            "best_metrics": best,
            "all_thresholds": rows,
        }
    finally:
        del model, dataset, raw_matrices, va_preds, hidden_states
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_group_sweep(
    runs_root: Path,
    group: str,
    task_domain: str,
    test_data: Path,
    sidecar: Path,
    gold: Path,
    seeds: list[int] | None = None,
    thresholds: list[float] | None = None,
    batch_size: int = 0,
    output_path: Path | None = None,
) -> dict:
    normalized_group = normalize_group_name(group)
    selected_runs, missing = select_latest_runs(runs_root, normalized_group, task_domain, seeds)
    per_seed = []
    for run in selected_runs:
        per_seed.append(
            run_single_sweep(
                run.run_dir,
                test_data=test_data,
                sidecar=sidecar,
                gold=gold,
                thresholds=thresholds,
                batch_size=batch_size,
                work_dir=runs_root,
            )
        )

    best_values = [row["best_metrics"]["cF1"] for row in per_seed]
    summary = {
        "runs_root": str(runs_root),
        "group": normalized_group,
        "group_description": GROUP_DESCRIPTIONS[normalized_group],
        "task_domain": task_domain,
        "completed_seeds": [row["seed"] for row in per_seed],
        "missing_seeds": missing,
        "per_seed": per_seed,
        "mean_best_cF1": mean(best_values) if best_values else None,
        "std_best_cF1": pstdev(best_values) if len(best_values) > 1 else (0.0 if best_values else None),
        "best_single": max(per_seed, key=lambda row: row["best_metrics"]["cF1"]) if per_seed else None,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch threshold sweep for one experiment group")
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--task_domain", type=str, default="eng_restaurant")
    parser.add_argument("--test_data", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = args.output
    if output_path is None:
        output_path = args.runs_root / "eval_reports" / f"{normalize_group_name(args.group)}_sweep.json"
    summary = run_group_sweep(
        runs_root=args.runs_root,
        group=args.group,
        task_domain=args.task_domain,
        test_data=args.test_data,
        sidecar=args.sidecar,
        gold=args.gold,
        seeds=args.seeds,
        thresholds=args.thresholds,
        batch_size=args.batch_size,
        output_path=output_path,
    )
    print(json.dumps(
        {
            "group": summary["group"],
            "completed_seeds": summary["completed_seeds"],
            "missing_seeds": summary["missing_seeds"],
            "mean_best_cF1": summary["mean_best_cF1"],
            "best_single_seed": summary["best_single"]["seed"] if summary["best_single"] else None,
            "best_single_cF1": summary["best_single"]["best_metrics"]["cF1"] if summary["best_single"] else None,
            "saved_to": str(output_path),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
