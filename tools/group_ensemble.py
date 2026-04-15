from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_groups import DEFAULT_SEEDS, GROUP_DESCRIPTIONS, normalize_group_name, select_latest_runs
from tools import ensemble_eval as ens_mod
from tools.evaluate_local import evaluate_predictions, read_jsonl_file
from tools.generate_submission import load_sidecar
from group_sweep import DEFAULT_THRESHOLDS, run_group_sweep
from utils.utils import load_train_args


def _evaluate_submissions(submissions: list[dict], gold_data: list[dict], work_dir: Path) -> dict:
    fd, temp_path = tempfile.mkstemp(prefix="ensemble_", suffix=".jsonl", dir=str(work_dir))
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


def run_group_ensemble(
    runs_root: Path,
    group: str,
    task_domain: str,
    test_data: Path,
    sidecar: Path,
    gold: Path,
    seeds: list[int] | None = None,
    thresholds: list[float] | None = None,
    batch_size: int = 0,
    sweep_summary: dict | None = None,
    output_path: Path | None = None,
) -> dict:
    normalized_group = normalize_group_name(group)
    selected_runs, missing = select_latest_runs(runs_root, normalized_group, task_domain, seeds)
    if not selected_runs:
        raise RuntimeError(f"No runs found for group={normalized_group}")

    if sweep_summary is None:
        sweep_summary = run_group_sweep(
            runs_root=runs_root,
            group=normalized_group,
            task_domain=task_domain,
            test_data=test_data,
            sidecar=sidecar,
            gold=gold,
            seeds=seeds,
            thresholds=thresholds,
            batch_size=batch_size,
            output_path=None,
        )

    best_single = sweep_summary.get("best_single")
    va_model_path = Path(best_single["run_dir"]) if best_single else selected_runs[0].run_dir
    threshold_list = list(thresholds or DEFAULT_THRESHOLDS)
    sidecar_data = load_sidecar(str(sidecar))
    gold_data = read_jsonl_file(str(gold), task=3, data_type="gold")
    model_paths = [str(run.run_dir) for run in selected_runs]
    eval_batch_size = batch_size or load_train_args(str(selected_runs[0].run_dir)).get("per_gpu_eval_batch_size", 16)
    temp_dir = output_path.parent if output_path is not None else runs_root
    temp_dir.mkdir(parents=True, exist_ok=True)

    va_preds_list = []
    hidden_states_list = []
    va_head = None
    dataset = None
    training_args = None
    va_mode = None
    avg_matrices = None
    avg_mmap_path = None

    dataset, avg_mmap_path, avg_mmap_shape, va_preds_list, hidden_states_list, va_head, training_args, va_mode = (
        ens_mod.stream_average_matrices(
            model_paths=model_paths,
            test_data_path=str(test_data),
            batch_size=eval_batch_size,
            va_model_path=str(va_model_path),
            keep_hidden=(selected_runs[0].va_mode != "position"),
            temp_dir=str(temp_dir),
        )
    )

    try:
        avg_matrices = np.memmap(avg_mmap_path, mode="r", dtype=np.float32, shape=avg_mmap_shape)
        rows = []
        for threshold in threshold_list:
            if va_mode == "position":
                all_preds = ens_mod.decode_at_threshold_with_position_va(
                    dataset, avg_matrices, va_preds_list, threshold, training_args
                )
            else:
                all_preds = ens_mod.decode_at_threshold_with_span_va(
                    dataset, avg_matrices, hidden_states_list, va_head, threshold, training_args
                )

            submissions = ens_mod.preds_to_submission(all_preds, sidecar_data)
            metrics = _evaluate_submissions(submissions, gold_data, runs_root)
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
        summary = {
            "runs_root": str(runs_root),
            "group": normalized_group,
            "group_description": GROUP_DESCRIPTIONS[normalized_group],
            "task_domain": task_domain,
            "completed_seeds": [run.seed for run in selected_runs],
            "missing_seeds": missing,
            "model_paths": model_paths,
            "va_model_path": str(va_model_path),
            "best_single_seed": best_single["seed"] if best_single else None,
            "best_threshold": best["threshold"],
            "best_metrics": best,
            "all_thresholds": rows,
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary
    finally:
        if avg_matrices is not None:
            del avg_matrices
        if avg_mmap_path is not None and os.path.exists(avg_mmap_path):
            os.remove(avg_mmap_path)
        del va_preds_list, hidden_states_list, va_head, dataset, training_args
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble evaluation for one experiment group")
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--task_domain", type=str, default="eng_restaurant")
    parser.add_argument("--test_data", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sweep_json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sweep_summary = None
    if args.sweep_json is not None and args.sweep_json.exists():
        with args.sweep_json.open("r", encoding="utf-8") as f:
            sweep_summary = json.load(f)
    output_path = args.output
    if output_path is None:
        output_path = args.runs_root / "eval_reports" / f"{normalize_group_name(args.group)}_ensemble.json"
    summary = run_group_ensemble(
        runs_root=args.runs_root,
        group=args.group,
        task_domain=args.task_domain,
        test_data=args.test_data,
        sidecar=args.sidecar,
        gold=args.gold,
        seeds=args.seeds,
        thresholds=args.thresholds,
        batch_size=args.batch_size,
        sweep_summary=sweep_summary,
        output_path=output_path,
    )
    print(json.dumps(
        {
            "group": summary["group"],
            "completed_seeds": summary["completed_seeds"],
            "best_threshold": summary["best_threshold"],
            "ensemble_cF1": summary["best_metrics"]["cF1"],
            "saved_to": str(output_path),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
