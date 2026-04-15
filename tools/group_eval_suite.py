from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_groups import DEFAULT_SEEDS
from group_ensemble import run_group_ensemble
from group_sweep import run_group_sweep


DEFAULT_RESTAURANT_GROUPS = [
    "table1_position",
    "table1_span_pair",
    "table1_og_full",
    "ablation_plain_sp",
    "ablation_sp_prior",
    "ablation_og_no_prior",
    "ablation_og_full",
]


def run_suite(
    runs_root: Path,
    task_domain: str,
    test_data: Path,
    sidecar: Path,
    gold: Path,
    groups: list[str],
    seeds: list[int],
    report_dir: Path,
) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    detailed = {"runs_root": str(runs_root), "task_domain": task_domain, "groups": {}}

    for group in groups:
        sweep_path = report_dir / f"{group}_sweep.json"
        ensemble_path = report_dir / f"{group}_ensemble.json"
        sweep_summary = run_group_sweep(
            runs_root=runs_root,
            group=group,
            task_domain=task_domain,
            test_data=test_data,
            sidecar=sidecar,
            gold=gold,
            seeds=seeds,
            output_path=sweep_path,
        )
        ensemble_summary = run_group_ensemble(
            runs_root=runs_root,
            group=group,
            task_domain=task_domain,
            test_data=test_data,
            sidecar=sidecar,
            gold=gold,
            seeds=seeds,
            sweep_summary=sweep_summary,
            output_path=ensemble_path,
        )
        detailed["groups"][group] = {
            "sweep": sweep_summary,
            "ensemble": ensemble_summary,
        }
        rows.append(
            {
                "group": group,
                "completed_seeds": ",".join(str(seed) for seed in sweep_summary["completed_seeds"]),
                "missing_seeds": ",".join(str(seed) for seed in sweep_summary["missing_seeds"]),
                "mean_best_cF1": sweep_summary["mean_best_cF1"],
                "std_best_cF1": sweep_summary["std_best_cF1"],
                "best_single_seed": sweep_summary["best_single"]["seed"] if sweep_summary["best_single"] else "",
                "best_single_cF1": sweep_summary["best_single"]["best_metrics"]["cF1"] if sweep_summary["best_single"] else "",
                "best_single_threshold": sweep_summary["best_single"]["best_threshold"] if sweep_summary["best_single"] else "",
                "ensemble_cF1": ensemble_summary["best_metrics"]["cF1"],
                "ensemble_threshold": ensemble_summary["best_threshold"],
                "ensemble_cPrecision": ensemble_summary["best_metrics"]["cPrecision"],
                "ensemble_cRecall": ensemble_summary["best_metrics"]["cRecall"],
                "va_model_seed": ensemble_summary["best_single_seed"],
            }
        )

    json_path = report_dir / f"{task_domain}_suite_summary.json"
    csv_path = report_dir / f"{task_domain}_suite_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "rows": rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sweep + ensemble for multiple experiment groups")
    parser.add_argument("--runs_root", type=Path, default=Path("output/output_v2"))
    parser.add_argument("--task_domain", type=str, default="eng_restaurant")
    parser.add_argument("--test_data", type=Path, default=Path("data/eng/eng_restaurant_dev.txt"))
    parser.add_argument("--sidecar", type=Path, default=Path("data/eng/eng_restaurant_dev_sidecar.json"))
    parser.add_argument("--gold", type=Path, default=Path("data/eng/eng_restaurant_dev.jsonl"))
    parser.add_argument("--groups", type=str, nargs="+", default=DEFAULT_RESTAURANT_GROUPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--report_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report_dir = args.report_dir or (args.runs_root / "eval_reports")
    result = run_suite(
        runs_root=args.runs_root,
        task_domain=args.task_domain,
        test_data=args.test_data,
        sidecar=args.sidecar,
        gold=args.gold,
        groups=args.groups,
        seeds=args.seeds,
        report_dir=report_dir,
    )

    print("\nRestaurant/Group summary:")
    for row in result["rows"]:
        print(
            f"{row['group']}: "
            f"mean={row['mean_best_cF1']:.4f} | "
            f"best_single={row['best_single_cF1']:.4f} (seed {row['best_single_seed']}) | "
            f"ensemble={row['ensemble_cF1']:.4f} @ {row['ensemble_threshold']}"
        )
    print(f"\nJSON: {result['json_path']}")
    print(f"CSV:  {result['csv_path']}")


if __name__ == "__main__":
    main()
