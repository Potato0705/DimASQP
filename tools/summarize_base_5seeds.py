import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


PAPER_TARGETS = {
    "Laptop-ACOS": 41.37,
    "Restaurant-ACOS": 59.78,
}


def latest_run_dir(runs_root: Path, dataset: str, seed: int) -> Path | None:
    pattern = f"{dataset}_sentiment_mul_microsoft-deberta-v3-base_seed{seed}_mask0.6_*"
    dirs = sorted(
        [p for p in runs_root.glob(pattern) if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return dirs[0] if dirs else None


def _agg(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": mean(values), "std": pstdev(values)}


def _pack_split(records: list[dict], split: str) -> dict:
    p_vals = [r[split]["precision"] for r in records]
    r_vals = [r[split]["recall"] for r in records]
    f_vals = [r[split]["f1"] for r in records]
    return {
        "precision": _agg(p_vals),
        "recall": _agg(r_vals),
        "f1": _agg(f_vals),
    }


def summarize_dataset(runs_root: Path, dataset: str, seeds: list[int]) -> dict:
    per_seed = []
    missing = []
    for seed in seeds:
        run_dir = latest_run_dir(runs_root, dataset, seed)
        if run_dir is None:
            missing.append(seed)
            continue
        final_path = run_dir / "final_metrics.json"
        if not final_path.exists():
            missing.append(seed)
            continue
        with final_path.open("r", encoding="utf-8-sig") as f:
            final = json.load(f)
        dev = final["dev_quad_report"]
        test = final["test_quad_report"]
        per_seed.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "best_epoch": final.get("best_epoch"),
                "dev": {
                    "precision": dev["precision"],
                    "recall": dev["recall"],
                    "f1": dev["f1"],
                },
                "test": {
                    "precision": test["precision"],
                    "recall": test["recall"],
                    "f1": test["f1"],
                },
            }
        )

    per_seed = sorted(per_seed, key=lambda x: x["seed"])
    dev_summary = _pack_split(per_seed, "dev")
    test_summary = _pack_split(per_seed, "test")

    target = PAPER_TARGETS.get(dataset)
    gap = None
    if target is not None and test_summary["f1"]["mean"] is not None:
        gap = test_summary["f1"]["mean"] * 100 - target

    return {
        "dataset": dataset,
        "model_size": "base",
        "seeds": seeds,
        "completed_seeds": [x["seed"] for x in per_seed],
        "missing_seeds": missing,
        "per_seed": per_seed,
        "dev_summary": dev_summary,
        "test_summary": test_summary,
        "paper_target_test_f1": target,
        "test_f1_gap_vs_paper": gap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs_root",
        type=Path,
        default=Path("D:/Python_main/One_ASQP/One_ASQP/outputs/source_aligned_runs"),
    )
    parser.add_argument(
        "--summary_dir",
        type=Path,
        default=Path("D:/Python_main/One_ASQP/One_ASQP/outputs/source_aligned_runs/summaries"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    args.summary_dir.mkdir(parents=True, exist_ok=True)

    laptop = summarize_dataset(args.runs_root, "Laptop-ACOS", args.seeds)
    restaurant = summarize_dataset(args.runs_root, "Restaurant-ACOS", args.seeds)

    laptop_path = args.summary_dir / "laptop_base_5seeds_summary.json"
    restaurant_path = args.summary_dir / "restaurant_base_5seeds_summary.json"
    combined_path = args.summary_dir / "english_base_5seeds_summary.json"

    with laptop_path.open("w", encoding="utf-8") as f:
        json.dump(laptop, f, indent=2, ensure_ascii=False)
    with restaurant_path.open("w", encoding="utf-8") as f:
        json.dump(restaurant, f, indent=2, ensure_ascii=False)

    combined = {
        "model_size": "base",
        "seeds": args.seeds,
        "datasets": {
            "Laptop-ACOS": {
                "test_f1_mean": laptop["test_summary"]["f1"]["mean"],
                "test_f1_std": laptop["test_summary"]["f1"]["std"],
                "paper_target": PAPER_TARGETS["Laptop-ACOS"],
                "gap": laptop["test_f1_gap_vs_paper"],
            },
            "Restaurant-ACOS": {
                "test_f1_mean": restaurant["test_summary"]["f1"]["mean"],
                "test_f1_std": restaurant["test_summary"]["f1"]["std"],
                "paper_target": PAPER_TARGETS["Restaurant-ACOS"],
                "gap": restaurant["test_f1_gap_vs_paper"],
            },
        },
        "files": {
            "laptop_summary": str(laptop_path),
            "restaurant_summary": str(restaurant_path),
        },
    }
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(str(laptop_path))
    print(str(restaurant_path))
    print(str(combined_path))


if __name__ == "__main__":
    main()
