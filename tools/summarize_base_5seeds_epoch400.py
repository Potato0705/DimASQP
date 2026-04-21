import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


PAPER_TARGETS = {
    "Laptop-ACOS": 41.37,
    "Restaurant-ACOS": 59.78,
}


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


def latest_epoch400_run_dir(runs_root: Path, dataset: str, seed: int, mask_rate: float = 0.6) -> Path | None:
    pattern = f"{dataset}_sentiment_mul_microsoft-deberta-v3-base_seed{seed}_mask{mask_rate}_*"
    dirs = sorted(
        [p for p in runs_root.glob(pattern) if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in dirs:
        args_path = run_dir / "args.json"
        if not args_path.exists():
            continue
        with args_path.open("r", encoding="utf-8-sig") as f:
            args = json.load(f)
        if args.get("epoch") == 400 and args.get("early_stop") == 5 and float(args.get("mask_rate", -1)) == mask_rate:
            return run_dir
    return None


def read_old_summary(summary_dir: Path, dataset: str) -> dict:
    if dataset == "Laptop-ACOS":
        file_name = "laptop_base_5seeds_summary.json"
    else:
        file_name = "restaurant_base_5seeds_summary.json"
    path = summary_dir / file_name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def summarize_dataset(runs_root: Path, summary_dir: Path, dataset: str, seeds: list[int]) -> dict:
    per_seed = []
    missing = []
    failed = []
    for seed in seeds:
        run_dir = latest_epoch400_run_dir(runs_root, dataset, seed, mask_rate=0.6)
        if run_dir is None:
            missing.append(seed)
            continue

        final_path = run_dir / "final_metrics.json"
        best_path = run_dir / "best_score.json"
        if not final_path.exists():
            missing.append(seed)
            continue
        with final_path.open("r", encoding="utf-8-sig") as f:
            final = json.load(f)
        if not best_path.exists():
            failed.append({"seed": seed, "run_dir": str(run_dir), "reason": "best_score.json missing"})
            continue
        with best_path.open("r", encoding="utf-8-sig") as f:
            best = json.load(f)
        dev = final["dev_quad_report"]
        test = final["test_quad_report"]
        per_seed.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "best_epoch": best.get("best_epoch", final.get("best_epoch")),
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

    old = read_old_summary(summary_dir, dataset)
    old_dev = old.get("dev_summary", {})
    old_test = old.get("test_summary", {})
    old_test_f1_mean = (((old_test.get("f1") or {}).get("mean")))
    old_dev_f1_mean = (((old_dev.get("f1") or {}).get("mean")))

    delta = {
        "dev_precision_mean_delta_vs_epoch30": None,
        "dev_recall_mean_delta_vs_epoch30": None,
        "dev_f1_mean_delta_vs_epoch30": None,
        "test_precision_mean_delta_vs_epoch30": None,
        "test_recall_mean_delta_vs_epoch30": None,
        "test_f1_mean_delta_vs_epoch30": None,
    }

    if old_dev:
        old_dev_p = ((old_dev.get("precision") or {}).get("mean"))
        old_dev_r = ((old_dev.get("recall") or {}).get("mean"))
        old_dev_f = ((old_dev.get("f1") or {}).get("mean"))
        if old_dev_p is not None and dev_summary["precision"]["mean"] is not None:
            delta["dev_precision_mean_delta_vs_epoch30"] = dev_summary["precision"]["mean"] - old_dev_p
        if old_dev_r is not None and dev_summary["recall"]["mean"] is not None:
            delta["dev_recall_mean_delta_vs_epoch30"] = dev_summary["recall"]["mean"] - old_dev_r
        if old_dev_f is not None and dev_summary["f1"]["mean"] is not None:
            delta["dev_f1_mean_delta_vs_epoch30"] = dev_summary["f1"]["mean"] - old_dev_f

    if old_test:
        old_test_p = ((old_test.get("precision") or {}).get("mean"))
        old_test_r = ((old_test.get("recall") or {}).get("mean"))
        old_test_f = ((old_test.get("f1") or {}).get("mean"))
        if old_test_p is not None and test_summary["precision"]["mean"] is not None:
            delta["test_precision_mean_delta_vs_epoch30"] = test_summary["precision"]["mean"] - old_test_p
        if old_test_r is not None and test_summary["recall"]["mean"] is not None:
            delta["test_recall_mean_delta_vs_epoch30"] = test_summary["recall"]["mean"] - old_test_r
        if old_test_f is not None and test_summary["f1"]["mean"] is not None:
            delta["test_f1_mean_delta_vs_epoch30"] = test_summary["f1"]["mean"] - old_test_f

    return {
        "dataset": dataset,
        "model_size": "base",
        "epoch_setting": 400,
        "early_stop": 5,
        "mask_rate": 0.6,
        "seeds": seeds,
        "completed_seeds": [x["seed"] for x in per_seed],
        "missing_seeds": missing,
        "failed_seeds": failed,
        "per_seed": per_seed,
        "dev_summary": dev_summary,
        "test_summary": test_summary,
        "paper_target_test_f1": target,
        "test_f1_gap_vs_paper": gap,
        "epoch30_reference_test_f1_mean": old_test_f1_mean,
        "epoch30_reference_dev_f1_mean": old_dev_f1_mean,
        "delta_vs_epoch30": delta,
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

    laptop = summarize_dataset(args.runs_root, args.summary_dir, "Laptop-ACOS", args.seeds)
    restaurant = summarize_dataset(args.runs_root, args.summary_dir, "Restaurant-ACOS", args.seeds)

    laptop_path = args.summary_dir / "laptop_base_5seeds_epoch400_summary.json"
    restaurant_path = args.summary_dir / "restaurant_base_5seeds_epoch400_summary.json"
    combined_path = args.summary_dir / "english_base_5seeds_epoch400_summary.json"

    with laptop_path.open("w", encoding="utf-8") as f:
        json.dump(laptop, f, indent=2, ensure_ascii=False)
    with restaurant_path.open("w", encoding="utf-8") as f:
        json.dump(restaurant, f, indent=2, ensure_ascii=False)

    combined = {
        "model_size": "base",
        "epoch_setting": 400,
        "early_stop": 5,
        "seeds": args.seeds,
        "datasets": {
            "Laptop-ACOS": {
                "test_f1_mean": laptop["test_summary"]["f1"]["mean"],
                "test_f1_std": laptop["test_summary"]["f1"]["std"],
                "paper_target": PAPER_TARGETS["Laptop-ACOS"],
                "gap": laptop["test_f1_gap_vs_paper"],
                "delta_vs_epoch30_test_f1_mean": laptop["delta_vs_epoch30"]["test_f1_mean_delta_vs_epoch30"],
            },
            "Restaurant-ACOS": {
                "test_f1_mean": restaurant["test_summary"]["f1"]["mean"],
                "test_f1_std": restaurant["test_summary"]["f1"]["std"],
                "paper_target": PAPER_TARGETS["Restaurant-ACOS"],
                "gap": restaurant["test_f1_gap_vs_paper"],
                "delta_vs_epoch30_test_f1_mean": restaurant["delta_vs_epoch30"]["test_f1_mean_delta_vs_epoch30"],
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
