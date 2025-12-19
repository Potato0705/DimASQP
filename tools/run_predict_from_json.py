import json
import subprocess
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/run_predict_from_json.py <config.json> [--dry-run]")
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv[2:]

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Required keys
    required = ["input", "train_stats", "ckpt", "model_name", "output"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required keys in config: {missing}")

    # Build predict command
    cmd = [
        sys.executable, "predict.py",
        "--input", cfg["input"],
        "--train_stats", cfg["train_stats"],
        "--ckpt", cfg["ckpt"],
        "--model_name", cfg["model_name"],
        "--output", cfg["output"],
    ]

    # Optional scalar args (only add if present)
    scalar_map = {
        "max_len": "--max_len",
        "batch": "--batch",
        "label_pattern": "--label_pattern",
        "thr_aux": "--thr_aux",
        "topk_aux": "--topk_aux",
        "max_span_len": "--max_span_len",
        "thr_rel": "--thr_rel",
        "topk_rel": "--topk_rel",
        "max_pair_dist": "--max_pair_dist",
        "max_quads": "--max_quads",
        "min_score": "--min_score",
        "null_thr_o": "--null_thr_o",
        "va_stat": "--va_stat",
        "cat_case": "--cat_case",
        "retrieval_index": "--retrieval_index",
        "ret_topk": "--ret_topk",
        "ret_min_sim": "--ret_min_sim",
        "ret_vote_margin": "--ret_vote_margin",
    }

    for k, flag in scalar_map.items():
        if k in cfg and cfg[k] is not None:
            cmd.extend([flag, str(cfg[k])])

    # Optional boolean flags
    if cfg.get("refine_span", False):
        cmd.append("--refine_span")
    if cfg.get("diag", False):
        cmd.append("--diag")

    print("Command:")
    print(" ".join(cmd))

    if dry_run:
        return

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
