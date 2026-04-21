"""
End-to-end evaluation pipeline: predict -> generate submission -> evaluate cF1.

Usage:
    python tools/eval_pipeline.py \
        --model_path output/eng_restaurant_category_... \
        --test_data data/eng/eng_restaurant_dev.txt \
        --sidecar data/eng/eng_restaurant_dev_sidecar.json \
        --gold data/eng/eng_restaurant_dev.jsonl \
        --threshold -0.3
"""
import json
import os
import sys
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="End-to-end evaluation pipeline")
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--test_data", required=True, type=str)
    parser.add_argument("--sidecar", required=True, type=str)
    parser.add_argument("--gold", required=True, type=str)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tag", type=str, default="", help="Tag for output files")
    args = parser.parse_args()

    tag = args.tag or os.path.basename(args.model_path)[:40]

    print(f"\n{'='*60}")
    print(f"Evaluation Pipeline: {tag}")
    print(f"  Model: {args.model_path}")
    print(f"  Test:  {args.test_data}")
    print(f"  Threshold: {args.threshold}")
    print(f"{'='*60}\n")

    # Step 1: Predict
    print("[Step 1/3] Running prediction...")
    predict_cmd = [
        sys.executable, "predict.py",
        "--model_path", args.model_path,
        "--test_data", args.test_data,
        "--per_gpu_test_batch_size", str(args.batch_size),
        "--threshold", str(args.threshold),
    ]
    result = subprocess.run(predict_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Prediction failed:\n{result.stderr[-500:]}")
        return
    print("  Prediction done.")

    # Find predictions JSON
    test_name = os.path.splitext(os.path.basename(args.test_data))[0]
    pred_json = os.path.join(args.model_path, f"{test_name}_predictions.json")
    if not os.path.exists(pred_json):
        print(f"  Prediction file not found: {pred_json}")
        return

    # Step 2: Generate submission
    print("[Step 2/3] Generating submission JSONL...")
    submission_path = f"submission/{tag}_thresh{args.threshold:.1f}.jsonl"
    os.makedirs("submission", exist_ok=True)

    gen_cmd = [
        sys.executable, "tools/generate_submission.py",
        "--pred_json", pred_json,
        "--sidecar", args.sidecar,
        "--output", submission_path,
        "--use_va_pred",
    ]
    result = subprocess.run(gen_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Submission generation failed:\n{result.stderr[-500:]}")
        return
    print(f"  {result.stdout.strip()}")

    # Step 3: Evaluate
    print("[Step 3/3] Running cF1 evaluation...")
    eval_cmd = [
        sys.executable, "tools/evaluate_local.py",
        "--pred", submission_path,
        "--gold", args.gold,
        "--task", "3",
    ]
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    print(result.stdout)

    # Save summary
    summary_path = f"submission/{tag}_thresh{args.threshold:.1f}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"Test data: {args.test_data}\n")
        f.write(result.stdout)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
