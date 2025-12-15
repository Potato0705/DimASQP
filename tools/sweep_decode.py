# tools/sweep_decode.py
# -*- coding: utf-8 -*-
import os, re, csv, json, argparse, itertools, subprocess, ast
from datetime import datetime

RE_FINAL = re.compile(r"Final Results:\s*(\{.*\})")

def run(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    return p.returncode, out

def parse_final_metrics(text):
    m = RE_FINAL.search(text)
    if not m:
        return None
    try:
        d = ast.literal_eval(m.group(1))
        return d
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="gold jsonl for evaluation (e.g. valid_gold_task3.jsonl)")
    ap.add_argument("--input", required=True, help="predict input jsonl (same as gold for offline eval)")
    ap.add_argument("--train_stats", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", required=True)

    ap.add_argument("--out_dir", default="./output/sweep2")
    ap.add_argument("--csv_path", default="./output/sweep2_results.csv")

    # fixed predict args
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--label_pattern", default="sentiment_dim")

    ap.add_argument("--thr_aux", type=float, default=0.05)
    ap.add_argument("--topk_aux", type=int, default=80)
    ap.add_argument("--max_span_len", type=int, default=12)
    ap.add_argument("--topk_rel", type=int, default=800)
    ap.add_argument("--max_pair_dist", type=int, default=120)
    ap.add_argument("--cat_case", default="upper")
    ap.add_argument("--va_stat", default="median")

    # sweep grids
    ap.add_argument("--thr_rel_list", default="0.08,0.10,0.12,0.14,0.16,0.18,0.20")
    ap.add_argument("--min_score_list", default="0.6,0.8,1.0,1.2,1.4")
    ap.add_argument("--null_thr_o_list", default="0.10,0.12,0.14,0.16")
    ap.add_argument("--max_quads_list", default="2")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    thr_rel_list = [float(x) for x in args.thr_rel_list.split(",") if x.strip()]
    min_score_list = [float(x) for x in args.min_score_list.split(",") if x.strip()]
    null_thr_o_list = [float(x) for x in args.null_thr_o_list.split(",") if x.strip()]
    max_quads_list = [int(x) for x in args.max_quads_list.split(",") if x.strip()]

    header = ["thr_rel","min_score","null_thr_o","max_quads","TP","FP","FN","cPrecision","cRecall","cF1","pred_path"]
    if not os.path.exists(args.csv_path):
        with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    total = 0
    for thr_rel, min_score, null_thr_o, max_quads in itertools.product(
        thr_rel_list, min_score_list, null_thr_o_list, max_quads_list
    ):
        total += 1
        tag = f"tr{thr_rel}_ms{min_score}_nt{null_thr_o}_k{max_quads}"
        pred_path = os.path.join(args.out_dir, f"pred_{tag}.jsonl")

        cmd_pred = (
            f"python predict.py --input {args.input} --train_stats {args.train_stats} --ckpt {args.ckpt} "
            f"--model_name {args.model_name} --max_len {args.max_len} --batch {args.batch} --label_pattern {args.label_pattern} "
            f"--thr_aux {args.thr_aux} --topk_aux {args.topk_aux} --max_span_len {args.max_span_len} "
            f"--thr_rel {thr_rel} --topk_rel {args.topk_rel} --max_pair_dist {args.max_pair_dist} "
            f"--max_quads {max_quads} --null_thr_o {null_thr_o} --min_score {min_score} "
            f"--va_stat {args.va_stat} --cat_case {args.cat_case} --output {pred_path}"
        )

        rc1, out1 = run(cmd_pred)
        if rc1 != 0:
            print(f"[ERR] predict failed: {tag}\n{out1[:2000]}")
            continue

        cmd_eval = f"python metrics_subtask_1_2_3.py -g {args.gold} -p {pred_path} -t 3"
        rc2, out2 = run(cmd_eval)
        d = parse_final_metrics(out2)
        if (rc2 != 0) or (d is None):
            print(f"[ERR] eval failed: {tag}\n{out2[:2000]}")
            continue

        row = [
            thr_rel, min_score, null_thr_o, max_quads,
            d.get("TP"), d.get("FP"), d.get("FN"),
            d.get("cPrecision"), d.get("cRecall"), d.get("cF1"),
            pred_path
        ]
        with open(args.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

        print(f"[OK] {tag} cF1={d.get('cF1'):.6f}")

if __name__ == "__main__":
    main()
