# -*- coding: utf-8 -*-
import os, re, csv, itertools, subprocess, sys

# === 你只需要改这里：基础路径/命令参数 ===
PRED_PY = "predict.py"
METRIC_PY = "metrics_subtask_1_2_3.py"

INPUT_GOLD = "./output/valid_gold_task3.jsonl"
TRAIN_STATS = "./output/train_gold_task3.jsonl"
CKPT = "./output/best_model.pt"
MODEL_NAME = "./deberta-v3-base"

OUT_DIR = "./output/sweep"
os.makedirs(OUT_DIR, exist_ok=True)

# 你当前固定参数（建议 sweep 时先固定住，保证可比）
FIXED = {
    "max_len": 256,
    "batch": 8,
    "label_pattern": "sentiment_dim",
    "topk_aux": 80,
    "max_span_len": 12,
    "topk_rel": 800,
    "max_pair_dist": 120,
    "va_stat": "median",
    "cat_case": "upper",
    # 常用开关：你要不要扫它们可以后面再加维度
    "apostrophe_norm": True,
    "refine_span": False,
    "dedup_by_aspect": False,
    "no_pair2cat_when_op_null": False,
}

# === 第一阶段：只扫最关键的 3 个参数，组合数小，最快定位方向 ===
GRID = {
    "thr_rel":    [0.12, 0.15, 0.18, 0.20],
    "max_quads":  [1, 2],
    "null_thr_o": [0.10, 0.12, 0.14, 0.16],
    # 可选：如果你想把 aux 也一起扫，打开下面一行（组合数会变大）
    # "thr_aux":    [0.03, 0.05, 0.07],
}

# metrics 脚本最后会输出 "Final Results: {... 'cF1': 0.xxx}"
RE_C = re.compile(r"'TP':\s*([0-9.]+).*?'FP':\s*([0-9.]+).*?'FN':\s*([0-9.]+).*?'cPrecision':\s*([0-9.]+).*?'cRecall':\s*([0-9.]+).*?'cF1':\s*([0-9.]+)", re.S)

def bool_flag(name, v):
    return f" --{name}" if v else ""

def build_cmd(params):
    # 输出文件名（保证每组不同）
    tag = "_".join([f"{k}{params[k]}" for k in sorted(params.keys())])
    out_path = os.path.join(OUT_DIR, f"pred_{tag}.jsonl").replace("\\", "/")

    # predict 命令
    cmd = (
        f"python {PRED_PY}"
        f" --input {INPUT_GOLD}"
        f" --train_stats {TRAIN_STATS}"
        f" --ckpt {CKPT}"
        f" --model_name {MODEL_NAME}"
        f" --max_len {FIXED['max_len']}"
        f" --batch {FIXED['batch']}"
        f" --label_pattern {FIXED['label_pattern']}"
        f" --thr_aux {params.get('thr_aux', FIXED.get('thr_aux', 0.05))}"
        f" --topk_aux {FIXED['topk_aux']}"
        f" --max_span_len {FIXED['max_span_len']}"
        f" --thr_rel {params['thr_rel']}"
        f" --topk_rel {FIXED['topk_rel']}"
        f" --max_pair_dist {FIXED['max_pair_dist']}"
        f" --max_quads {params['max_quads']}"
        f" --null_thr_o {params['null_thr_o']}"
        f" --va_stat {FIXED['va_stat']}"
        f" --cat_case {FIXED['cat_case']}"
        f"{bool_flag('apostrophe_norm', FIXED['apostrophe_norm'])}"
        f"{bool_flag('refine_span', FIXED['refine_span'])}"
        f"{bool_flag('dedup_by_aspect', FIXED['dedup_by_aspect'])}"
        f"{bool_flag('no_pair2cat_when_op_null', FIXED['no_pair2cat_when_op_null'])}"
        f" --output {out_path}"
    )

    # metrics 命令
    cmd2 = f"python {METRIC_PY} -g {INPUT_GOLD} -p {out_path} -t 3"
    return cmd, cmd2, out_path

def run_one(params):
    cmd, cmd2, out_path = build_cmd(params)
    # 运行 predict
    r1 = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r1.returncode != 0:
        return None, out_path, cmd, r1.stdout + "\n" + r1.stderr

    # 运行 metrics
    r2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
    txt = r2.stdout + "\n" + r2.stderr
    if r2.returncode != 0:
        return None, out_path, cmd2, txt

    m = RE_C.search(txt)
    if not m:
        return None, out_path, cmd2, txt

    TP, FP, FN, cP, cR, cF1 = map(float, m.groups())
    return (TP, FP, FN, cP, cR, cF1), out_path, cmd + " ; " + cmd2, txt

def main():
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))

    rows = []
    for vals in combos:
        params = dict(zip(keys, vals))
        metrics, out_path, fullcmd, logtxt = run_one(params)
        if metrics is None:
            print(f"[FAIL] {params}\n{fullcmd}\n---\n{logtxt}\n")
            continue
        TP, FP, FN, cP, cR, cF1 = metrics
        row = {**params, "TP": TP, "FP": FP, "FN": FN, "cPrecision": cP, "cRecall": cR, "cF1": cF1, "pred_path": out_path}
        rows.append(row)
        print(f"[OK] {params} -> cF1={cF1:.6f} (FP={int(FP)} FN={int(FN)})")

    rows.sort(key=lambda x: x["cF1"], reverse=True)
    out_csv = os.path.join(OUT_DIR, "sweep_results.csv").replace("\\", "/")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    print("\n=== TOP 10 ===")
    for r in rows[:10]:
        print({k: r[k] for k in keys + ["cF1", "FP", "FN"]})

    print(f"\nSaved: {out_csv}")

if __name__ == "__main__":
    main()
