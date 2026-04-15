"""Offline merger for gold + LLM pseudo-labeled TXT files.

Given a gold train TXT (e.g., data/v2/eng/eng_restaurant_train.txt) and one or
more pseudo-labeled TXT files produced by data/llm_pseudo_labeler.py, this
script samples pseudo rows at a chosen ratio, shuffles them with the gold rows,
and writes a single TXT that train.py can consume via its existing --train_data
flag. Matching sidecar JSON and gold-style JSONL are written in parallel so the
rest of the pipeline (threshold_sweep / ensemble_eval) keeps working.

Leak guard: refuses to merge if any --pseudo path looks like a dev/test split.

Typical usage:
    python data/merge_pseudo_with_gold.py \
        --gold data/v2/eng/eng_restaurant_train.txt \
        --pseudo data/v2/eng/eng_restaurant_train_pseudo__llama31-70b.txt \
        --ratio 1.0 --seed 42 \
        --out data/v2/eng/eng_restaurant_train__gold+pseudo_llama31-70b_r1.0.txt

Ratio semantics:
    --ratio R means: keep floor(R * len(pseudo)) pseudo rows (sampled without
    replacement) alongside all gold rows. R=1.0 means "use all pseudo", R=0.5
    "half of pseudo", R=2.0 "duplicate-sample pseudo to 2x" (with replacement).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Optional


def _looks_like_eval_split(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    return (
        "_dev" in p
        or "_test" in p
        or p.endswith("/dev.txt")
        or p.endswith("/test.txt")
        or p.endswith("/dev.jsonl")
        or p.endswith("/test.jsonl")
    )


def _read_txt(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def _read_sidecar(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_gold_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def _sister_paths(txt_path: str) -> dict:
    """Given a TXT path, guess the matching sidecar / gold-jsonl paths."""
    root, _ext = os.path.splitext(txt_path)
    return {
        "sidecar": root + "_sidecar.json",
        "jsonl": root + ".jsonl",
    }


def _sample_indices(n: int, ratio: float, seed: int) -> List[int]:
    rng = random.Random(seed)
    target = int(round(ratio * n))
    if target <= 0:
        return []
    if target <= n:
        return rng.sample(range(n), target)
    # ratio > 1: sample with replacement for the overflow.
    base = list(range(n))
    rng.shuffle(base)
    extra = [rng.randrange(n) for _ in range(target - n)]
    return base + extra


def main():
    parser = argparse.ArgumentParser(
        description="Merge gold training TXT with LLM pseudo-labeled TXT(s)."
    )
    parser.add_argument("--gold", required=True, help="Path to the gold train TXT.")
    parser.add_argument("--pseudo", nargs="+", required=True,
                        help="One or more pseudo TXT files (same ####-format).")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Fraction of pseudo rows to keep (per --pseudo file).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True,
                        help="Output TXT path. Parallel .jsonl and _sidecar.json are also written.")
    parser.add_argument("--no_shuffle", action="store_true",
                        help="Keep gold rows first, pseudo rows appended (default: shuffle).")
    args = parser.parse_args()

    if _looks_like_eval_split(args.gold):
        parser.error(f"--gold {args.gold!r} looks like a dev/test split. Refusing.")
    for p in args.pseudo:
        if _looks_like_eval_split(p):
            parser.error(f"--pseudo {p!r} looks like a dev/test split. Refusing.")

    # ------------------------------------------------------------------ load
    gold_txt = _read_txt(args.gold)
    gold_sides = _sister_paths(args.gold)
    gold_sidecar = _read_sidecar(gold_sides["sidecar"])
    gold_jsonl = _read_gold_jsonl(gold_sides["jsonl"])

    if gold_sidecar and len(gold_sidecar) != len(gold_txt):
        print(f"[warn] gold sidecar len={len(gold_sidecar)} != TXT len={len(gold_txt)}; sidecar will be dropped.")
        gold_sidecar = []
    if gold_jsonl and len(gold_jsonl) != len(gold_txt):
        print(f"[warn] gold jsonl len={len(gold_jsonl)} != TXT len={len(gold_txt)}; jsonl will be dropped.")
        gold_jsonl = []

    merged_txt: List[str] = list(gold_txt)
    merged_sidecar: List[dict] = list(gold_sidecar) if gold_sidecar else []
    merged_jsonl: List[dict] = list(gold_jsonl) if gold_jsonl else []

    # -------------------------------------------------------- sample + merge
    rng_seed = args.seed
    per_file_counts = []
    for p in args.pseudo:
        p_txt = _read_txt(p)
        p_sides = _sister_paths(p)
        p_sidecar = _read_sidecar(p_sides["sidecar"])
        p_jsonl = _read_gold_jsonl(p_sides["jsonl"])
        if p_sidecar and len(p_sidecar) != len(p_txt):
            p_sidecar = []
        if p_jsonl and len(p_jsonl) != len(p_txt):
            p_jsonl = []

        idxs = _sample_indices(len(p_txt), args.ratio, rng_seed)
        rng_seed += 1  # keep per-file sampling deterministic but distinct

        taken = [p_txt[i] for i in idxs]
        merged_txt.extend(taken)
        if merged_sidecar is not None and p_sidecar:
            merged_sidecar.extend([p_sidecar[i] for i in idxs])
        elif merged_sidecar:
            # one file has sidecar, another doesn't → drop sidecar entirely
            merged_sidecar = []
        if merged_jsonl is not None and p_jsonl:
            merged_jsonl.extend([p_jsonl[i] for i in idxs])
        elif merged_jsonl:
            merged_jsonl = []
        per_file_counts.append((p, len(p_txt), len(idxs)))

    # ------------------------------------------------------------ shuffle
    if not args.no_shuffle:
        order = list(range(len(merged_txt)))
        random.Random(args.seed).shuffle(order)
        merged_txt = [merged_txt[i] for i in order]
        if merged_sidecar and len(merged_sidecar) == len(order):
            merged_sidecar = [merged_sidecar[i] for i in order]
            # Sidecar line_index must be rewritten to match the new order.
            for new_idx, rec in enumerate(merged_sidecar):
                rec["line_index"] = new_idx
        if merged_jsonl and len(merged_jsonl) == len(order):
            merged_jsonl = [merged_jsonl[i] for i in order]

    # -------------------------------------------------------------- write
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(merged_txt) + ("\n" if merged_txt else ""))

    root, _ext = os.path.splitext(args.out)
    if merged_sidecar:
        with open(root + "_sidecar.json", "w", encoding="utf-8") as f:
            json.dump(merged_sidecar, f, ensure_ascii=False, indent=2)
    if merged_jsonl:
        with open(root + ".jsonl", "w", encoding="utf-8") as f:
            for g in merged_jsonl:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print(
        f"Merged: gold_rows={len(gold_txt)}  "
        f"+ pseudo_rows={sum(c[2] for c in per_file_counts)}  "
        f"=> total={len(merged_txt)}"
    )
    for p, total, kept in per_file_counts:
        print(f"  - {p}: kept {kept}/{total} (ratio={args.ratio})")
    print(f"  out TXT     : {args.out}")
    if merged_sidecar:
        print(f"  out sidecar : {root}_sidecar.json")
    if merged_jsonl:
        print(f"  out JSONL   : {root}.jsonl")


if __name__ == "__main__":
    main()
