# -*- coding: utf-8 -*-
import os
import json
import argparse
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from loguru import logger

from models.model import QuadrupleModel
from utils.utils import set_seeds


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_ws(s: str) -> str:
    return " ".join(str(s).strip().split()) if s is not None else ""


def parse_va(va_str: str):
    try:
        a, b = str(va_str).split("#")
        return float(a), float(b)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_stats", required=True)     # train_gold_task3.jsonl
    ap.add_argument("--ckpt", required=True)            # best_model.pt
    ap.add_argument("--model_name", required=True)      # deberta-v3-base
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])
    args = ap.parse_args()

    set_seeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Device] {device}")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    rows = read_jsonl(args.train_stats)

    queries = []
    meta = []
    for r in rows:
        text = clean_ws(r.get("Text") or r.get("text") or "")
        qs = r.get("Quadruplet") or r.get("Quadruplets") or r.get("quadruplets") or []
        for q in qs:
            a = clean_ws(q.get("Aspect", "null"))
            o = clean_ws(q.get("Opinion", "null"))
            cat = clean_ws(q.get("Category", "LAPTOP#GENERAL")).upper()
            va = q.get("VA", "5.00#5.00")
            xy = parse_va(va)
            if xy is None:
                v, ar = 5.0, 5.0
            else:
                v, ar = xy
            queries.append(f"{text} [SEP] {a} [SEP] {o}")
            meta.append({"Category": cat, "V": float(v), "A": float(ar)})

    logger.info(f"[Index] entries={len(queries)}")

    # model
    # num_label_types/num_dims 在建库不重要，这里随便给合理值即可
    model = QuadrupleModel(num_label_types=3, num_dimension_types=3, max_seq_len=args.max_len, pretrain_model_path=args.model_name).to(device)
    state = torch.load(args.ckpt, map_location=device)
    # 只加载 encoder 权重 + 对应模块权重；严格加载会因为 head 数不同报错，所以 strict=False
    model.load_state_dict(state, strict=False)
    model.eval()

    all_emb = []
    for i in tqdm(range(0, len(queries), args.batch), desc="BuildIndex", ncols=90):
        chunk = queries[i:i+args.batch]
        enc = tok(chunk, truncation=True, max_length=args.max_len, padding="max_length", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            emb = model.encode_embeddings(enc["input_ids"], enc.get("token_type_ids", None), enc["attention_mask"], pooling=args.pooling, normalize=True)
        all_emb.append(emb.detach().cpu().numpy().astype(np.float32))

    emb = np.concatenate(all_emb, axis=0)
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(args.out_npz, emb=emb.astype(np.float16), meta=np.array(meta, dtype=object))
    logger.success(f"[Saved] {args.out_npz}  emb={emb.shape}")


if __name__ == "__main__":
    main()
