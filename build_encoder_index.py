# -*- coding: utf-8 -*-
"""
Build Encoder-Same Embedding Index (Quadruplet-level)

Outputs:
  - index.npz: text_emb [N,D], asp_emb [N,D], ids [N]
  - index_meta.jsonl: per-item metadata (Aspect/Opinion/Category/VA/Text...)

Usage (one-liner):
python build_encoder_index.py --train_stats ./output/train_gold_task3.jsonl --ckpt ./output/best_model.pt --model_name ./deberta-v3-base --max_len 256 --batch 64 --out_dir ./output/index
"""

import os, json, argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from loguru import logger

from models.model import QuadrupleModel  # 你的工程里已有
from utils.utils import set_seeds

set_seeds(42)

NULL_STR = "null"

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

def get_text_field(row: Dict[str, Any]) -> str:
    for k in ["Text", "text", "sentence", "raw_text"]:
        if k in row and row[k]:
            return clean_ws(row[k])
    return ""

def get_quad_list(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return row.get("Quadruplet") or row.get("Quadruplets") or row.get("quadruplets") or []

def get_backbone(model: torch.nn.Module):
    """
    尝试从 QuadrupleModel 中拿到 HuggingFace backbone.
    你无需改 model.py；这里做鲁棒探测。
    """
    for name in ["encoder", "backbone", "deberta", "bert", "roberta", "plm", "pretrained_model", "transformer"]:
        if hasattr(model, name):
            return getattr(model, name)
    # 最后兜底：扫描子模块里是否有 transformers 模型
    for _, m in model.named_modules():
        if m.__class__.__name__.endswith("Model") and hasattr(m, "forward"):
            # 可能会误选；但一般不会走到这里
            return m
    raise RuntimeError("Cannot locate backbone encoder in QuadrupleModel. Please expose it as self.encoder/self.backbone.")

@torch.no_grad()
def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B,L,1]
    x = last_hidden * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return x.sum(dim=1) / denom

@torch.no_grad()
def encode_texts(backbone, tokenizer, texts: List[str], max_len: int, device: str, batch_size: int = 64) -> np.ndarray:
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encode", ncols=90):
        batch = texts[i:i+batch_size]
        tok = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        out = backbone(**tok, return_dict=True)
        last = out.last_hidden_state
        pooled = mean_pool(last, tok["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        embs.append(pooled.detach().cpu().numpy())
    return np.concatenate(embs, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_stats", required=True, help="jsonl with Text + Quadruplet list")
    ap.add_argument("--ckpt", required=True, help="best_model.pt")
    ap.add_argument("--model_name", required=True, help="hf model path/dir")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out_dir", type=str, default="./output/index")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Device] {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # 读取并展开为 quadruplet-level items
    rows = read_jsonl(args.train_stats)
    items = []
    for r in rows:
        sid = r.get("ID") or r.get("id") or ""
        text = get_text_field(r)
        qs = get_quad_list(r)
        for qi, q in enumerate(qs):
            a = clean_ws(q.get("Aspect", NULL_STR)) or NULL_STR
            o = clean_ws(q.get("Opinion", NULL_STR)) or NULL_STR
            c = clean_ws(q.get("Category", "LAPTOP#GENERAL")) or "LAPTOP#GENERAL"
            va = clean_ws(q.get("VA", "5.00#5.00")) or "5.00#5.00"
            items.append({
                "qid": f"{sid}#{qi}",
                "sid": sid,
                "text": text,
                "Aspect": a,
                "Opinion": o,
                "Category": c,
                "VA": va,
            })

    logger.info(f"[Items] quadruplet-level items={len(items)} from rows={len(rows)}")

    # Load fine-tuned model and backbone
    # 注意：这里 num_label_types/num_dims/max_seq_len 需要与你训练时一致
    # 你若不确定，建议从 dataset 里读取（后续我们也可以改成自动推断）
    # 这里先用“保守写法”：从 ckpt 加载不依赖头shape的做法通常不行，所以保持一致是必须的。
    # 若你当前 QuadrupleModel init 需要这些参数，请按你训练时填。
    # 下面这行如果报错，我们再改成从 AcqpDataset 读取配置。
    model = QuadrupleModel(
        num_label_types=11,          # 你日志里 num_label_types=11（sentiment_dim）
        num_dimension_types=3,       # usability/quality/performance
        max_seq_len=args.max_len,
        pretrain_model_path=args.model_name,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    backbone = get_backbone(model)
    logger.info(f"[Backbone] {backbone.__class__.__name__}")

    texts = [it["text"] for it in items]
    aspects = [it["Aspect"] if it["Aspect"] != NULL_STR else "" for it in items]

    text_emb = encode_texts(backbone, tokenizer, texts, args.max_len, device, args.batch)

    # Aspect 为 null 的用 0 向量（避免检索时“乱相似”）
    asp_emb = np.zeros_like(text_emb)
    non_null_idx = [i for i, a in enumerate(aspects) if a.strip()]
    if non_null_idx:
        asp_texts = [aspects[i] for i in non_null_idx]
        asp_vecs = encode_texts(backbone, tokenizer, asp_texts, args.max_len, device, args.batch)
        asp_emb[non_null_idx] = asp_vecs

    ids = np.array([it["qid"] for it in items], dtype=object)

    npz_path = os.path.join(args.out_dir, "index.npz")
    meta_path = os.path.join(args.out_dir, "index_meta.jsonl")

    np.savez_compressed(npz_path, ids=ids, text_emb=text_emb.astype(np.float16), asp_emb=asp_emb.astype(np.float16))
    with open(meta_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    logger.success(f"Saved index -> {npz_path}")
    logger.success(f"Saved meta  -> {meta_path}")
    logger.info("Reminder: after you retrain the model, rebuild this index (because encoder weights change).")

if __name__ == "__main__":
    main()
