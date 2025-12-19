# -*- coding: utf-8 -*-
"""
SemEval-2026 Task 3 (DimABSA) Train

"""
import os
import json
import argparse
import re
import subprocess
import sys
from pathlib import Path

from collections import defaultdict

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from loguru import logger
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import torch.nn.functional as F

from dataset.dataset import AcqpDataset, collate_fn
from losses.losses import global_pointer_crossentropy, masked_bce_with_logits
from models.model import QuadrupleModel
from utils.utils import set_seeds
from utils.config_loader import load_config, apply_cfg_defaults


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _build_token_mask(input_ids, attention_mask, tok):
    m = attention_mask.bool().clone()
    if tok.pad_token_id is not None:
        m[input_ids == tok.pad_token_id] = False
    if tok.sep_token_id is not None:
        m[input_ids == tok.sep_token_id] = False
    return m


def build_pair_batch(cat_pair_indices: torch.Tensor, cat_pair_labels: torch.Tensor, cat_pair_mask: torch.Tensor):
    """
    cat_pair_indices: [B,K,2]
    cat_pair_labels:  [B,K]
    cat_pair_mask:    [B,K] bool
    Return:
      pair_idx: [N,3] (b, a_start, o_end)
      labels:  [N]
      m:       [B,K] (original mask)
    """
    B, K, _ = cat_pair_indices.shape
    device = cat_pair_indices.device
    m = cat_pair_mask.bool()
    if m.sum().item() == 0:
        return None, None, m

    b_ids = torch.arange(B, device=device)[:, None].expand(B, K)[m]
    pairs = cat_pair_indices[m]
    labels = cat_pair_labels[m].long()
    pair_idx = torch.stack([b_ids.long(), pairs[:, 0].long(), pairs[:, 1].long()], dim=-1)
    return pair_idx, labels, m


def compute_category_loss(
    model,
    seq_out,
    batch,
    device,
    hard_neg_ratio=0.3,
    log_every=0,
    step_i=0,
):
    """
    Category head loss with candidate masking:
    - Ensure INVALID always candidate
    - Ensure gold positive label always candidate
    - Use safe masking value (-1e4) for fp16 stability
    - Optionally hard-negative mining on INVALID pairs
    """
    pair_idx, labels, m = build_pair_batch(
        batch["cat_pair_indices"],
        batch["cat_pair_labels"],
        batch["cat_pair_mask"]
    )
    if pair_idx is None:
        return torch.tensor(0.0, device=device)

    labels = labels.to(device)

    cand_mask = batch["cat_pair_cand_mask"][m].to(device)  # [N, C]
    logits = model.classify_pairs(seq_out, pair_idx)        # [N, C]
    if logits.size(-1) != cand_mask.size(-1):
        raise RuntimeError(f"Category dim mismatch: logits={logits.shape}, cand_mask={cand_mask.shape}")

    INVALID_ID = model.num_categories - 1

    # Make sure INVALID always candidate
    cand_mask[:, INVALID_ID] = True

    # Make sure gold positive is candidate
    pos_mask = labels != INVALID_ID
    if pos_mask.any():
        cand_mask[pos_mask, labels[pos_mask]] = True

    if log_every and (step_i % int(log_every) == 0):
        with torch.no_grad():
            ok_gold = cand_mask[torch.arange(labels.size(0), device=device), labels].float().mean().item()
        logger.info(
            f"[CatCand] gold_in_cand_rate={ok_gold:.4f} "
            f"N={labels.size(0)} pos={pos_mask.sum().item()} neg={(~pos_mask).sum().item()}"
        )

    # fp16-safe large negative
    logits = logits.masked_fill(~cand_mask, -1e4)

    neg_mask = labels == INVALID_ID

    loss_pos = torch.tensor(0.0, device=device)
    if pos_mask.any():
        loss_pos = F.cross_entropy(logits[pos_mask], labels[pos_mask], reduction="mean")

    loss_neg = torch.tensor(0.0, device=device)
    if neg_mask.any():
        neg_logits = logits[neg_mask]
        neg_labels = labels[neg_mask]

        if hard_neg_ratio is None or hard_neg_ratio >= 0.999:
            loss_neg = F.cross_entropy(neg_logits, neg_labels, reduction="mean")
        else:
            with torch.no_grad():
                prob = F.softmax(neg_logits, dim=-1)
                # "hardness": max non-invalid prob
                diff = prob[:, :-1].max(dim=-1).values
            k = max(1, int(diff.size(0) * float(hard_neg_ratio)))
            _, hard_idx = torch.topk(diff, k=k, largest=True)
            loss_neg = F.cross_entropy(neg_logits[hard_idx], neg_labels[hard_idx], reduction="mean")

    if pos_mask.any() and neg_mask.any():
        return loss_pos + loss_neg
    elif pos_mask.any():
        return loss_pos
    else:
        return loss_neg


def _acc_add(acc: dict, key: str, val: float):
    acc[key] = acc.get(key, 0.0) + float(val)


def _acc_mean(acc: dict, n: int):
    out = {}
    n = max(1, int(n))
    for k, v in acc.items():
        out[k] = float(v) / n
    return out


def train_one_epoch(
    model, dl, optimizer, scheduler, scaler, device, fp16,
    ent_heads, rel_heads, w_ent, w_rel, w_dim, w_dim_seq, w_sen_seq, w_cat, tok
):
    model.train()
    acc = {}
    steps = 0

    for batch in dl:
        for k in batch:
            batch[k] = batch[k].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(fp16 and device == "cuda")):
            out = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])
            logits_mat = out["matrix"]
            logits_dim = out["dimension"]
            logits_dim_seq = out["dimension_sequence"]
            logits_sen_seq = out["sentiment_sequence"]
            seq_out = out["sequence_output"]

            y_true = batch["matrix_ids"]

            loss_ent = global_pointer_crossentropy(
                y_true[:, ent_heads, :, :],
                logits_mat[:, ent_heads, :, :],
                attention_mask=batch["attention_mask"],
                tril_mask=False,
            )

            loss_rel = global_pointer_crossentropy(
                y_true[:, rel_heads, :, :],
                logits_mat[:, rel_heads, :, :],
                attention_mask=batch["attention_mask"],
                tril_mask=False,
            )

            loss_dim = nn.BCEWithLogitsLoss()(logits_dim, batch["dimension_ids"])

            token_mask = _build_token_mask(batch["input_ids"], batch["attention_mask"], tok)
            loss_dim_seq = masked_bce_with_logits(logits_dim_seq, batch["dimension_sequences"], token_mask)
            loss_sen_seq = masked_bce_with_logits(logits_sen_seq, batch["sentiment_sequences"], token_mask)

            loss_cat = compute_category_loss(
                model, seq_out, batch, device,
                hard_neg_ratio=0.3,
                log_every=200,
                step_i=steps
            )

            loss_total = (w_ent * loss_ent +
                          w_rel * loss_rel +
                          w_dim * loss_dim +
                          w_dim_seq * loss_dim_seq +
                          w_sen_seq * loss_sen_seq +
                          w_cat * loss_cat)

        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        _acc_add(acc, "ent", loss_ent.item())
        _acc_add(acc, "rel", loss_rel.item())
        _acc_add(acc, "dim", loss_dim.item())
        _acc_add(acc, "dim_seq", loss_dim_seq.item())
        _acc_add(acc, "sen_seq", loss_sen_seq.item())
        _acc_add(acc, "cat", loss_cat.item())
        _acc_add(acc, "total", loss_total.item())

        steps += 1

    return _acc_mean(acc, steps)


@torch.no_grad()
def evaluate(
    model, dl, device, fp16,
    ent_heads, rel_heads, w_ent, w_rel, w_dim, w_dim_seq, w_sen_seq, w_cat, tok
):
    model.eval()
    acc = {}
    steps = 0

    for batch in dl:
        for k in batch:
            batch[k] = batch[k].to(device)

        with autocast(enabled=(fp16 and device == "cuda")):
            out = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])
            logits_mat = out["matrix"]
            logits_dim = out["dimension"]
            logits_dim_seq = out["dimension_sequence"]
            logits_sen_seq = out["sentiment_sequence"]
            seq_out = out["sequence_output"]

            y_true = batch["matrix_ids"]

            loss_ent = global_pointer_crossentropy(
                y_true[:, ent_heads, :, :],
                logits_mat[:, ent_heads, :, :],
                attention_mask=batch["attention_mask"],
                tril_mask=False,
            )
            loss_rel = global_pointer_crossentropy(
                y_true[:, rel_heads, :, :],
                logits_mat[:, rel_heads, :, :],
                attention_mask=batch["attention_mask"],
                tril_mask=False,
            )
            loss_dim = nn.BCEWithLogitsLoss()(logits_dim, batch["dimension_ids"])

            token_mask = _build_token_mask(batch["input_ids"], batch["attention_mask"], tok)
            loss_dim_seq = masked_bce_with_logits(logits_dim_seq, batch["dimension_sequences"], token_mask)
            loss_sen_seq = masked_bce_with_logits(logits_sen_seq, batch["sentiment_sequences"], token_mask)

            loss_cat = compute_category_loss(
                model, seq_out, batch, device,
                hard_neg_ratio=1.0,
                log_every=0,
                step_i=steps
            )

            loss_total = (w_ent * loss_ent +
                          w_rel * loss_rel +
                          w_dim * loss_dim +
                          w_dim_seq * loss_dim_seq +
                          w_sen_seq * loss_sen_seq +
                          w_cat * loss_cat)

        _acc_add(acc, "ent", loss_ent.item())
        _acc_add(acc, "rel", loss_rel.item())
        _acc_add(acc, "dim", loss_dim.item())
        _acc_add(acc, "dim_seq", loss_dim_seq.item())
        _acc_add(acc, "sen_seq", loss_sen_seq.item())
        _acc_add(acc, "cat", loss_cat.item())
        _acc_add(acc, "total", loss_total.item())

        steps += 1

    return _acc_mean(acc, steps)


def _run_valid_cf1(args, ckpt_path, repo_root: Path):
    """
    Run predict.py + metrics_subtask_1_2_3.py using absolute paths
    so it works no matter where the process is launched from (e.g., tools/).
    """
    pred_path = os.path.join(args.output_dir, "pred_valid_ep.jsonl")

    predict_py = str((repo_root / "predict.py").resolve())
    metrics_py = str((repo_root / "metrics_subtask_1_2_3.py").resolve())

    train_stats_path = args.train_stats if getattr(args, "train_stats", None) else args.train

    cmd_pred = [
        sys.executable, predict_py,
        "--input", args.valid,
        "--train_stats", train_stats_path,
        "--ckpt", ckpt_path,
        "--model_name", args.model_name,
        "--max_len", str(args.max_len),
        "--batch", str(args.pred_batch),
        "--label_pattern", args.label_pattern,
        "--thr_aux", str(args.thr_aux),
        "--topk_aux", str(args.topk_aux),
        "--max_span_len", str(args.max_span_len),
        "--thr_rel", str(args.thr_rel),
        "--topk_rel", str(args.topk_rel),
        "--max_pair_dist", str(args.max_pair_dist),
        "--max_quads", str(args.max_quads),
        "--min_score", str(args.min_score),
        "--null_thr_o", str(args.null_thr_o),
        "--va_stat", args.va_stat,
        "--cat_case", args.cat_case,
        "--categories_path", args.categories_path,
        "--cat_source", args.cat_source,
        "--cat_head_min_conf", str(args.cat_head_min_conf),
        "--diag",
        "--output", pred_path,
    ]

    r1 = subprocess.run(cmd_pred, capture_output=True, text=True)
    if r1.returncode != 0:
        raise RuntimeError(f"[predict.py failed]\nSTDOUT:\n{r1.stdout}\nSTDERR:\n{r1.stderr}")

    cmd_met = [
        sys.executable, metrics_py,
        "-g", args.valid,
        "-p", pred_path,
        "-t", "3",
    ]
    r2 = subprocess.run(cmd_met, capture_output=True, text=True)
    if r2.returncode != 0:
        raise RuntimeError(f"[metrics failed]\nSTDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}")

    m = re.search(r"'cF1'\s*:\s*([0-9]*\.?[0-9]+)", r2.stdout)
    if m is None:
        m = re.search(r"\bcF1\s*:\s*([0-9]*\.?[0-9]+)", r2.stdout)
    if m is None:
        raise RuntimeError(f"Cannot parse cF1 from metrics output:\n{r2.stdout}")

    cf1 = float(m.group(1))
    return cf1, pred_path


def _fmt_losses(prefix: str, d: dict):
    keys = ["ent", "rel", "dim", "dim_seq", "sen_seq", "cat", "total"]
    parts = []
    for k in keys:
        parts.append(f"{k}={float(d.get(k, 0.0)):.4f}")
    return f"{prefix} " + " ".join(parts)


def main():
    ap = argparse.ArgumentParser()

    # ---- NEW ----
    ap.add_argument("--config", type=str, default=None, help="Path to configs/<lang>-<domain>/data.yaml")

    # paths (can be filled by config)
    ap.add_argument("--train", type=str, default=None)
    ap.add_argument("--valid", type=str, default=None)
    ap.add_argument("--test", type=str, default=None)
    ap.add_argument("--dev", type=str, default=None)

    # stats for priors (prefer train_all from config)
    ap.add_argument("--train_stats", type=str, default=None, help="Stats source for priors; prefer train_all.")

    ap.add_argument("--categories_path", type=str, default=None)

    ap.add_argument("--model_name", required=True)
    ap.add_argument("--output_dir", default="./output")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--label_pattern", type=str, default="sentiment_dim",
                    choices=["raw", "sentiment", "sentiment_dim"])

    ap.add_argument("--w_ent", type=float, default=2.0)
    ap.add_argument("--w_rel", type=float, default=1.0)
    ap.add_argument("--w_dim", type=float, default=0.2)
    ap.add_argument("--w_dim_seq", type=float, default=0.2)
    ap.add_argument("--w_sen_seq", type=float, default=0.2)
    ap.add_argument("--w_cat", type=float, default=1.0)

    # Dataset negative sampling knobs
    ap.add_argument("--neg_ratio", type=float, default=3.0)
    ap.add_argument("--neg_shift", type=int, default=1)
    ap.add_argument("--neg_max_per_sample", type=int, default=64)

    # IMPORTANT: default keep ON (align with dataset defaults); set 0 to disable.
    ap.add_argument("--neg_include_cross", type=int, default=1, choices=[0, 1])
    ap.add_argument("--neg_include_random", type=int, default=1, choices=[0, 1])

    # Early Stopping
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min_delta", type=float, default=0.0)
    ap.add_argument("--min_epochs", type=int, default=6)

    # Valid selection criterion
    ap.add_argument("--select_by", type=str, default="cf1", choices=["loss", "cf1"])

    # predict/metrics params for valid cF1 (only used when select_by=cf1)
    ap.add_argument("--pred_batch", type=int, default=8)
    ap.add_argument("--thr_aux", type=float, default=0.05)
    ap.add_argument("--topk_aux", type=int, default=80)
    ap.add_argument("--max_span_len", type=int, default=12)
    ap.add_argument("--thr_rel", type=float, default=0.12)
    ap.add_argument("--topk_rel", type=int, default=300)
    ap.add_argument("--max_pair_dist", type=int, default=80)
    ap.add_argument("--max_quads", type=int, default=2)
    ap.add_argument("--min_score", type=float, default=1.4)
    ap.add_argument("--null_thr_o", type=float, default=0.10)

    ap.add_argument("--va_stat", type=str, default="median", choices=["mode", "median", "mean"])
    ap.add_argument("--cat_case", type=str, default="upper", choices=["upper", "lower"])
    ap.add_argument("--cat_source", type=str, default="prior", choices=["prior", "head", "mix"])
    ap.add_argument("--cat_head_min_conf", type=float, default=0.55)

    args = ap.parse_args()

    # Determine repo root robustly (works when launched from tools/)
    repo_root = Path(__file__).resolve().parent

    # ---- load config and fill defaults ----
    if args.config:
        cfg = load_config(args.config, repo_root=str(repo_root))
        args = apply_cfg_defaults(args, cfg, {
            "train": "train",
            "valid": "valid",
            "test": "test",
            "dev": "dev",
            "train_all": "train_stats",        # ✅ key improvement: train_all -> train_stats
            "categories": "categories_path",
        })

    # validate required paths
    if not args.train or not args.valid:
        raise ValueError("Missing --train/--valid. Provide them directly or via --config data.yaml")
    if not args.categories_path:
        raise ValueError("Missing --categories_path. Provide it directly or via --config data.yaml")

    # stats fallback: if no train_all/train_stats, fallback to train
    if not args.train_stats:
        args.train_stats = args.train

    os.makedirs(args.output_dir, exist_ok=True)
    logger.add(os.path.join(args.output_dir, "train.log"), enqueue=True)

    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Device] {device}")
    if args.config:
        logger.info(f"[Config] {os.path.abspath(args.config)}")

    logger.info(
        f"[Paths] train={os.path.abspath(args.train)} valid={os.path.abspath(args.valid)} "
        f"train_stats={os.path.abspath(args.train_stats)} categories={os.path.abspath(args.categories_path)}"
    )

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    train_ds = AcqpDataset(
        "Train", args.train, args.max_len, tok,
        label_pattern=args.label_pattern,
        categories_path=args.categories_path,
        neg_ratio=args.neg_ratio,
        neg_shift=args.neg_shift,
        neg_max_per_sample=args.neg_max_per_sample,
        neg_include_cross=bool(args.neg_include_cross),
        neg_include_random=bool(args.neg_include_random),
    )
    valid_ds = AcqpDataset(
        "Valid", args.valid, args.max_len, tok,
        label_pattern=args.label_pattern,
        categories_path=args.categories_path,
        neg_ratio=args.neg_ratio,
        neg_shift=args.neg_shift,
        neg_max_per_sample=args.neg_max_per_sample,
        neg_include_cross=bool(args.neg_include_cross),
        neg_include_random=bool(args.neg_include_random),
    )

    logger.info(f"[Data] train={len(train_ds)} valid={len(valid_ds)}")
    logger.info(f"[Labels] num_label_types={len(train_ds.label_types)} num_dims={len(train_ds.dimension2id)} "
                f"num_cats(incl.INVALID)={train_ds.num_categories}")

    if "BA-BO" not in train_ds.label_types or "EA-EO" not in train_ds.label_types:
        raise RuntimeError(f"label_types missing BA-BO/EA-EO: {train_ds.label_types}")

    h_ba_bo = train_ds.label_types.index("BA-BO")
    h_ea_eo = train_ds.label_types.index("EA-EO")
    ent_heads = [h_ba_bo, h_ea_eo]
    rel_heads = [i for i in range(len(train_ds.label_types)) if i not in ent_heads]

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    model = QuadrupleModel(
        num_label_types=len(train_ds.label_types),
        num_dimension_types=len(train_ds.dimension2id),
        max_seq_len=args.max_len,
        pretrain_model_path=args.model_name,
        num_categories=train_ds.num_categories,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_dl) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=(args.fp16 and device == "cuda"))

    save_json(os.path.join(args.output_dir, "train_config.json"), vars(args))
    logger.info(f"[ConfigDump] saved -> {os.path.join(args.output_dir, 'train_config.json')}")

    best_path = os.path.join(args.output_dir, "best_model.pt")
    last_path = os.path.join(args.output_dir, "last_model.pt")

    best_loss = 1e18
    best_cf1 = -1.0
    best_epoch = 0
    no_improve = 0
    last_ep_ran = 0

    for ep in range(1, args.epochs + 1):
        last_ep_ran = ep

        tr = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device, args.fp16,
            ent_heads, rel_heads, args.w_ent, args.w_rel, args.w_dim, args.w_dim_seq, args.w_sen_seq, args.w_cat, tok
        )
        vl = evaluate(
            model, valid_dl, device, args.fp16,
            ent_heads, rel_heads, args.w_ent, args.w_rel, args.w_dim, args.w_dim_seq, args.w_sen_seq, args.w_cat, tok
        )

        logger.info(f"[Epoch {ep}] " + _fmt_losses("train:", tr))
        logger.info(f"[Epoch {ep}] " + _fmt_losses("valid:", vl))

        torch.save(model.state_dict(), last_path)

        if args.select_by == "loss":
            improved = (best_loss - float(vl["total"])) > float(args.min_delta)
            if improved:
                best_loss = float(vl["total"])
                best_epoch = ep
                no_improve = 0
                torch.save(model.state_dict(), best_path)
                logger.success(f"[BEST] saved -> {best_path} (valid_total_loss={best_loss:.4f})")
            else:
                no_improve += 1
        else:
            cf1, pred_path = _run_valid_cf1(args, last_path, repo_root=repo_root)
            logger.info(f"[Epoch {ep}] valid_cF1={cf1:.6f} (pred={pred_path})")

            improved = (cf1 - best_cf1) > float(args.min_delta)
            if improved:
                best_cf1 = cf1
                best_epoch = ep
                no_improve = 0
                torch.save(model.state_dict(), best_path)
                logger.success(f"[BEST] saved -> {best_path} (valid_cF1={best_cf1:.6f})")
            else:
                no_improve += 1

        if args.early_stop and ep >= int(args.min_epochs):
            logger.info(f"[EarlyStop] no_improve={no_improve}/{args.patience} (min_delta={args.min_delta})")
            if no_improve >= int(args.patience):
                if args.select_by == "loss":
                    logger.warning(f"[EarlyStop] triggered at epoch={ep}. Best valid_total_loss={best_loss:.4f}")
                else:
                    logger.warning(f"[EarlyStop] triggered at epoch={ep}. Best valid_cF1={best_cf1:.6f}")
                break

    # machine-readable best summary (for tools/run_train_from_yaml.py)
    best_metrics = {
        "select_by": args.select_by,
        "best_loss": float(best_loss) if best_loss < 1e17 else None,
        "best_cf1": float(best_cf1) if best_cf1 >= 0 else None,
        "best_epoch": int(best_epoch),
        "best_path": os.path.abspath(best_path),
        "last_path": os.path.abspath(last_path),
        "epochs_ran": int(last_ep_ran),
        "config": os.path.abspath(args.config) if args.config else None,
        "train": os.path.abspath(args.train),
        "valid": os.path.abspath(args.valid),
        "train_stats": os.path.abspath(args.train_stats) if args.train_stats else None,
        "categories_path": os.path.abspath(args.categories_path),
    }
    with open(os.path.join(args.output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"[BestMetrics] saved -> {os.path.join(args.output_dir, 'best_metrics.json')}")

    logger.success("Training finished.")


if __name__ == "__main__":
    main()
