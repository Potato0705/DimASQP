# -*- coding: utf-8 -*-
"""
DimASQP / SemEval Task3 - Train (V2.1)

在你当前版本基础上做的关键增强：
1) Early Stopping（按 valid_loss，patience 可配）
2) Gradient Accumulation（--grad_accum，扩大有效 batch，训练更稳）
3) 更快的数据加载（num_workers/pin_memory/persistent_workers 可配）
4) 训练日志更细：输出 ent/rel/dim/bnd 四个分量
5) step 级 scheduler：按“更新步数”而不是“batch数”计算 total_steps

注意：仍然按 valid_loss 保存 best_model.pt（后续若要按 cF1 保存，再把 predict+metrics 挂进来）
"""

import os
import json
import math
import time
import argparse

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from loguru import logger
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset.dataset import AcqpDataset, collate_fn
from losses.losses import global_pointer_crossentropy, boundary_bce_loss
from models.model import QuadrupleModel
from utils.utils import set_seeds


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_valid_pos(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor, keep_cls=True):
    """
    与 predict 统一：保留 CLS=0，仅屏蔽 SEP/PAD
    returns: valid_pos [B,L] bool
    """
    valid = attention_mask.bool().clone()
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    if sep_id is not None:
        valid &= (input_ids != sep_id)
    if pad_id is not None:
        valid &= (input_ids != pad_id)
    if keep_cls:
        valid[:, 0] = True
    return valid


def build_pair_mask(valid_pos: torch.Tensor) -> torch.Tensor:
    # [B,L] -> [B,L,L]
    return valid_pos.unsqueeze(2) & valid_pos.unsqueeze(1)


def build_rel_pair_mask(valid_pos: torch.Tensor, max_pair_dist: int):
    """
    relation head 的候选约束：|i-j|<=max_pair_dist，或者 i==0 或 j==0（允许 NULL）
    """
    mask = build_pair_mask(valid_pos)
    if max_pair_dist is None or max_pair_dist <= 0:
        return mask
    B, L = valid_pos.shape
    idx = torch.arange(L, device=valid_pos.device)
    dist = (idx[None, :] - idx[:, None]).abs()  # [L,L]
    allow = (dist <= max_pair_dist) | (idx[:, None] == 0) | (idx[None, :] == 0)
    return mask & allow.unsqueeze(0).expand(B, L, L)


def compute_boundary_logits_from_heads(logits_mat: torch.Tensor, h_ba_bo: int, h_ea_eo: int):
    """
    logits_mat: [B,C,L,L]
    horns 对应：
      BA-BO: (a_start, o_start)
        a_start=max over o_start; o_start=max over a_start
      EA-EO: (a_end, o_end)
        a_end=max over o_end;   o_end=max over a_end
    returns boundary logits: a_start/o_start/a_end/o_end each [B,L]
    """
    ba_bo = logits_mat[:, h_ba_bo]  # [B,L,L]
    ea_eo = logits_mat[:, h_ea_eo]  # [B,L,L]
    a_start = ba_bo.max(dim=2).values
    o_start = ba_bo.max(dim=1).values
    a_end = ea_eo.max(dim=2).values
    o_end = ea_eo.max(dim=1).values
    return a_start, o_start, a_end, o_end


def train_one_epoch(
    model, dl, optimizer, scheduler, scaler, device, fp16,
    ent_heads, rel_heads,
    w_ent, w_rel, w_dim, w_bnd,
    tokenizer,
    train_max_pair_dist,
    neg_topk_ent, neg_topk_rel,
    grad_accum: int = 1,
    clip_norm: float = 1.0,
    log_steps: int = 0,
):
    model.train()

    total_loss = 0.0
    total_ent = 0.0
    total_rel = 0.0
    total_dim = 0.0
    total_bnd = 0.0

    step_updates = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dl, start=1):
        for k in batch:
            batch[k] = batch[k].to(device)

        valid_pos = build_valid_pos(tokenizer, batch["input_ids"], batch["attention_mask"], keep_cls=True)
        pair_mask_ent = build_pair_mask(valid_pos)
        pair_mask_rel = build_rel_pair_mask(valid_pos, train_max_pair_dist)

        with autocast(enabled=(fp16 and device.type == "cuda")):
            out = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])
            logits_mat = out["matrix"]          # [B,C,L,L]
            logits_dim = out["dimension"]       # [B,D]
            y_true = batch["matrix_ids"]        # [B,C,L,L]

            # 主损失：实体头与关系头分开 + hard negatives + 候选约束
            loss_ent = global_pointer_crossentropy(
                y_true[:, ent_heads, :, :],
                logits_mat[:, ent_heads, :, :],
                attention_mask=None,
                pair_mask=pair_mask_ent,
                tril_mask=False,
                neg_topk=neg_topk_ent,
            )
            loss_rel = global_pointer_crossentropy(
                y_true[:, rel_heads, :, :],
                logits_mat[:, rel_heads, :, :],
                attention_mask=None,
                pair_mask=pair_mask_rel,
                tril_mask=False,
                neg_topk=neg_topk_rel,
            )
            loss_dim = nn.BCEWithLogitsLoss()(logits_dim, batch["dimension_ids"])

            # 轻量 span 正则：边界 BCE（稳定 span 边界）
            h_ba_bo, h_ea_eo = ent_heads[0], ent_heads[1]
            a_start_logit, o_start_logit, a_end_logit, o_end_logit = compute_boundary_logits_from_heads(
                logits_mat, h_ba_bo, h_ea_eo
            )
            bnd = 0.0
            bnd += boundary_bce_loss(a_start_logit, batch["a_start_ids"], valid_pos)
            bnd += boundary_bce_loss(o_start_logit, batch["o_start_ids"], valid_pos)
            bnd += boundary_bce_loss(a_end_logit, batch["a_end_ids"], valid_pos)
            bnd += boundary_bce_loss(o_end_logit, batch["o_end_ids"], valid_pos)
            loss_bnd = bnd / 4.0

            loss = w_ent * loss_ent + w_rel * loss_rel + w_dim * loss_dim + w_bnd * loss_bnd

            # gradient accumulation：缩放 loss
            loss_scaled = loss / max(1, grad_accum)

        scaler.scale(loss_scaled).backward()

        total_loss += float(loss.item())
        total_ent += float(loss_ent.item())
        total_rel += float(loss_rel.item())
        total_dim += float(loss_dim.item())
        total_bnd += float(loss_bnd.item())

        do_step = (step % max(1, grad_accum) == 0) or (step == len(dl))
        if do_step:
            scaler.unscale_(optimizer)
            if clip_norm and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            step_updates += 1

        if log_steps and log_steps > 0 and (step % log_steps == 0):
            logger.info(
                f"[Train step {step}/{len(dl)}] "
                f"loss={total_loss/step:.4f} ent={total_ent/step:.4f} rel={total_rel/step:.4f} "
                f"dim={total_dim/step:.4f} bnd={total_bnd/step:.4f}"
            )

    denom = max(1, len(dl))
    return {
        "loss": total_loss / denom,
        "ent": total_ent / denom,
        "rel": total_rel / denom,
        "dim": total_dim / denom,
        "bnd": total_bnd / denom,
        "updates": step_updates,
    }


@torch.no_grad()
def evaluate(
    model, dl, device, fp16,
    ent_heads, rel_heads,
    w_ent, w_rel, w_dim, w_bnd,
    tokenizer,
    train_max_pair_dist,
    neg_topk_ent, neg_topk_rel
):
    model.eval()

    total_loss = 0.0
    total_ent = 0.0
    total_rel = 0.0
    total_dim = 0.0
    total_bnd = 0.0

    for batch in dl:
        for k in batch:
            batch[k] = batch[k].to(device)

        valid_pos = build_valid_pos(tokenizer, batch["input_ids"], batch["attention_mask"], keep_cls=True)
        pair_mask_ent = build_pair_mask(valid_pos)
        pair_mask_rel = build_rel_pair_mask(valid_pos, train_max_pair_dist)

        with autocast(enabled=(fp16 and device.type == "cuda")):
            out = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])
            logits_mat = out["matrix"]
            logits_dim = out["dimension"]
            y_true = batch["matrix_ids"]

            loss_ent = global_pointer_crossentropy(
                y_true[:, ent_heads, :, :],
                logits_mat[:, ent_heads, :, :],
                attention_mask=None,
                pair_mask=pair_mask_ent,
                tril_mask=False,
                neg_topk=neg_topk_ent,
            )
            loss_rel = global_pointer_crossentropy(
                y_true[:, rel_heads, :, :],
                logits_mat[:, rel_heads, :, :],
                attention_mask=None,
                pair_mask=pair_mask_rel,
                tril_mask=False,
                neg_topk=neg_topk_rel,
            )
            loss_dim = nn.BCEWithLogitsLoss()(logits_dim, batch["dimension_ids"])

            h_ba_bo, h_ea_eo = ent_heads[0], ent_heads[1]
            a_start_logit, o_start_logit, a_end_logit, o_end_logit = compute_boundary_logits_from_heads(
                logits_mat, h_ba_bo, h_ea_eo
            )
            bnd = 0.0
            bnd += boundary_bce_loss(a_start_logit, batch["a_start_ids"], valid_pos)
            bnd += boundary_bce_loss(o_start_logit, batch["o_start_ids"], valid_pos)
            bnd += boundary_bce_loss(a_end_logit, batch["a_end_ids"], valid_pos)
            bnd += boundary_bce_loss(o_end_logit, batch["o_end_ids"], valid_pos)
            loss_bnd = bnd / 4.0

            loss = w_ent * loss_ent + w_rel * loss_rel + w_dim * loss_dim + w_bnd * loss_bnd

        total_loss += float(loss.item())
        total_ent += float(loss_ent.item())
        total_rel += float(loss_rel.item())
        total_dim += float(loss_dim.item())
        total_bnd += float(loss_bnd.item())

    denom = max(1, len(dl))
    return {
        "loss": total_loss / denom,
        "ent": total_ent / denom,
        "rel": total_rel / denom,
        "dim": total_dim / denom,
        "bnd": total_bnd / denom,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
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

    # loss weights
    ap.add_argument("--w_ent", type=float, default=2.0)
    ap.add_argument("--w_rel", type=float, default=1.0)
    ap.add_argument("--w_dim", type=float, default=0.5)
    ap.add_argument("--w_bnd", type=float, default=0.15)

    # cleaner negatives / candidate constraint
    ap.add_argument("--train_max_pair_dist", type=int, default=120)
    ap.add_argument("--neg_topk_ent", type=int, default=2048)
    ap.add_argument("--neg_topk_rel", type=int, default=4096)

    # training stability / speed
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--clip_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--log_steps", type=int, default=0)

    # early stopping (按 valid_loss)
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min_delta", type=float, default=0.0)

    # save last
    ap.add_argument("--save_last", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.add(os.path.join(args.output_dir, "train.log"), enqueue=True)

    set_seeds(args.seed)

    # 一些速度/稳定性开关（不改逻辑）
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Device] {device.type}")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    train_ds = AcqpDataset("Train", args.train, args.max_len, tok, label_pattern=args.label_pattern)
    valid_ds = AcqpDataset("Valid", args.valid, args.max_len, tok, label_pattern=args.label_pattern)

    logger.info(f"[Data] train={len(train_ds)} valid={len(valid_ds)}")
    logger.info(f"[Labels] num_label_types={len(train_ds.label_types)} num_dims={len(train_ds.dimension2id)}")

    try:
        h_ba_bo = train_ds.label_types.index("BA-BO")
        h_ea_eo = train_ds.label_types.index("EA-EO")
    except ValueError:
        raise RuntimeError(f"label_types missing BA-BO/EA-EO. label_types={train_ds.label_types}")

    ent_heads = [h_ba_bo, h_ea_eo]
    rel_heads = [i for i in range(len(train_ds.label_types)) if i not in ent_heads]

    pin_memory = bool(args.pin_memory and device.type == "cuda")
    num_workers = max(0, int(args.num_workers))

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    model = QuadrupleModel(
        num_label_types=len(train_ds.label_types),
        num_dimension_types=len(train_ds.dimension2id),
        max_seq_len=args.max_len,
        pretrain_model_path=args.model_name,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # total_steps 按“更新步数”计算（考虑 grad_accum）
    steps_per_epoch = math.ceil(len(train_dl) / max(1, args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=(args.fp16 and device.type == "cuda"))

    save_json(os.path.join(args.output_dir, "train_config.json"), vars(args))
    logger.info(f"[Config] saved -> {os.path.join(args.output_dir, 'train_config.json')}")
    logger.info(f"[Steps] steps_per_epoch={steps_per_epoch} total_steps={total_steps} warmup_steps={warmup_steps}")

    best = float("inf")
    best_path = os.path.join(args.output_dir, "best_model.pt")
    last_path = os.path.join(args.output_dir, "last_model.pt")

    bad_epochs = 0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device, args.fp16,
            ent_heads, rel_heads,
            args.w_ent, args.w_rel, args.w_dim, args.w_bnd,
            tok,
            args.train_max_pair_dist,
            args.neg_topk_ent, args.neg_topk_rel,
            grad_accum=args.grad_accum,
            clip_norm=args.clip_norm,
            log_steps=args.log_steps,
        )

        vl = evaluate(
            model, valid_dl, device, args.fp16,
            ent_heads, rel_heads,
            args.w_ent, args.w_rel, args.w_dim, args.w_bnd,
            tok,
            args.train_max_pair_dist,
            args.neg_topk_ent, args.neg_topk_rel,
        )

        dt = time.time() - t0

        logger.info(
            f"[Epoch {ep}] time={dt:.1f}s "
            f"train_loss={tr['loss']:.4f} (ent={tr['ent']:.4f} rel={tr['rel']:.4f} dim={tr['dim']:.4f} bnd={tr['bnd']:.4f}) "
            f"valid_loss={vl['loss']:.4f} (ent={vl['ent']:.4f} rel={vl['rel']:.4f} dim={vl['dim']:.4f} bnd={vl['bnd']:.4f}) "
            f"updates={tr['updates']}"
        )

        # 保存 best
        if vl["loss"] < (best - float(args.min_delta)):
            best = vl["loss"]
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            logger.success(f"[BEST] saved -> {best_path} (valid_loss={vl['loss']:.4f})")
        else:
            bad_epochs += 1
            logger.info(f"[EarlyStop] no improvement: bad_epochs={bad_epochs}/{args.patience}")

        # 可选保存 last
        if args.save_last:
            torch.save(model.state_dict(), last_path)

        # early stopping
        if args.early_stop and bad_epochs >= int(args.patience):
            logger.warning(
                f"[EarlyStop] triggered at epoch {ep}: best_valid_loss={best:.4f}, "
                f"patience={args.patience}, min_delta={args.min_delta}"
            )
            break

    logger.success("Training finished.")


if __name__ == "__main__":
    main()
