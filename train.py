# -*- coding: utf-8 -*-
import os, json, argparse
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from loguru import logger
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset.dataset import AcqpDataset, collate_fn
from losses.losses import global_pointer_crossentropy
from models.model import QuadrupleModel
from utils.utils import set_seeds


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def train_one_epoch(model, dl, optimizer, scheduler, scaler, device, fp16, ent_heads, rel_heads, w_ent, w_rel, w_dim):
    model.train()
    total = 0.0

    for batch in dl:
        for k in batch:
            batch[k] = batch[k].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(fp16 and device == "cuda")):
            out = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])
            logits_mat = out["matrix"]          # logits
            logits_dim = out["dimension"]       # logits

            y_true = batch["matrix_ids"]

            # 实体头：BA-BO / EA-EO
            loss_ent = global_pointer_crossentropy(
                y_true[:, ent_heads, :, :],
                logits_mat[:, ent_heads, :, :],
                attention_mask=batch["attention_mask"],
                tril_mask=False,
            )

            # 其余关系头（sentiment_dim 9 heads）
            loss_rel = global_pointer_crossentropy(
                y_true[:, rel_heads, :, :],
                logits_mat[:, rel_heads, :, :],
                attention_mask=batch["attention_mask"],
                tril_mask=False,
            )

            loss_dim = nn.BCEWithLogitsLoss()(logits_dim, batch["dimension_ids"])
            loss = w_ent * loss_ent + w_rel * loss_rel + w_dim * loss_dim

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total += loss.item()

    return total / max(1, len(dl))


@torch.no_grad()
def evaluate(model, dl, device, fp16, ent_heads, rel_heads, w_ent, w_rel, w_dim):
    model.eval()
    total = 0.0
    for batch in dl:
        for k in batch:
            batch[k] = batch[k].to(device)

        with autocast(enabled=(fp16 and device == "cuda")):
            out = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])
            logits_mat = out["matrix"]
            logits_dim = out["dimension"]
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
            loss = w_ent * loss_ent + w_rel * loss_rel + w_dim * loss_dim

        total += loss.item()

    return total / max(1, len(dl))


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

    # loss weights
    ap.add_argument("--w_ent", type=float, default=2.0)
    ap.add_argument("--w_rel", type=float, default=1.0)
    ap.add_argument("--w_dim", type=float, default=0.5)

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.add(os.path.join(args.output_dir, "train.log"), enqueue=True)

    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Device] {device}")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = AcqpDataset("Train", args.train, args.max_len, tok, label_pattern="sentiment_dim")
    valid_ds = AcqpDataset("Valid", args.valid, args.max_len, tok, label_pattern="sentiment_dim")

    logger.info(f"[Data] train={len(train_ds)} valid={len(valid_ds)}")
    logger.info(f"[Labels] num_label_types={len(train_ds.label_types)} num_dims={len(train_ds.dimension2id)}")

    # head split
    try:
        h_ba_bo = train_ds.label_types.index("BA-BO")
        h_ea_eo = train_ds.label_types.index("EA-EO")
    except ValueError:
        raise RuntimeError(f"label_types missing BA-BO/EA-EO. label_types={train_ds.label_types}")

    ent_heads = [h_ba_bo, h_ea_eo]
    rel_heads = [i for i in range(len(train_ds.label_types)) if i not in ent_heads]

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    model = QuadrupleModel(
        num_label_types=len(train_ds.label_types),
        num_dimension_types=len(train_ds.dimension2id),
        max_seq_len=args.max_len,
        pretrain_model_path=args.model_name,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_dl) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=(args.fp16 and device == "cuda"))

    # save config
    save_json(os.path.join(args.output_dir, "train_config.json"), vars(args))
    logger.info(f"[Config] saved -> {os.path.join(args.output_dir, 'train_config.json')}")

    best = 1e18
    best_path = os.path.join(args.output_dir, "best_model.pt")

    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_dl, optimizer, scheduler, scaler, device, args.fp16,
                             ent_heads, rel_heads, args.w_ent, args.w_rel, args.w_dim)
        vl = evaluate(model, valid_dl, device, args.fp16,
                      ent_heads, rel_heads, args.w_ent, args.w_rel, args.w_dim)

        logger.info(f"[Epoch {ep}] train_loss={tr:.4f} valid_loss={vl:.4f}")

        if vl < best:
            best = vl
            torch.save(model.state_dict(), best_path)
            logger.success(f"[BEST] saved -> {best_path} (valid_loss={vl:.4f})")

    logger.success("Training finished.")


if __name__ == "__main__":
    main()
