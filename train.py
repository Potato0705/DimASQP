"""
@Time : 2022/12/1717:11
@Auth : zhoujx
@File ：main.py
@DESCRIPTION:

"""
import json
import os
import random
import time

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset.dataset import AcqpDataset, collate_fn
from losses.losses import global_pointer_crossentropy
from models.layers import MetricsCalculator
from models.model import QuadrupleModel
from utils.adversarial import FGM
from utils.argparse import get_argparse
from utils.utils import make_output_dir, dump_args, set_seeds

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Use {device} device")

RESUME_STATE_NAME = "resume_checkpoint.pt"
RESUME_ARG_KEYS = (
    "task_domain",
    "train_data",
    "valid_data",
    "nrows",
    "max_seq_len",
    "split_word",
    "label_pattern",
    "use_efficient_global_pointer",
    "model_name_or_path",
    "head_size",
    "dropout_rate",
    "mode",
    "mask_rate",
    "epoch",
    "weight1",
    "weight2",
    "weight3",
    "weight4",
    "early_stop",
    "per_gpu_train_batch_size",
    "gradient_accumulation_steps",
    "use_amp",
    "with_adversarial_training",
    "encoder_learning_rate",
    "task_learning_rate",
    "weight_decay",
    "adam_epsilon",
    "max_grad_norm",
    "va_mode",
    "use_va_prior_aux",
    "weight_va_prior",
    "use_va_contrastive",
    "weight_va_cl",
    "seed",
)


def set_optimer(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    encoder_param_optimizer = list(model.encoder.named_parameters())
    logger.info(f"encoder参数: {[name for name, _ in encoder_param_optimizer]}")
    task_param_optimizer = [x for x in list(model.named_parameters()) if 'encoder' not in x[0]]
    logger.info(f"任务层参数: {[name for name, _ in task_param_optimizer]}")

    optimizer_grouped_parameters = [
        {'params': [p for n, p in encoder_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.encoder_learning_rate},
        {'params': [p for n, p in encoder_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.encoder_learning_rate},
        {'params': [p for n, p in task_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.task_learning_rate},
        {'params': [p for n, p in task_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.task_learning_rate},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.task_learning_rate,
                      weight_decay=args.weight_decay
                      # eps=args.adam_epsilon
                      )
    return optimizer


def compute_va_loss(pred_va, va_targets, va_mask):
    """Masked MSE loss for VA regression. Only computes on positions with VA annotations."""
    # pred_va: [B, L, 2], va_targets: [B, L, 2], va_mask: [B, L]
    mask = va_mask.unsqueeze(-1)  # [B, L, 1]
    diff = (pred_va - va_targets) * mask
    n_valid = mask.sum().clamp(min=1.0)
    mse = (diff ** 2).sum() / n_valid
    return mse


def compute_va_contrastive_loss(va_cl_embeds, quad_va, quad_mask):
    """VA-aware contrastive loss: enforce representation distance ∝ VA distance.

    For all valid quad pairs (i, j) across the batch:
      sim_pred(i,j) = cosine(embed_i, embed_j)       -- in [-1, 1]
      sim_gold(i,j) = 1 - ||VA_i - VA_j|| / sqrt(128) -- in [0, 1]
    Loss = MSE(sim_pred, sim_gold)

    This encourages quads with similar VA to cluster together and quads with
    different VA to separate in the projected representation space.
    """
    B, Q = quad_mask.shape
    device = va_cl_embeds.device

    # Vectorized flatten: boolean indexing replaces nested Python for-loop
    valid = quad_mask.bool()  # [B, Q]
    n = valid.sum().item()
    if n < 2:
        return torch.tensor(0.0, device=device)

    embeds = va_cl_embeds[valid]  # [N, proj_dim]
    va_vals = quad_va[valid]      # [N, 2]

    # Pairwise cosine similarity (embeds are already L2-normalized), scaled to [0, 1]
    sim_pred = (torch.mm(embeds, embeds.t()) + 1.0) / 2.0  # [N, N], in [0, 1]

    # Pairwise gold VA similarity
    va_diff = va_vals.unsqueeze(0) - va_vals.unsqueeze(1)  # [N, N, 2]
    va_dist = va_diff.norm(dim=-1)  # [N, N], Euclidean distance
    sqrt_128 = 11.3137  # sqrt(128), max possible distance in VA space
    sim_gold = 1.0 - va_dist / sqrt_128  # [N, N], in [0, 1]

    # MSE between predicted and gold similarity matrices
    loss = torch.nn.functional.mse_loss(sim_pred, sim_gold)
    return loss


def _safe_loss(loss_tensor, name="loss", max_val=1e4):
    """Clamp a loss value and replace NaN/Inf with a safe fallback. Returns (loss, was_anomalous)."""
    if torch.isnan(loss_tensor) or torch.isinf(loss_tensor):
        logger.warning(f"[LossSafety] {name} is NaN/Inf — replaced with 0")
        return torch.tensor(0.0, device=loss_tensor.device, dtype=loss_tensor.dtype), True
    if loss_tensor.item() > max_val:
        logger.warning(f"[LossSafety] {name}={loss_tensor.item():.2f} exceeds {max_val} — clamped")
        return loss_tensor.clamp(max=max_val), True
    return loss_tensor, False


def compute_loss(args, pred, data, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn):
    """Compute total loss based on label_pattern. Returns (loss, loss_mat, loss_cls_dim, loss_va, loss_cl)."""
    loss_mat = loss_matrix_fn(y_true=data["matrix_ids"], y_pred=pred["matrix"], mask_rate=args.mask_rate)
    loss_mat, _ = _safe_loss(loss_mat, "loss_mat")
    loss_cls_dim = loss_cls_dim_fn(target=data['dimension_ids'], input=pred['dimension'])
    loss_cls_dim, _ = _safe_loss(loss_cls_dim, "loss_cls_dim")
    loss_va = torch.tensor(0.0, device=loss_mat.device)
    loss_cl = torch.tensor(0.0, device=loss_mat.device)
    if args.label_pattern == 'sentiment_dim':
        loss = args.weight1 * loss_mat + args.weight2 * loss_cls_dim
    elif args.label_pattern == 'sentiment':
        loss_dim_seq = loss_dim_seq_fn(y_true=data["dimension_sequences"], y_pred=pred["dimension_sequence"],
                                       mask_rate=args.mask_rate)
        loss_dim_seq, _ = _safe_loss(loss_dim_seq, "loss_dim_seq")
        loss = args.weight1 * loss_mat + args.weight2 * loss_cls_dim + args.weight3 * loss_dim_seq
    elif args.label_pattern == 'raw':
        loss_dim_seq = loss_dim_seq_fn(y_true=data["dimension_sequences"], y_pred=pred["dimension_sequence"],
                                       mask_rate=args.mask_rate)
        loss_dim_seq, _ = _safe_loss(loss_dim_seq, "loss_dim_seq")
        loss_sen_seq = loss_dim_seq_fn(y_true=data["sentiment_sequences"], y_pred=pred["sentiment_sequence"],
                                       mask_rate=args.mask_rate)
        loss_sen_seq, _ = _safe_loss(loss_sen_seq, "loss_sen_seq")
        loss = args.weight1 * loss_mat + args.weight2 * loss_cls_dim + args.weight3 * loss_dim_seq + args.weight4 * loss_sen_seq
    elif args.label_pattern == 'category':
        loss = args.weight1 * loss_mat + args.weight2 * loss_cls_dim
        if args.weight4 > 0:
            va_mode = getattr(args, 'va_mode', 'position')
            if va_mode in ('span_pair', 'opinion_guided') and "span_va" in pred:
                # Span-Pair / Opinion-Guided VA loss (masked MSE over gold quadruplets)
                span_va_pred = pred["span_va"]                    # [B, Q, 2]
                span_va_gold = data["quad_va"].to(span_va_pred.device)   # [B, Q, 2]
                mask = data["quad_mask"].to(span_va_pred.device).unsqueeze(-1)  # [B, Q, 1]
                diff = (span_va_pred - span_va_gold) * mask
                n_valid = mask.sum().clamp(min=1.0)
                loss_va = (diff ** 2).sum() / n_valid

                # Opinion prior auxiliary loss:
                # - always enabled for opinion_guided
                # - explicitly enabled for span_pair via --use_va_prior_aux
                use_prior_aux = va_mode == 'opinion_guided' or getattr(args, 'use_va_prior_aux', False)
                if use_prior_aux and getattr(args, 'weight_va_prior', 0.0) > 0 and "va_prior" in pred:
                    va_prior_pred = pred["va_prior"]              # [B, Q, 2]
                    diff_prior = (va_prior_pred - span_va_gold) * mask
                    loss_va_prior = (diff_prior ** 2).sum() / n_valid
                    weight_prior = getattr(args, 'weight_va_prior', 0.3)
                    loss_va = loss_va + weight_prior * loss_va_prior
            else:
                # Per-position VA loss (original)
                loss_va = compute_va_loss(pred["va"], data["va_targets"], data["va_mask"])
            loss_va, _ = _safe_loss(loss_va, "loss_va")
            loss = loss + args.weight4 * loss_va

        # VA-Aware Contrastive loss (optional, training only)
        use_va_cl = getattr(args, 'use_va_contrastive', False)
        if use_va_cl and "va_cl_embeds" in pred:
            loss_cl = compute_va_contrastive_loss(
                pred["va_cl_embeds"],
                data["quad_va"].to(pred["va_cl_embeds"].device),
                data["quad_mask"].to(pred["va_cl_embeds"].device),
            )
            loss_cl, _ = _safe_loss(loss_cl, "loss_va_cl")
            weight_cl = getattr(args, 'weight_va_cl', 0.1)
            loss = loss + weight_cl * loss_cl

    # Final total loss safety check
    loss, _ = _safe_loss(loss, "total_loss")
    return loss, loss_mat, loss_cls_dim, loss_va, loss_cl


def _progress_bar(current, total, width=25):
    """Render a simple ASCII progress bar string."""
    filled = int(width * current / total)
    bar = '#' * filled + '.' * (width - filled)
    return f'[{bar}] {current}/{total}'


def _is_resume_requested(args):
    model_path = str(getattr(args, "model_path", 0)).strip()
    return model_path not in ("", "0", "None", "none")


def _normalize_resume_value(value):
    if isinstance(value, float):
        return round(value, 12)
    return value


def _load_json_file(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _assert_resume_args_compatible(args, saved_args):
    mismatches = []
    for key in RESUME_ARG_KEYS:
        current_value = _normalize_resume_value(getattr(args, key, None))
        saved_value = _normalize_resume_value(saved_args.get(key))
        if current_value != saved_value:
            mismatches.append(f"{key}: saved={saved_value} current={current_value}")

    if mismatches:
        mismatch_text = "\n".join(mismatches[:20])
        if len(mismatches) > 20:
            mismatch_text += f"\n... and {len(mismatches) - 20} more"
        raise ValueError(
            "Resume 参数与原训练不一致，已拒绝恢复。请保持训练参数完全一致。\n"
            + mismatch_text
        )


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _move_optimizer_state_to_device(optimizer, target_device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(target_device)


def _save_resume_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        scaler,
        completed_epochs,
        best_score,
        best_epoch,
        early_stop_count,
        train_history,
        epoch_times,
        elapsed_training_seconds):
    checkpoint = {
        "completed_epochs": completed_epochs,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "early_stop_count": early_stop_count,
        "train_history": train_history,
        "epoch_times": epoch_times,
        "elapsed_training_seconds": elapsed_training_seconds,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": _capture_rng_state(),
    }
    tmp_path = checkpoint_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def _load_resume_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer, device)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    _restore_rng_state(checkpoint.get("rng_state"))
    return checkpoint


def _load_model_weights_into_model(model, model_path):
    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, nn.Module):
        model.load_state_dict(loaded.state_dict())
    elif isinstance(loaded, dict):
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            model.load_state_dict(loaded["state_dict"])
        else:
            model.load_state_dict(loaded)
    else:
        raise TypeError(f"Unsupported model file format: {type(loaded)}")


def _load_legacy_resume_state(model_dir, model):
    model_pt_path = os.path.join(model_dir, "model.pt")
    best_model_path = os.path.join(model_dir, "best_model.pt")
    best_score_path = os.path.join(model_dir, "best_score.json")
    history_path = os.path.join(model_dir, "train_history.json")

    if not os.path.exists(history_path) or not os.path.exists(best_score_path):
        raise FileNotFoundError(
            f"Legacy resume 失败：目录缺少 train_history.json 或 best_score.json: {model_dir}"
        )

    train_history = _load_json_file(history_path)
    best_info = _load_json_file(best_score_path)
    best_score = float(best_info.get("best_score", 0.0))
    best_epoch = int(best_info.get("best_epoch", 0))
    history_count = len(train_history)
    epoch_times = [float(x.get("epoch_time", 0.0)) for x in train_history if isinstance(x, dict)]

    if os.path.exists(model_pt_path):
        _load_model_weights_into_model(model, model_pt_path)
        completed_epochs = history_count
        early_stop_count = max(0, history_count - best_epoch)
        resume_history = list(train_history)
        resume_epoch_times = list(epoch_times)
        source = model_pt_path
        logger.warning(
            f"Resume checkpoint 缺失，使用 legacy model.pt 恢复：{model_pt_path}"
        )
    elif os.path.exists(best_model_path):
        _load_model_weights_into_model(model, best_model_path)
        completed_epochs = best_epoch
        early_stop_count = 0
        resume_history = list(train_history[:best_epoch])
        resume_epoch_times = list(epoch_times[:best_epoch])
        source = best_model_path
        logger.warning(
            "Resume checkpoint 与 model.pt 都缺失，退回到 best_model.pt 恢复；"
            "将从 best_epoch 之后继续训练。"
        )
    else:
        raise FileNotFoundError(
            f"Legacy resume 失败：目录既没有 model.pt 也没有 best_model.pt: {model_dir}"
        )

    return {
        "completed_epochs": completed_epochs,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "early_stop_count": early_stop_count,
        "train_history": resume_history,
        "epoch_times": resume_epoch_times,
        "elapsed_training_seconds": sum(resume_epoch_times),
        "source": source,
        "is_legacy_resume": True,
    }


def train(args, dataloader, model, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn, optimizer, scaler,
          adversial_model=None, **kwargs):
    model.train()
    total_loss = 0
    total_loss_mat = 0
    total_loss_cls_dim = 0
    total_loss_va = 0
    total_loss_cl = 0
    total_step = len(dataloader)
    accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
    # Print at every 10%
    print_steps = set(max(0, int(total_step * p) - 1) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    t_start = time.time()

    for step, data in enumerate(dataloader):
        for key in data:
            data[key] = data[key].to(device)

        # Build forward kwargs (pass span info when using span_pair/opinion_guided VA)
        fwd_kwargs = dict(input_ids=data["input_ids"],
                          token_type_ids=data["token_type_ids"],
                          attention_mask=data["attention_mask"])
        va_mode = getattr(args, 'va_mode', 'position')
        if va_mode in ('span_pair', 'opinion_guided') and args.label_pattern == 'category' and args.weight4 > 0:
            fwd_kwargs["quad_spans"] = data["quad_spans"]
            fwd_kwargs["quad_mask"] = data["quad_mask"]
            fwd_kwargs["va_mode"] = va_mode

        # Compute prediction error
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            pred = model(**fwd_kwargs)
            loss, loss_mat, loss_cls_dim, loss_va, loss_cl = compute_loss(args, pred, data, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn)
            loss = loss / accum_steps

        # Backpropagation
        scaler.scale(loss).backward()

        # adversial_training
        if adversial_model is not None:
            adversial_model.attack()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(**fwd_kwargs)
                loss_adv, _, _, _, _ = compute_loss(args, pred, data, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn)
                loss_adv = loss_adv / accum_steps

            scaler.scale(loss_adv).backward()
            adversial_model.restore()

        total_loss += loss.item() * accum_steps
        total_loss_mat += loss_mat.item()
        total_loss_cls_dim += loss_cls_dim.item()
        if args.label_pattern == 'category' and args.weight4 > 0:
            total_loss_va += loss_va.item()
        if getattr(args, 'use_va_contrastive', False):
            total_loss_cl += loss_cl.item()

        # Gradient accumulation: only step every accum_steps
        if (step + 1) % accum_steps == 0 or (step + 1) == total_step:
            if args.use_amp:
                scaler.unscale_(optimizer)

            # Check for NaN/Inf gradients before clipping
            grad_ok = True
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    grad_ok = False
                    break

            if not grad_ok:
                logger.warning(f"[GradSafety] NaN/Inf gradients at step {step+1} — skipping optimizer step")
                scaler.update()
                optimizer.zero_grad()
            else:
                max_grad_norm = getattr(args, 'max_grad_norm', 1.0)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        if step in print_steps:
            n = step + 1
            elapsed = time.time() - t_start
            speed = n / elapsed
            parts = [f"mat={total_loss_mat/n:.4f}", f"dim={total_loss_cls_dim/n:.4f}"]
            if args.label_pattern == 'category' and args.weight4 > 0:
                parts.append(f"va={total_loss_va/n:.4f}")
            if getattr(args, 'use_va_contrastive', False):
                parts.append(f"cl={total_loss_cl/n:.4f}")
            print(f"  Train {_progress_bar(n, total_step)} {speed:.1f}it/s | L={total_loss/n:.4f} {' '.join(parts)}")

    # Return epoch-level average losses
    avg_losses = {
        'loss': total_loss / total_step,
        'loss_mat': total_loss_mat / total_step,
        'loss_cls_dim': total_loss_cls_dim / total_step,
    }
    if args.label_pattern == 'category' and args.weight4 > 0:
        avg_losses['loss_va'] = total_loss_va / total_step
    if getattr(args, 'use_va_contrastive', False):
        avg_losses['loss_cl'] = total_loss_cl / total_step
    return avg_losses


def validate(args, dataloader, model, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn, metrics_mat, metrics_dim_seq,
             metrics_sen_seq):
    model.eval()
    metrics_mat.reset()
    metrics_dim_seq.reset()
    metrics_sen_seq.reset()
    total_loss = 0
    total_loss_mat = 0
    total_loss_cls_dim = 0
    total_loss_va = 0
    total_step = len(dataloader)
    t_start = time.time()

    with torch.no_grad():
        for step, data in enumerate(dataloader):
            for key in data:
                data[key] = data[key].to(device)

            # Build forward kwargs
            fwd_kwargs = dict(input_ids=data["input_ids"],
                              token_type_ids=data["token_type_ids"],
                              attention_mask=data["attention_mask"])
            va_mode = getattr(args, 'va_mode', 'position')
            if va_mode in ('span_pair', 'opinion_guided') and args.label_pattern == 'category' and args.weight4 > 0:
                fwd_kwargs["quad_spans"] = data["quad_spans"]
                fwd_kwargs["quad_mask"] = data["quad_mask"]
                fwd_kwargs["va_mode"] = va_mode

            pred = model(**fwd_kwargs)

            loss, _, _, loss_va, _ = compute_loss(args, pred, data, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn)
            loss_mat = loss_matrix_fn(y_true=data["matrix_ids"], y_pred=pred["matrix"], mask_rate=args.mask_rate)
            loss_cls_dim = loss_cls_dim_fn(target=data['dimension_ids'], input=pred['dimension'])

            metrics_mat.update(y_true=data["matrix_ids"], y_pred=pred["matrix"])

            if args.label_pattern == 'sentiment':
                metrics_dim_seq.update(y_true=data["dimension_sequences"], y_pred=pred["dimension_sequence"])
            if args.label_pattern == 'raw':
                metrics_dim_seq.update(y_true=data["dimension_sequences"], y_pred=pred["dimension_sequence"])
                metrics_sen_seq.update(y_true=data["sentiment_sequences"], y_pred=pred["sentiment_sequence"])

            total_loss += loss.item()
            total_loss_mat += loss_mat.item()
            total_loss_cls_dim += loss_cls_dim.item()
            if args.label_pattern == 'category' and args.weight4 > 0:
                total_loss_va += loss_va.item()

    elapsed = time.time() - t_start
    parts = [f"mat={total_loss_mat/total_step:.4f}", f"dim={total_loss_cls_dim/total_step:.4f}"]
    if args.label_pattern == 'category' and args.weight4 > 0:
        parts.append(f"va={total_loss_va/total_step:.4f}")
    print(f"  Valid {_progress_bar(total_step, total_step)} {elapsed:.0f}s    | L={total_loss/total_step:.4f} {' '.join(parts)}")

    avg_losses = {
        'loss': total_loss / total_step,
        'loss_mat': total_loss_mat / total_step,
        'loss_cls_dim': total_loss_cls_dim / total_step,
    }
    if args.label_pattern == 'category' and args.weight4 > 0:
        avg_losses['loss_va'] = total_loss_va / total_step
    return metrics_mat.result(), metrics_dim_seq.result(), metrics_sen_seq.result(), avg_losses


def main(args):
    set_seeds(args.seed)
    logger.info(f"parameters: {json.dumps(vars(args), indent=4)}")

    resume_requested = _is_resume_requested(args)
    resume_checkpoint_path = None
    if resume_requested:
        args.model_path = os.path.abspath(os.path.normpath(str(args.model_path)))
        if not os.path.isdir(args.model_path):
            raise FileNotFoundError(f"Resume 目录不存在: {args.model_path}")
        saved_args_path = os.path.join(args.model_path, "args.json")
        if not os.path.exists(saved_args_path):
            raise FileNotFoundError(f"Resume 目录缺少 args.json: {saved_args_path}")
        saved_args = _load_json_file(saved_args_path)
        _assert_resume_args_compatible(args, saved_args)
        args.output_dir = args.model_path
        resume_checkpoint_path = os.path.join(args.output_dir, RESUME_STATE_NAME)
    else:
        args.output_dir = make_output_dir(args)
        dump_args(args.output_dir, args)


    # dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_dataset = AcqpDataset(task_domain=args.task_domain,
                                tokenizer=tokenizer,
                                data_path=args.train_data,
                                max_seq_len=args.max_seq_len,
                                label_pattern=args.label_pattern,
                                nrows=args.nrows)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.per_gpu_train_batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  collate_fn=collate_fn)
    valid_dataset = AcqpDataset(task_domain=args.task_domain,
                                tokenizer=tokenizer,
                                data_path=args.valid_data,
                                max_seq_len=args.max_seq_len,
                                label_pattern=args.label_pattern,
                                nrows=args.nrows)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=args.per_gpu_train_batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  drop_last=False,
                                  collate_fn=collate_fn)
    logger.info(f'train_dataset : {len(train_dataset)}, valid_dataset : {len(valid_dataset)}')

    logger.info(f'num_label_types: {train_dataset.num_label_types}, label_types：{train_dataset.label_types}')
    logger.info(
        f'num_dimension_types: {train_dataset.num_dimension_types}, dimension_types：{list(train_dataset.dimension2id.keys())}')

    # model
    model = QuadrupleModel(num_label_types=train_dataset.num_label_types,
                           num_dimension_types=train_dataset.num_dimension_types,
                           max_seq_len=args.max_seq_len,
                           head_size=args.head_size,
                           mode=args.mode,
                           dimension_hidden_size=256,
                           dimension_sequence_hidden_size=256,
                           sentiment_sequence_hidden_size=256,
                           pretrain_model_path=args.model_name_or_path,
                           with_adversarial_training=True,
                           dropout_rate=args.dropout_rate,
                           use_efficient_global_pointer=args.use_efficient_global_pointer)

    # model = torch.compile(model)
    model.to(device)

    # optimer
    optimizer = set_optimer(args, model)

    #
    loss_matrix_fn = global_pointer_crossentropy
    loss_cls_dim_fn = nn.BCEWithLogitsLoss()
    loss_dim_seq_fn = global_pointer_crossentropy

    # metrics
    metrics_mat = MetricsCalculator(labels=train_dataset.label_types, ignore_index=2)
    metrics_dim_seq = MetricsCalculator(labels=train_dataset.dimension_types)
    metrics_sen_seq = MetricsCalculator(labels=train_dataset.sentiment_types)

    # mix-precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    #
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')

    # Visualizing Your Model
    # input_ids = torch.ones(1, 128, device=device, dtype=torch.int32)
    # token_type_ids = torch.ones(1, 128, device=device, dtype=torch.int32)
    # attention_mask = torch.ones(1, 128, device=device, dtype=torch.int32)
    # writer.add_graph(model, (input_ids, token_type_ids, attention_mask), use_strict_trace=False)
    # writer.flush()

    # lr_scheduler
    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                  mode='max',
                                  factor=0.9,
                                  patience=2,
                                  min_lr=1e-05,
                                  )
    if args.with_adversarial_training:
        fgm = FGM(model, epsilon=1, emb_name='word_embeddings.weight')
    else:
        fgm = None
    best_score, best_epoch, early_stop_count = 0, 0, 0
    output_model_path = os.path.join(args.output_dir, "model.pt")
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    train_history = []
    training_start_time = time.time()
    epoch_times = []
    start_epoch = 0

    if resume_requested:
        if os.path.exists(resume_checkpoint_path):
            resume_state = _load_resume_checkpoint(resume_checkpoint_path, model, optimizer, scheduler, scaler)
        else:
            resume_state = _load_legacy_resume_state(args.output_dir, model)
        start_epoch = int(resume_state.get("completed_epochs", 0))
        best_score = float(resume_state.get("best_score", 0))
        best_epoch = int(resume_state.get("best_epoch", 0))
        early_stop_count = int(resume_state.get("early_stop_count", 0))
        train_history = list(resume_state.get("train_history", []))
        epoch_times = list(resume_state.get("epoch_times", []))
        elapsed_training_seconds = float(resume_state.get("elapsed_training_seconds", 0.0))
        training_start_time = time.time() - elapsed_training_seconds
        logger.info(
            f"Resumed from {resume_checkpoint_path}: completed_epochs={start_epoch}, "
            f"best_score={best_score:.4f}, best_epoch={best_epoch}, early_stop_count={early_stop_count}"
        )
        print(f"  Resume: {resume_state.get('source', resume_checkpoint_path)}")
        print(f"  Resume State: completed_epochs={start_epoch}, best={best_score:.4f} (epoch {best_epoch})")
        _save_resume_checkpoint(
            checkpoint_path=resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            completed_epochs=start_epoch,
            best_score=best_score,
            best_epoch=best_epoch,
            early_stop_count=early_stop_count,
            train_history=train_history,
            epoch_times=epoch_times,
            elapsed_training_seconds=elapsed_training_seconds,
        )

    va_mode = getattr(args, 'va_mode', 'position')
    print(f"\n{'='*70}")
    print(f"  Training: {args.task_domain} | {args.label_pattern} | seed={args.seed}")
    print(f"  Epochs: {args.epoch} | BS: {args.per_gpu_train_batch_size}x{getattr(args,'gradient_accumulation_steps',1)} | Early stop: {args.early_stop}")
    va_cl_str = f" | CL: w={getattr(args, 'weight_va_cl', 0.1)}" if getattr(args, 'use_va_contrastive', False) else ""
    print(f"  Weights: mat={args.weight1} dim={args.weight2} va={args.weight4} | VA: {va_mode}{va_cl_str}")
    print(f"  Grad clip: {getattr(args, 'max_grad_norm', 0)} | mask={args.mask_rate}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}\n")

    if start_epoch >= args.epoch:
        print(f"  Resume checkpoint 已完成全部 {args.epoch} 个 epoch，无需继续训练。")
        writer.close()
        return

    if early_stop_count >= args.early_stop:
        print(f"  Resume checkpoint 已达到 early-stop 条件（wait={early_stop_count}/{args.early_stop}），无需继续训练。")
        writer.close()
        return

    for t in range(start_epoch, args.epoch):
        epoch_start = time.time()

        # --- Train ---
        train_losses = train(args,
              train_dataloader,
              model,
              loss_matrix_fn,
              loss_cls_dim_fn,
              loss_dim_seq_fn,
              optimizer,
              scaler,
              adversial_model=fgm,
              tokenizer=tokenizer)

        # --- Validate ---
        dic_mat_report, dic_dim_seq_report, dic_sen_seq_report, val_losses = validate(args,
                                                                          valid_dataloader,
                                                                          model,
                                                                          loss_matrix_fn,
                                                                          loss_cls_dim_fn,
                                                                          loss_dim_seq_fn,
                                                                          metrics_mat,
                                                                          metrics_dim_seq,
                                                                          metrics_sen_seq)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        try:
            mat_f1_score = dic_mat_report['micro avg']['f1-score']
        except:
            mat_f1_score = 1
        try:
            mat_precision = dic_mat_report['micro avg']['precision']
            mat_recall = dic_mat_report['micro avg']['recall']
        except:
            mat_precision, mat_recall = 0, 0
        try:
            dim_seq_f1_score = dic_dim_seq_report['micro avg']['f1-score']
        except:
            dim_seq_f1_score = 1
        try:
            sen_seq_f1_score = dic_sen_seq_report['micro avg']['f1-score']
        except:
            sen_seq_f1_score = 1

        if args.label_pattern == 'sentiment_dim':
            score = mat_f1_score
        elif args.label_pattern == 'sentiment':
            score = (mat_f1_score + dim_seq_f1_score) / 2
        elif args.label_pattern == 'raw':
            score = (mat_f1_score + dim_seq_f1_score + sen_seq_f1_score) / 3
        elif args.label_pattern == 'category':
            score = mat_f1_score

        scheduler.step(score)

        # --- Epoch Summary ---
        is_best = score >= best_score or score < 0.02
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.epoch - (t + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_min = eta_seconds / 60

        # Build loss string
        loss_parts = [f"mat={val_losses['loss_mat']:.4f}", f"dim={val_losses['loss_cls_dim']:.4f}"]
        if 'loss_va' in val_losses:
            loss_parts.append(f"va={val_losses['loss_va']:.4f}")
        loss_str = ' '.join(loss_parts)

        status = "** BEST **" if is_best else f"wait({early_stop_count+1}/{args.early_stop})"
        lr_current = optimizer.param_groups[0]['lr']

        print(f"\n== Epoch {t+1:3d}/{args.epoch} {'='*50}")
        print(f"   Score: {score:.4f}  (P={mat_precision:.4f} R={mat_recall:.4f} F1={mat_f1_score:.4f})  {status}")
        print(f"   Loss:  {loss_str}  |  LR: {lr_current:.2e}")
        print(f"   Best:  {max(best_score, score) if is_best else best_score:.4f}  |  Time: {epoch_time:.0f}s  |  ETA: {eta_min:.0f}min")
        print(f"{'='*65}")

        history_entry = {
            "epoch": t + 1,
            "matrix_f1": mat_f1_score,
            "matrix_precision": mat_precision,
            "matrix_recall": mat_recall,
            "dimension_seq_f1": dim_seq_f1_score,
            "sentiment_seq_f1": sen_seq_f1_score,
            "score": score,
            "train_loss": train_losses.get('loss', 0),
            "val_loss": val_losses.get('loss', 0),
            "epoch_time": epoch_time,
        }
        if 'loss_va' in val_losses:
            history_entry['val_loss_va'] = val_losses['loss_va']
        if 'loss_va' in train_losses:
            history_entry['train_loss_va'] = train_losses['loss_va']
        if 'loss_cl' in train_losses:
            history_entry['train_loss_cl'] = train_losses['loss_cl']
        train_history.append(history_entry)
        with open(os.path.join(args.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
            json.dump(train_history, f, indent=2, ensure_ascii=False)
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': torch.tensor(4), 'Validation': torch.tensor(4)},
                           t)

        if is_best:
            best_score = score
            best_epoch = t + 1
            early_stop_count = 0
            torch.save(model, best_model_path)
            with open(os.path.join(args.output_dir, "best_score.json"), "w", encoding="utf-8") as f:
                json.dump({"best_score": best_score, "best_epoch": best_epoch}, f, indent=2, ensure_ascii=False)
        else:
            early_stop_count += 1

        if resume_checkpoint_path is None:
            resume_checkpoint_path = os.path.join(args.output_dir, RESUME_STATE_NAME)
        _save_resume_checkpoint(
            checkpoint_path=resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            completed_epochs=t + 1,
            best_score=best_score,
            best_epoch=best_epoch,
            early_stop_count=early_stop_count,
            train_history=train_history,
            epoch_times=epoch_times,
            elapsed_training_seconds=time.time() - training_start_time,
        )

        if early_stop_count >= args.early_stop:
            total_time = (time.time() - training_start_time) / 60
            print(f"\n{'='*70}")
            print(f"  EARLY STOP at epoch {t+1} | Best: {best_score:.4f} | Total: {total_time:.1f}min")
            print(f"{'='*70}")
            torch.save(model, output_model_path)
            writer.close()
            return

    total_time = (time.time() - training_start_time) / 60
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE ({args.epoch} epochs) | Best: {best_score:.4f} | Total: {total_time:.1f}min")
    print(f"{'='*70}")
    torch.save(model, output_model_path)
    writer.close()
    print("Done!")


if __name__ == '__main__':
    args = get_argparse().parse_args()
    main(args)
