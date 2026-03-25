"""
@Time : 2022/12/1717:11
@Auth : zhoujx
@File ：main.py
@DESCRIPTION:

"""
import json
import os
import time

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


def compute_loss(args, pred, data, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn):
    """Compute total loss based on label_pattern. Returns (loss, loss_mat, loss_cls_dim, loss_va)."""
    loss_mat = loss_matrix_fn(y_true=data["matrix_ids"], y_pred=pred["matrix"], mask_rate=args.mask_rate)
    loss_cls_dim = loss_cls_dim_fn(target=data['dimension_ids'], input=pred['dimension'])
    loss_va = torch.tensor(0.0, device=loss_mat.device)
    if args.label_pattern == 'sentiment_dim':
        loss = args.weight1 * loss_mat + args.weight2 * loss_cls_dim
    elif args.label_pattern == 'sentiment':
        loss_dim_seq = loss_dim_seq_fn(y_true=data["dimension_sequences"], y_pred=pred["dimension_sequence"],
                                       mask_rate=args.mask_rate)
        loss = args.weight1 * loss_mat + args.weight2 * loss_cls_dim + args.weight3 * loss_dim_seq
    elif args.label_pattern == 'raw':
        loss_dim_seq = loss_dim_seq_fn(y_true=data["dimension_sequences"], y_pred=pred["dimension_sequence"],
                                       mask_rate=args.mask_rate)
        loss_sen_seq = loss_dim_seq_fn(y_true=data["sentiment_sequences"], y_pred=pred["sentiment_sequence"],
                                       mask_rate=args.mask_rate)
        loss = args.weight1 * loss_mat + args.weight2 * loss_cls_dim + args.weight3 * loss_dim_seq + args.weight4 * loss_sen_seq
    elif args.label_pattern == 'category':
        loss = args.weight1 * loss_mat + args.weight2 * loss_cls_dim
        if args.weight4 > 0:
            va_mode = getattr(args, 'va_mode', 'position')
            if va_mode == 'span_pair' and "span_va" in pred:
                # Span-Pair Conditioned VA loss (masked MSE over gold quadruplets)
                span_va_pred = pred["span_va"]                    # [B, Q, 2]
                span_va_gold = data["quad_va"].to(span_va_pred.device)   # [B, Q, 2]
                mask = data["quad_mask"].to(span_va_pred.device).unsqueeze(-1)  # [B, Q, 1]
                diff = (span_va_pred - span_va_gold) * mask
                n_valid = mask.sum().clamp(min=1.0)
                loss_va = (diff ** 2).sum() / n_valid
            else:
                # Per-position VA loss (original)
                loss_va = compute_va_loss(pred["va"], data["va_targets"], data["va_mask"])
            loss = loss + args.weight4 * loss_va
    return loss, loss_mat, loss_cls_dim, loss_va


def _progress_bar(current, total, width=25):
    """Render a simple ASCII progress bar string."""
    filled = int(width * current / total)
    bar = '#' * filled + '.' * (width - filled)
    return f'[{bar}] {current}/{total}'


def train(args, dataloader, model, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn, optimizer, scaler,
          adversial_model=None, **kwargs):
    model.train()
    total_loss = 0
    total_loss_mat = 0
    total_loss_cls_dim = 0
    total_loss_va = 0
    total_step = len(dataloader)
    accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
    # Print at every 10%
    print_steps = set(max(0, int(total_step * p) - 1) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    t_start = time.time()

    for step, data in enumerate(dataloader):
        for key in data:
            data[key] = data[key].to(device)

        # Build forward kwargs (pass span info when using span_pair VA)
        fwd_kwargs = dict(input_ids=data["input_ids"],
                          token_type_ids=data["token_type_ids"],
                          attention_mask=data["attention_mask"])
        va_mode = getattr(args, 'va_mode', 'position')
        if va_mode == 'span_pair' and args.label_pattern == 'category' and args.weight4 > 0:
            fwd_kwargs["quad_spans"] = data["quad_spans"]
            fwd_kwargs["quad_mask"] = data["quad_mask"]

        # Compute prediction error
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            pred = model(**fwd_kwargs)
            loss, loss_mat, loss_cls_dim, loss_va = compute_loss(args, pred, data, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn)
            loss = loss / accum_steps

        # Backpropagation
        scaler.scale(loss).backward()

        # adversial_training
        if adversial_model is not None:
            adversial_model.attack()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(**fwd_kwargs)
                loss_adv, _, _, _ = compute_loss(args, pred, data, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn)
                loss_adv = loss_adv / accum_steps

            scaler.scale(loss_adv).backward()
            adversial_model.restore()

        total_loss += loss.item() * accum_steps
        total_loss_mat += loss_mat.item()
        total_loss_cls_dim += loss_cls_dim.item()
        if args.label_pattern == 'category' and args.weight4 > 0:
            total_loss_va += loss_va.item()

        # Gradient accumulation: only step every accum_steps
        if (step + 1) % accum_steps == 0 or (step + 1) == total_step:
            if args.use_amp:
                scaler.unscale_(optimizer)
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
            print(f"  Train {_progress_bar(n, total_step)} {speed:.1f}it/s | L={total_loss/n:.4f} {' '.join(parts)}")

    # Return epoch-level average losses
    avg_losses = {
        'loss': total_loss / total_step,
        'loss_mat': total_loss_mat / total_step,
        'loss_cls_dim': total_loss_cls_dim / total_step,
    }
    if args.label_pattern == 'category' and args.weight4 > 0:
        avg_losses['loss_va'] = total_loss_va / total_step
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
            if va_mode == 'span_pair' and args.label_pattern == 'category' and args.weight4 > 0:
                fwd_kwargs["quad_spans"] = data["quad_spans"]
                fwd_kwargs["quad_mask"] = data["quad_mask"]

            pred = model(**fwd_kwargs)

            loss, _, _, loss_va = compute_loss(args, pred, data, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn)
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

    # output
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
    best_score, early_stop_count = 0, 0
    output_model_path = os.path.join(args.output_dir, "model.pt")
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    train_history = []
    training_start_time = time.time()
    epoch_times = []

    va_mode = getattr(args, 'va_mode', 'position')
    print(f"\n{'='*70}")
    print(f"  Training: {args.task_domain} | {args.label_pattern} | seed={args.seed}")
    print(f"  Epochs: {args.epoch} | BS: {args.per_gpu_train_batch_size}x{getattr(args,'gradient_accumulation_steps',1)} | Early stop: {args.early_stop}")
    print(f"  Weights: mat={args.weight1} dim={args.weight2} va={args.weight4} | VA: {va_mode}")
    print(f"  Grad clip: {getattr(args, 'max_grad_norm', 0)} | mask={args.mask_rate}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}\n")

    for t in range(args.epoch):
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
        train_history.append(history_entry)
        with open(os.path.join(args.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
            json.dump(train_history, f, indent=2, ensure_ascii=False)
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': torch.tensor(4), 'Validation': torch.tensor(4)},
                           t)

        if is_best:
            best_score = score
            early_stop_count = 0
            torch.save(model, best_model_path)
            with open(os.path.join(args.output_dir, "best_score.json"), "w", encoding="utf-8") as f:
                json.dump({"best_score": best_score, "best_epoch": t + 1}, f, indent=2, ensure_ascii=False)
        else:
            early_stop_count += 1
            if early_stop_count >= args.early_stop:
                total_time = (time.time() - training_start_time) / 60
                print(f"\n{'='*70}")
                print(f"  EARLY STOP at epoch {t+1} | Best: {best_score:.4f} | Total: {total_time:.1f}min")
                print(f"{'='*70}")
                torch.save(model, output_model_path)
                return

    total_time = (time.time() - training_start_time) / 60
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE ({args.epoch} epochs) | Best: {best_score:.4f} | Total: {total_time:.1f}min")
    print(f"{'='*70}")
    torch.save(model, output_model_path)
    print("Done!")


if __name__ == '__main__':
    args = get_argparse().parse_args()
    main(args)
