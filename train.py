"""
@Time : 2022/12/1717:11
@Auth : zhoujx
@File ：main.py
@DESCRIPTION:

"""
import json
import os

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


def compute_loss(args, pred, data, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn):
    """Compute total loss based on label_pattern."""
    loss_mat = loss_matrix_fn(y_true=data["matrix_ids"], y_pred=pred["matrix"], mask_rate=args.mask_rate)
    loss_cls_dim = loss_cls_dim_fn(target=data['dimension_ids'], input=pred['dimension'])
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
    return loss, loss_mat, loss_cls_dim


def train(args, dataloader, model, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn, optimizer, scaler,
          adversial_model=None, **kwargs):
    model.train()
    total_loss = 0
    total_loss_mat = 0
    total_loss_cls_dim = 0
    total_loss_dim_seq = 0
    total_loss_sen_seq = 0
    total_step = len(dataloader)
    accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
    loop = tqdm(enumerate(dataloader), total=total_step, ncols=150)
    for step, data in loop:
        for key in data:
            data[key] = data[key].to(device)

        # Compute prediction error
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            pred = model(input_ids=data["input_ids"],
                         token_type_ids=data["token_type_ids"],
                         attention_mask=data["attention_mask"])
            loss, loss_mat, loss_cls_dim = compute_loss(args, pred, data, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn)
            loss = loss / accum_steps

        # Backpropagation
        scaler.scale(loss).backward()

        # adversial_training
        if adversial_model is not None:
            adversial_model.attack()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(input_ids=data["input_ids"],
                             token_type_ids=data["token_type_ids"],
                             attention_mask=data["attention_mask"])
                loss_adv, _, _ = compute_loss(args, pred, data, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn)
                loss_adv = loss_adv / accum_steps

            scaler.scale(loss_adv).backward()
            adversial_model.restore()

        total_loss += loss.item() * accum_steps
        total_loss_mat += loss_mat.item()
        total_loss_cls_dim += loss_cls_dim.item()

        # Gradient accumulation: only step every accum_steps
        if (step + 1) % accum_steps == 0 or (step + 1) == total_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loop.set_description(f'Train step [{step}/{total_step}]')
        postfix_dic = {'loss': total_loss / (step + 1),
                       'loss_mat': total_loss_mat / (step + 1),
                       'loss_cls_dim': total_loss_cls_dim / (step + 1)}

        loop.set_postfix(postfix_dic)


def validate(args, dataloader, model, loss_matrix_fn, loss_cls_dim_fn, loss_dim_seq_fn, metrics_mat, metrics_dim_seq,
             metrics_sen_seq):
    model.eval()
    metrics_mat.reset()
    metrics_dim_seq.reset()
    metrics_sen_seq.reset()
    # pbar = ProgressBar(n_total=len(dataloader), desc='testing')
    total_loss = 0
    total_loss_mat = 0
    total_loss_cls_dim = 0
    total_loss_dim_seq = 0
    total_loss_sen_seq = 0
    total_step = len(dataloader)
    loop = tqdm(enumerate(dataloader), total=total_step, ncols=150)
    with torch.no_grad():
        for step, data in loop:
            for key in data:
                data[key] = data[key].to(device)

            # Compute prediction error
            pred = model(input_ids=data["input_ids"],
                         token_type_ids=data["token_type_ids"],
                         attention_mask=data["attention_mask"])
            loss_mat = loss_matrix_fn(y_true=data["matrix_ids"], y_pred=pred["matrix"], mask_rate=args.mask_rate)
            loss_cls_dim = loss_cls_dim_fn(target=data['dimension_ids'], input=pred['dimension'])

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

            metrics_mat.update(y_true=data["matrix_ids"], y_pred=pred["matrix"])

            if args.label_pattern == 'sentiment':
                metrics_dim_seq.update(y_true=data["dimension_sequences"], y_pred=pred["dimension_sequence"])
            if args.label_pattern == 'raw':
                metrics_dim_seq.update(y_true=data["dimension_sequences"], y_pred=pred["dimension_sequence"])
                metrics_sen_seq.update(y_true=data["sentiment_sequences"], y_pred=pred["sentiment_sequence"])

            total_loss += loss.item()
            total_loss_mat += loss_mat.item()
            total_loss_cls_dim += loss_cls_dim.item()

            loop.set_description(f'Valid step [{step}/{total_step}]')
            postfix_dic = {'loss': total_loss / (step + 1),
                           'loss_mat': total_loss_mat / (step + 1),
                           'loss_cls_dim': total_loss_cls_dim / (step + 1)}
            if args.label_pattern == 'sentiment':
                total_loss_dim_seq += loss_dim_seq.item()
                postfix_dic.update({'loss_dim_seq': total_loss_dim_seq / (step + 1)})
            elif args.label_pattern == 'raw':
                total_loss_dim_seq += loss_dim_seq.item()
                postfix_dic.update({'loss_dim_seq': total_loss_dim_seq / (step + 1)})
                total_loss_sen_seq += loss_sen_seq.item()
                postfix_dic.update({'loss_sen_seq': total_loss_sen_seq / (step + 1)})

            loop.set_postfix(postfix_dic)

    return metrics_mat.result(), metrics_dim_seq.result(), metrics_sen_seq.result()


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
    best_score, early_stop = 0, 0
    output_model_path = os.path.join(args.output_dir, "model.pt")
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    train_history = []
    for t in range(args.epoch):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(args,
              train_dataloader,
              model,
              loss_matrix_fn,
              loss_cls_dim_fn,
              loss_dim_seq_fn,
              optimizer,
              scaler,
              adversial_model=fgm,
              tokenizer=tokenizer)
        dic_mat_report, dic_dim_seq_report, dic_sen_seq_report = validate(args,
                                                                          valid_dataloader,
                                                                          model,
                                                                          loss_matrix_fn,
                                                                          loss_cls_dim_fn,
                                                                          loss_dim_seq_fn,
                                                                          metrics_mat,
                                                                          metrics_dim_seq,
                                                                          metrics_sen_seq)
        try:
            mat_f1_score = dic_mat_report['micro avg']['f1-score']
        except:
            mat_f1_score = 1
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

        scheduler.step(score)
        train_history.append({
            "epoch": t + 1,
            "matrix_f1": mat_f1_score,
            "dimension_seq_f1": dim_seq_f1_score,
            "sentiment_seq_f1": sen_seq_f1_score,
            "score": score,
        })
        with open(os.path.join(args.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
            json.dump(train_history, f, indent=2, ensure_ascii=False)
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': torch.tensor(4), 'Validation': torch.tensor(4)},
                           # dic_cls_report,
                           t)

        if score >= best_score or score < 0.02:
            best_score = score
            logger.info(f"Best score : {best_score}")
            early_stop = 0
            torch.save(model, best_model_path)
            with open(os.path.join(args.output_dir, "best_score.json"), "w", encoding="utf-8") as f:
                json.dump({"best_score": best_score, "best_epoch": t + 1}, f, indent=2, ensure_ascii=False)
        else:
            early_stop += 1
            if early_stop >= args.early_stop:
                logger.info(f"Early stop: the best score is {best_score}")
                torch.save(model, output_model_path)
                logger.info(f"Saving model to {output_model_path}")
                return
    torch.save(model, output_model_path)
    logger.info(f"Saving model to {output_model_path}")
    print("Done!")


if __name__ == '__main__':
    args = get_argparse().parse_args()
    main(args)
