import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from loguru import logger


def set_seeds(seed=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def smart_read_csv(data_path, nrows=None):
    # Keep this helper for compatibility; not used in Laptop-ACOS txt flow.
    try:
        df = pd.read_csv(data_path, nrows=nrows)
    except Exception:
        df = pd.read_csv(data_path, nrows=nrows, encoding="gbk")
    df["Text_Id"] = list(range(len(df)))
    return df


def read_data_from_txt(data_path, nrows=None, split_word="####"):
    with open(data_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    if nrows is not None:
        lines = lines[:nrows]

    lines = [line.split(split_word) for line in lines]
    df = pd.DataFrame(lines, columns=["text", "answer"])
    df["answer"] = df["answer"].fillna("[]")
    df = df.loc[~df["answer"].str.startswith("#")]
    df["answer"] = df["answer"].apply(eval)
    df["Text_Id"] = list(range(len(df)))
    logger.info(f"{data_path}, dataset size: {df.shape}")
    return df


def make_output_dir(args):
    current_time = int(time.time())
    localtime = time.localtime(current_time)
    dt = time.strftime("%Y-%m-%d-%H-%M-%S", localtime)
    model_name = args.model_name_or_path.replace("/", "-")
    dir_name = (
        f"{args.task_domain}_{args.label_pattern}_{args.mode}_{model_name}"
        f"_seed{args.seed}_mask{args.mask_rate}_{dt}"
    )
    output_dir = os.path.join(args.output_dir, dir_name)
    if not os.path.exists(output_dir):
        logger.info(f"Making dir: {output_dir}")
        os.makedirs(output_dir)
    return output_dir


def load_train_model(model_path):
    best_model_path = os.path.join(model_path, "best_model.pt")
    fallback_model_path = os.path.join(model_path, "model.pt")
    if os.path.exists(best_model_path):
        return torch.load(best_model_path, weights_only=False)
    return torch.load(fallback_model_path, weights_only=False)


def dump_args(output_dir, args):
    with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


def load_train_args(model_path):
    with open(os.path.join(model_path, "args.json"), "r", encoding="utf-8-sig") as f:
        training_args_dic = json.load(f)
    return training_args_dic

