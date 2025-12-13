"""
@Time : 2022/12/1717:25
@Auth : zhoujx
@File ：utils.py
@DESCRIPTION:

"""
import json
import os
import time
import random

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

class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='training')
        >>> step = 2
        >>> pbar(step=step)
    '''

    def __init__(self, n_total, width=30, desc='Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')


def smart_read_csv(data_path, nrows=None):
    try:
        df = pd.read_csv(data_path, nrows=nrows)
    except:
        df = pd.read_csv(data_path, nrows=nrows, encoding="gbk")

    df["内容"] = df["内容"].str.split("【下文不标").str[0]
    df = df.drop_duplicates(subset=["内容"])
    df["答案"] = df["答案"].apply(json.loads)
    df["Text_Id"] = list(range(len(df)))

    logger.info(f"原始数据集：{df.shape}")
    return df


def read_data_from_txt(data_path, nrows=None, split_word='####'):
    with open(data_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    if nrows is not None:
        lines = lines[:nrows]

    lines = [line.split(split_word) for line in lines]
    df = pd.DataFrame(lines, columns=["text", "answer"])
    df['answer'] = df['answer'].fillna('[]')
    df = df.loc[~df['answer'].str.startswith('#')]
    df["answer"] = df["answer"].apply(eval)
    df["Text_Id"] = list(range(len(df)))
    logger.info(f'{data_path}, dataset size ：{df.shape}')

    return df


def make_output_dir(args):
    current_time = int(time.time())
    localtime = time.localtime(current_time)
    dt = time.strftime('%Y-%m-%d-%H-%M-%S', localtime)
    dir_name = f"{args.task_domain}_【{args.label_pattern}】_【{args.mode}】_{args.model_name_or_path.replace('/', '-')}_{args.weight1}_{args.weight2}_{args.weight3}_{args.weight4}_{args.nrows}_【{args.mask_rate}】_{dt}"
    output_dir = os.path.join(args.output_dir, dir_name)
    if not os.path.exists(output_dir):
        logger.info(f'Making dir: {output_dir}')
        os.makedirs(output_dir)
    return output_dir


def load_train_model(model_path):
    return torch.load(os.path.join(model_path, "model.pt"))


def dump_args(output_dir, args):
    with open(os.path.join(output_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False)


def load_train_args(model_path):
    with open(os.path.join(model_path, "args.json"), "r", encoding='utf-8-sig') as f:
        training_args_dic = json.load(f)
    return training_args_dic


if __name__ == '__main__':
    pass
