# -*- coding: utf-8 -*-
import json
import math
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import pandas as pd
from loguru import logger


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


def parse_va(va_str: str) -> Optional[Tuple[float, float]]:
    try:
        a, b = str(va_str).split("#")
        return float(a), float(b)
    except Exception:
        return None


def va_to_sentiment(v: float) -> str:
    # 与你 predict.py 内保持一致
    if v >= 6.0:
        return "positive"
    elif v <= 4.0:
        return "negative"
    else:
        return "neutral"


def va_to_dimension(a: float) -> str:
    if a >= 6.0:
        return "performance"
    elif a <= 4.0:
        return "usability"
    else:
        return "quality"


def _build_label_types(label_pattern: str) -> List[str]:
    """
    raw:           ["BA-BO","EA-EO","BA-EO"]
    sentiment:     ["BA-BO","EA-EO","BA-EO-negative","BA-EO-neutral","BA-EO-positive"]
    sentiment_dim: ["BA-BO","EA-EO"] + 3dims*3sents
    """
    if label_pattern == "raw":
        return ["BA-BO", "EA-EO", "BA-EO"]
    if label_pattern == "sentiment":
        return ["BA-BO", "EA-EO", "BA-EO-negative", "BA-EO-neutral", "BA-EO-positive"]
    if label_pattern == "sentiment_dim":
        dims = ["usability", "quality", "performance"]
        sents = ["negative", "neutral", "positive"]
        heads = ["BA-BO", "EA-EO"]
        for d in dims:
            for s in sents:
                heads.append(f"BA-EO-{d}-{s}")
        return heads
    raise ValueError(f"Unknown label_pattern={label_pattern}")


def _find_sublist(haystack: List[int], needle: List[int], start_from: int = 0) -> Optional[int]:
    """返回 needle 在 haystack 的起始 index；找不到返回 None"""
    if not needle:
        return None
    n = len(needle)
    H = len(haystack)
    for i in range(start_from, H - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return None


def _find_span_by_token_ids(input_ids: List[int], needle_ids: List[int], search_from: int = 1) -> Optional[Tuple[int, int]]:
    """在 input_ids 中找连续子串 needle_ids"""
    st = _find_sublist(input_ids, needle_ids, start_from=search_from)
    if st is None:
        return None
    return st, st + len(needle_ids) - 1


def _locate_phrase(tokenizer, input_ids: List[int], phrase: str) -> Optional[Tuple[int, int]]:
    """
    优先使用 token id 子串匹配，避免 char offset 的坑。
    phrase 为空/NULL -> None（交给上层映射到 CLS）
    """
    phrase = clean_ws(phrase)
    if not phrase or phrase.lower() in {"null", "none"}:
        return None

    # 1) 直接 encode
    ids1 = tokenizer.encode(phrase, add_special_tokens=False)
    sp = _find_span_by_token_ids(input_ids, ids1, search_from=1)
    if sp is not None:
        return sp

    # 2) 前面加空格再 encode（对 sentencepiece/roberta 类常见）
    ids2 = tokenizer.encode(" " + phrase, add_special_tokens=False)
    sp = _find_span_by_token_ids(input_ids, ids2, search_from=1)
    if sp is not None:
        return sp

    # 3) 低风险兜底：小写再试一次（有些 tokenizer 会受大小写影响）
    ids3 = tokenizer.encode(phrase.lower(), add_special_tokens=False)
    sp = _find_span_by_token_ids(input_ids, ids3, search_from=1)
    if sp is not None:
        return sp

    return None


class AcqpDataset(Dataset):
    """
    输出字段：
      input_ids, token_type_ids, attention_mask
      matrix_ids: [C,L,L] float
      dimension_ids: [D] float (multi-hot)
      boundary labels: a_start_ids/a_end_ids/o_start_ids/o_end_ids: [L] float
    """

    def __init__(
        self,
        name: str,
        jsonl_path: str,
        max_len: int,
        tokenizer,
        label_pattern: str = "sentiment_dim",
        keep_cls_as_null: bool = True,
    ):
        super().__init__()
        self.name = name
        self.path = jsonl_path
        self.max_len = int(max_len)
        self.tokenizer = tokenizer
        self.label_pattern = label_pattern
        self.keep_cls_as_null = keep_cls_as_null

        self.label_types = _build_label_types(label_pattern)
        self.label2id = {x: i for i, x in enumerate(self.label_types)}

        # 固定维度/情感空间（与你 predict 的 va_to_* 一致）
        self.dimensions = ["usability", "quality", "performance"]
        self.sentiments = ["negative", "neutral", "positive"]
        self.dimension2id = {d: i for i, d in enumerate(self.dimensions)}
        self.sentiment2id = {s: i for i, s in enumerate(self.sentiments)}

        rows = read_jsonl(jsonl_path)

        # 兼容字段命名
        items = []
        for r in rows:
            _id = r.get("ID") or r.get("id") or r.get("Id")
            _text = r.get("Text") or r.get("text") or ""
            qs = r.get("Quadruplet") or r.get("Quadruplets") or r.get("quadruplets") or []
            items.append({"id": _id, "text": _text, "quads": qs})

        self.df = pd.DataFrame(items)
        logger.info(
            f"[Dataset Ready] samples={len(self.df)} label_pattern={self.label_pattern} "
            f"num_label_types={len(self.label_types)} dimensions={self.dimensions} sentiments={self.sentiments}"
        )

    def __len__(self):
        return len(self.df)

    def _make_empty(self):
        C = len(self.label_types)
        L = self.max_len
        D = len(self.dimension2id)
        matrix = torch.zeros((C, L, L), dtype=torch.float32)
        dim_ids = torch.zeros((D,), dtype=torch.float32)
        a_s = torch.zeros((L,), dtype=torch.float32)
        a_e = torch.zeros((L,), dtype=torch.float32)
        o_s = torch.zeros((L,), dtype=torch.float32)
        o_e = torch.zeros((L,), dtype=torch.float32)
        return matrix, dim_ids, a_s, a_e, o_s, o_e

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = clean_ws(row["text"])
        quads = row["quads"] or []

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors=None,
            return_attention_mask=True,
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        token_type_ids = enc.get("token_type_ids", [0] * len(input_ids))

        # torch 化
        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        attn_t = torch.tensor(attn, dtype=torch.long)
        token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.long)

        matrix, dim_ids, a_s, a_e, o_s, o_e = self._make_empty()

        # CLS 作为 NULL
        null_pos = (0, 0) if self.keep_cls_as_null else None

        # 收集每条 quad -> span
        for q in quads:
            asp = q.get("Aspect", NULL_STR)
            opi = q.get("Opinion", NULL_STR)
            cat = q.get("Category", "LAPTOP#GENERAL")
            va = q.get("VA", "5.00#5.00")

            asp = clean_ws(asp)
            opi = clean_ws(opi)

            # span locate
            asp_span = _locate_phrase(self.tokenizer, input_ids, asp)
            opi_span = _locate_phrase(self.tokenizer, input_ids, opi)

            if asp_span is None:
                if null_pos is None:
                    continue
                a_st, a_ed = null_pos
            else:
                a_st, a_ed = asp_span

            if opi_span is None:
                if null_pos is None:
                    continue
                o_st, o_ed = null_pos
            else:
                o_st, o_ed = opi_span

            # 边界监督（span 正则用）
            a_s[a_st] = 1.0
            a_e[a_ed] = 1.0
            o_s[o_st] = 1.0
            o_e[o_ed] = 1.0

            # 维度多标签
            xy = parse_va(va)
            if xy is not None:
                v, ar = xy
                dim = va_to_dimension(ar)
                if dim in self.dimension2id:
                    dim_ids[self.dimension2id[dim]] = 1.0

            # heads 标注
            # BA-BO: (a_start, o_start)
            matrix[self.label2id["BA-BO"], a_st, o_st] = 1.0
            # EA-EO: (a_end, o_end)
            matrix[self.label2id["EA-EO"], a_ed, o_ed] = 1.0

            # relation: (a_start, o_end)
            if self.label_pattern == "raw":
                matrix[self.label2id["BA-EO"], a_st, o_ed] = 1.0
            elif self.label_pattern == "sentiment":
                # 从 VA 推 sent
                sent = "neutral"
                if xy is not None:
                    sent = va_to_sentiment(xy[0])
                head = f"BA-EO-{sent}"
                if head in self.label2id:
                    matrix[self.label2id[head], a_st, o_ed] = 1.0
            else:
                # sentiment_dim
                dim = "quality"
                sent = "neutral"
                if xy is not None:
                    sent = va_to_sentiment(xy[0])
                    dim = va_to_dimension(xy[1])
                head = f"BA-EO-{dim}-{sent}"
                if head in self.label2id:
                    matrix[self.label2id[head], a_st, o_ed] = 1.0

        return {
            "input_ids": input_ids_t,
            "token_type_ids": token_type_ids_t,
            "attention_mask": attn_t,
            "matrix_ids": matrix,
            "dimension_ids": dim_ids,
            "a_start_ids": a_s,
            "a_end_ids": a_e,
            "o_start_ids": o_s,
            "o_end_ids": o_e,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([x[k] for x in batch], dim=0)
    return out
