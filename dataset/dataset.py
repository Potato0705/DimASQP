# -*- coding: utf-8 -*-
"""
SemEval-2026 Task 3 (DimABSA) Dataset (Fixed Alignment Version)
- 直接基于 Text 编码（add_special_tokens=True, offsets_mapping）
- 标签矩阵与 input_ids 的 token index 严格对齐（不再使用历史 +2/+1 偏移）
- 隐式 Aspect/Opinion 使用 CLS 位置 (index=0) 表示
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger


# =========================
#  VA -> sentiment / dimension (仅作为辅助标签/通道)
# =========================
def _va_to_sentiment(valence: float) -> str:
    if valence >= 6.0:
        return "positive"
    elif valence <= 4.0:
        return "negative"
    else:
        return "neutral"


def _va_to_dimension(arousal: float) -> str:
    # 这里是你原来的简化映射（只是辅助），后续你会改成真正可学习的 Category/VA
    if arousal >= 6.0:
        return "performance"
    elif arousal <= 4.0:
        return "usability"
    else:
        return "quality"


def parse_quadruplet(q: Dict[str, Any]) -> Dict[str, Any]:
    """
    输入：{"Aspect": "...", "Category":"DOMAIN#ATTR", "Opinion":"...", "VA":"7.12#7.12"}
    输出：标准化 dict（同时附加 sentiment/dimension）
    """
    aspect = q.get("Aspect", "NULL")
    category = q.get("Category", "NULL")
    opinion = q.get("Opinion", "NULL")
    va = q.get("VA", "5.00#5.00")

    try:
        v, a = map(float, va.split("#"))
    except Exception:
        v, a = 5.0, 5.0

    sentiment = _va_to_sentiment(v)
    dimension = _va_to_dimension(a)

    return {
        "aspect": aspect,
        "category": category,
        "opinion": opinion,
        "va": f"{v:.2f}#{a:.2f}",
        "sentiment": sentiment,
        "dimension": dimension,
    }


class AcqpDataset(Dataset):
    """
    输出契约（兼容你现有 model.py / train.py）：
    - input_ids / token_type_ids / attention_mask
    - matrix_ids / dimension_ids / dimension_sequences / sentiment_sequences
    """

    def __init__(
        self,
        task_domain: str,
        data_path: str,
        max_seq_len: int,
        tokenizer,
        label_pattern: str = "sentiment_dim",
        **kwargs
    ):
        self.task_domain = task_domain
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.label_pattern = label_pattern

        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                "This dataset implementation requires a FAST tokenizer (offset_mapping). "
                "Please use AutoTokenizer(..., use_fast=True)."
            )

        # 辅助标签空间（你当前版本用于 BA-EO-{dim}-{sent} 通道）
        self.sentiment2id = {"negative": 0, "neutral": 1, "positive": 2}
        self.id2sentiment = {v: k for k, v in self.sentiment2id.items()}

        self.dimension2id = {"usability": 0, "quality": 1, "performance": 2}
        self.id2dimension = {v: k for k, v in self.dimension2id.items()}
        self.num_dimension_types = len(self.dimension2id)

        self.label_types = self.get_label_types()
        self.num_label_types = len(self.label_types)

        self.df_raw = self._read_jsonl(self.data_path)
        self.df = self._build_encoded_df(self.df_raw)

        logger.info(
            f"[Dataset Ready] samples={len(self.df)} "
            f"label_pattern={self.label_pattern} "
            f"num_label_types={self.num_label_types} "
            f"dimensions={list(self.dimension2id.keys())} "
            f"sentiments={list(self.sentiment2id.keys())}"
        )

    def get_label_types(self) -> List[str]:
        label_types = ["BA-BO", "EA-EO"]
        if self.label_pattern == "raw":
            label_types.append("BA-EO")
        elif self.label_pattern == "sentiment":
            for s in self.sentiment2id.keys():
                label_types.append(f"BA-EO-{s}")
        else:
            for dim in self.dimension2id.keys():
                for s in self.sentiment2id.keys():
                    label_types.append(f"BA-EO-{dim}-{s}")
        return label_types

    def _read_jsonl(self, path: str) -> pd.DataFrame:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)

                text = j.get("Text") or j.get("text") or ""
                sid = j.get("ID") or j.get("id") or f"sample_{i}"

                quads = j.get("Quadruplet") or j.get("quadruplet") or []
                parsed = [parse_quadruplet(q) for q in quads]

                rows.append({"id": sid, "text": text, "quads": parsed})

        df = pd.DataFrame(rows)
        df["Text_Id"] = np.arange(len(df))
        logger.info(f"[Load JSONL] {path} -> {len(df)} rows")
        return df

    @staticmethod
    def _find_first_span(text: str, phrase: str) -> Optional[Tuple[int, int]]:
        """
        找 phrase 在 text 的第一个出现区间 [start, end)
        phrase 为 NULL 或空返回 None
        """
        if not phrase or phrase == "NULL":
            return None
        p = phrase.strip("` ").strip()
        if not p:
            return None
        m = re.search(re.escape(p), text, flags=re.IGNORECASE)
        if not m:
            return None
        return m.start(), m.end()

    @staticmethod
    def _char_to_token_index(offsets: List[Tuple[int, int]], char_pos: int) -> Optional[int]:
        """
        给定 char_pos，找到包含它的 token index（跳过 special token 的 (0,0) offset）
        """
        for ti, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if s <= char_pos < e:
                return ti
        return None

    def _span_to_token_span(
        self, offsets: List[Tuple[int, int]], char_span: Optional[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """
        char_span [c_start, c_end) -> token span [t_start, t_end] (inclusive)
        """
        if char_span is None:
            return None
        c_start, c_end = char_span
        if c_end <= c_start:
            return None

        t_start = self._char_to_token_index(offsets, c_start)
        t_end = self._char_to_token_index(offsets, c_end - 1)
        if t_start is None or t_end is None:
            return None
        if t_end < t_start:
            return None
        return t_start, t_end

    def _build_encoded_df(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        把每条样本 encode 并构造 new_answer（token index 形式）
        """
        records = []
        for _, row in df_raw.iterrows():
            text = row["text"]
            sid = row["id"]
            quads = row["quads"]

            enc = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            )

            input_ids = enc["input_ids"]
            token_type_ids = enc.get("token_type_ids", [0] * len(input_ids))
            attention_mask = enc["attention_mask"]
            offsets = enc["offset_mapping"]

            # new_answer: list of dict
            new_answer = []
            for q in quads:
                aspect = q["aspect"]
                opinion = q["opinion"]
                sent = q["sentiment"]
                dim = q["dimension"]

                a_char = self._find_first_span(text, aspect)
                o_char = self._find_first_span(text, opinion)

                # 隐式用 CLS=0
                if a_char is None:
                    a_span = (0, 0)
                else:
                    a_span = self._span_to_token_span(offsets, a_char)
                    if a_span is None:
                        # 超出截断窗口/无法对齐：跳过该标签，避免训练噪声
                        continue

                if o_char is None:
                    o_span = (0, 0)
                else:
                    o_span = self._span_to_token_span(offsets, o_char)
                    if o_span is None:
                        continue

                new_answer.append(
                    {
                        "a_start": int(a_span[0]),
                        "a_end": int(a_span[1]),
                        "o_start": int(o_span[0]),
                        "o_end": int(o_span[1]),
                        "sent_id": int(self.sentiment2id.get(sent, 1)),
                        "dim_key": dim if dim in self.dimension2id else "quality",
                    }
                )

            records.append(
                {
                    "id": sid,
                    "text": text,
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                    "offsets": offsets,
                    "new_answer": new_answer,
                }
            )

        return pd.DataFrame(records)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        answers: List[Dict[str, Any]] = row["new_answer"]

        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        token_type_ids = torch.tensor(row["token_type_ids"], dtype=torch.long)
        attention_mask = torch.tensor(row["attention_mask"], dtype=torch.long)

        matrix_ids = torch.zeros((self.num_label_types, self.max_seq_len, self.max_seq_len), dtype=torch.float32)
        dimension_ids = torch.zeros(self.num_dimension_types, dtype=torch.float32)
        dimension_sequences = torch.zeros((self.num_dimension_types, self.max_seq_len), dtype=torch.float32)
        sentiment_sequences = torch.zeros((3, self.max_seq_len), dtype=torch.float32)

        for a in answers:
            a_start, a_end = a["a_start"], a["a_end"]
            o_start, o_end = a["o_start"], a["o_end"]
            sent_id = a["sent_id"]
            dim_key = a["dim_key"]

            if self.label_pattern == "raw":
                query = "BA-EO"
            elif self.label_pattern == "sentiment":
                query = f"BA-EO-{self.id2sentiment[sent_id]}"
            else:
                query = f"BA-EO-{dim_key}-{self.id2sentiment[sent_id]}"

            # 防越界
            if a_start >= self.max_seq_len or a_end >= self.max_seq_len or o_start >= self.max_seq_len or o_end >= self.max_seq_len:
                continue

            # 关系点（按照你原先 head 语义）
            matrix_ids[self.label_types.index("BA-BO"), a_start, o_start] = 1.0
            matrix_ids[self.label_types.index("EA-EO"), a_end, o_end] = 1.0
            matrix_ids[self.label_types.index(query), a_start, o_end] = 1.0

            # 序列辅助
            if a_start != 0 and a_end != 0 and a_end >= a_start:
                dimension_sequences[self.dimension2id[dim_key], a_start : a_end + 1] = 1.0
                sentiment_sequences[sent_id, a_start : a_end + 1] = 1.0
            if o_start != 0 and o_end != 0 and o_end >= o_start:
                dimension_sequences[self.dimension2id[dim_key], o_start : o_end + 1] = 1.0
                sentiment_sequences[sent_id, o_start : o_end + 1] = 1.0

            dimension_ids[self.dimension2id[dim_key]] = 1.0

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "matrix_ids": matrix_ids,
            "dimension_ids": dimension_ids,
            "dimension_sequences": dimension_sequences,
            "sentiment_sequences": sentiment_sequences,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "token_type_ids": torch.stack([x["token_type_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "matrix_ids": torch.stack([x["matrix_ids"] for x in batch]),
        "dimension_ids": torch.stack([x["dimension_ids"] for x in batch]),
        "dimension_sequences": torch.stack([x["dimension_sequences"] for x in batch]),
        "sentiment_sequences": torch.stack([x["sentiment_sequences"] for x in batch]),
    }
