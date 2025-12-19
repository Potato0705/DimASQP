# -*- coding: utf-8 -*-
"""
SemEval-2026 Task 3 (DimABSA) Dataset (Fixed Alignment + Best-Match Version)
+ Category pair supervision (sparse) with INVALID negatives
- FAST tokenizer offsets_mapping strict alignment
- Implicit Aspect/Opinion use CLS position (index=0)

输出契约（兼容 model.py / train.py / predict.py）：
- input_ids / token_type_ids / attention_mask
- matrix_ids / dimension_ids / dimension_sequences / sentiment_sequences
- cat_pair_indices / cat_pair_labels / cat_pair_mask / cat_pair_cand_mask

注意：
- cat_pair_indices: [K,2] => (a_start, o_end)
- INVALID 必须是最后一类 (invalid_id = num_categories-1)
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger

INVALID_CAT = "INVALID"


def _va_to_sentiment(valence: float) -> str:
    if valence >= 6.0:
        return "positive"
    elif valence <= 4.0:
        return "negative"
    else:
        return "neutral"


def _va_to_dimension(arousal: float) -> str:
    if arousal >= 6.0:
        return "performance"
    elif arousal <= 4.0:
        return "usability"
    else:
        return "quality"


def clean_ws(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def fix_apostrophes(s: str) -> str:
    if s is None:
        return ""
    return str(s).replace("’", "'").replace("`", "'").replace("´", "'").strip()


def norm_key(s: str, apostrophe_norm: bool = True) -> str:
    t = clean_ws(s)
    if apostrophe_norm:
        t = fix_apostrophes(t)
    if not t:
        return "null"
    u = t.strip().lower()
    if u in {"null", "none", "nil", "n/a", "na"}:
        return "null"
    return u


def canon_cat(s: str, default_category: str = INVALID_CAT) -> str:
    """
    Canonicalize category to match categories.txt (UPPER).
    Empty/NULL -> default_category (typically first non-invalid).
    """
    t = clean_ws(s)
    if (not t) or (t.lower() in {"null", "none", "n/a", "na"}):
        t = default_category
    return str(t).strip().upper()


def parse_quadruplet(q: Dict[str, Any]) -> Dict[str, Any]:
    aspect = q.get("Aspect", "NULL")
    category = q.get("Category", "NULL")
    opinion = q.get("Opinion", "NULL")
    va = q.get("VA", "5.00#5.00")

    try:
        v, a = map(float, str(va).split("#"))
    except Exception:
        v, a = 5.0, 5.0

    sentiment = _va_to_sentiment(v)
    dimension = _va_to_dimension(a)

    return {
        "aspect": aspect if aspect else "NULL",
        "category": category if category else "NULL",
        "opinion": opinion if opinion else "NULL",
        "va": f"{v:.2f}#{a:.2f}",
        "v": float(v),
        "a": float(a),
        "sentiment": sentiment,
        "dimension": dimension,
    }


def load_categories(path: str = None) -> List[str]:
    """
    Load categories.txt and enforce:
      - INVALID exists
      - INVALID appears exactly once
      - INVALID is the LAST category (so invalid_id == len(cats)-1)
    """
    if path is None:
        raise ValueError("categories_path must be explicitly provided")

    cands = [path]
    env_p = os.environ.get("DIMABSA_CATEGORIES", "").strip()
    if env_p:
        cands.append(env_p)

    real_path = None
    for p in cands:
        if p and os.path.exists(p):
            real_path = os.path.abspath(p)
            break
    if real_path is None:
        raise FileNotFoundError("categories file not found. Tried:\n  - " + "\n  - ".join(cands))

    cats = []
    with open(real_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            cats.append(t.strip().upper())

    if not cats:
        raise RuntimeError(f"categories file is empty: {real_path}")

    cats_no_invalid = [c for c in cats if c != INVALID_CAT]
    cats = cats_no_invalid + [INVALID_CAT]

    logger.info(f"[Categories] loaded {len(cats)} from {real_path} (INVALID=yes, idx={len(cats)-1})")
    return cats


class AcqpDataset(Dataset):
    def __init__(
        self,
        task_domain: str,
        data_path: str,
        max_seq_len: int,
        tokenizer,
        label_pattern: str = "sentiment_dim",
        categories_path: str = None,
        **kwargs
    ):
        self.task_domain = task_domain
        self.data_path = data_path
        self.max_seq_len = int(max_seq_len)
        self.tokenizer = tokenizer
        self.label_pattern = label_pattern

        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                "This dataset implementation requires a FAST tokenizer (offset_mapping). "
                "Please use AutoTokenizer(..., use_fast=True)."
            )

        self.apostrophe_norm = bool(kwargs.get("apostrophe_norm", True))

        # ===== sentiments / dimensions =====
        self.sentiment2id = {"negative": 0, "neutral": 1, "positive": 2}
        self.id2sentiment = {v: k for k, v in self.sentiment2id.items()}

        self.dimension2id = {"usability": 0, "quality": 1, "performance": 2}
        self.id2dimension = {v: k for k, v in self.dimension2id.items()}
        self.num_dimension_types = len(self.dimension2id)

        # ===== categories =====
        self.categories_path = categories_path
        self.categories = load_categories(self.categories_path)  # includes INVALID last
        self.num_categories = len(self.categories)
        self.category2id = {c: i for i, c in enumerate(self.categories)}
        self.id2category = {i: c for c, i in self.category2id.items()}

        self.invalid_category = INVALID_CAT
        self.invalid_id = int(self.num_categories - 1)  # enforced last

        non_invalid = [c for c in self.categories if c != INVALID_CAT]
        self.default_category = non_invalid[0] if len(non_invalid) else INVALID_CAT

        # ===== negative sampling knobs =====
        self.neg_ratio = float(kwargs.get("neg_ratio", 3.0))
        self.neg_shift = int(kwargs.get("neg_shift", 1))
        self.neg_max_per_sample = int(kwargs.get("neg_max_per_sample", 64))
        self.neg_include_cross = bool(kwargs.get("neg_include_cross", True))
        self.neg_include_random = bool(kwargs.get("neg_include_random", True))

        # ===== label types =====
        self.label_types = self.get_label_types()
        self.num_label_types = len(self.label_types)

        # ===== load + encode =====
        self.df_raw = self._read_jsonl(self.data_path)
        self.df = self._build_encoded_df(self.df_raw)

        logger.info(
            f"[Dataset Ready] samples={len(self.df)} "
            f"label_pattern={self.label_pattern} "
            f"num_label_types={self.num_label_types} "
            f"num_categories(incl.INVALID)={self.num_categories} "
            f"default_category={self.default_category} "
            f"categories_path={self.categories_path}"
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

                quads = j.get("Quadruplet") or j.get("Quadruplets") or j.get("quadruplets") or j.get("quadruplet") or []
                parsed = [parse_quadruplet(q) for q in quads]

                rows.append({"id": sid, "text": text, "quads": parsed})

        df = pd.DataFrame(rows)
        df["Text_Id"] = np.arange(len(df))
        logger.info(f"[Load JSONL] {path} -> {len(df)} rows")
        return df

    @staticmethod
    def _clean_phrase(phrase: str) -> str:
        if phrase is None:
            return ""
        return str(phrase).strip().strip("` ").strip()

    @classmethod
    def _find_all_spans(cls, text: str, phrase: str) -> List[Tuple[int, int]]:
        p = cls._clean_phrase(phrase)
        if not p or p.upper() == "NULL":
            return []
        spans = []
        for m in re.finditer(re.escape(p), text, flags=re.IGNORECASE):
            spans.append((m.start(), m.end()))
        return spans

    @staticmethod
    def _span_mid(span: Tuple[int, int]) -> float:
        return (span[0] + span[1]) / 2.0

    @classmethod
    def _select_best_span_pair(
        cls,
        a_spans: List[Tuple[int, int]],
        o_spans: List[Tuple[int, int]],
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        if not a_spans and not o_spans:
            return None, None
        if not a_spans:
            return None, o_spans[0]
        if not o_spans:
            return a_spans[0], None

        best = None
        best_cost = 1e18
        for a in a_spans:
            ma = cls._span_mid(a)
            for o in o_spans:
                mo = cls._span_mid(o)
                cost = abs(ma - mo)
                overlap = max(0, min(a[1], o[1]) - max(a[0], o[0]))
                if overlap > 0:
                    cost += 5.0
                if cost < best_cost:
                    best_cost = cost
                    best = (a, o)

        return best[0], best[1]

    @staticmethod
    def _char_to_token_index(offsets: List[Tuple[int, int]], char_pos: int) -> Optional[int]:
        for ti, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if s <= char_pos < e:
                return ti
        return None

    def _span_to_token_span(self, offsets: List[Tuple[int, int]], char_span: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
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
        return int(t_start), int(t_end)

    def _build_encoded_df(self, df_raw: pd.DataFrame) -> pd.DataFrame:
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

            new_answer = []
            for q in quads:
                aspect = q["aspect"]
                opinion = q["opinion"]
                sent = q["sentiment"]
                dim = q["dimension"]

                a_spans = self._find_all_spans(text, aspect)
                o_spans = self._find_all_spans(text, opinion)
                a_char, o_char = self._select_best_span_pair(a_spans, o_spans)

                # implicit => CLS=0
                if a_char is None:
                    a_span = (0, 0)
                else:
                    a_span = self._span_to_token_span(offsets, a_char)
                    if a_span is None:
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
                        "category": str(q.get("category", "NULL")),
                        "va": q.get("va", "5.00#5.00"),
                        "v": float(q.get("v", 5.0)),
                        "a": float(q.get("a", 5.0)),
                        "aspect": aspect,
                        "opinion": opinion,
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
        import random

        row = self.df.iloc[index]
        answers: List[Dict[str, Any]] = row["new_answer"]

        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        token_type_ids = torch.tensor(row["token_type_ids"], dtype=torch.long)
        attention_mask = torch.tensor(row["attention_mask"], dtype=torch.long)

        matrix_ids = torch.zeros((self.num_label_types, self.max_seq_len, self.max_seq_len), dtype=torch.float32)
        dimension_ids = torch.zeros(self.num_dimension_types, dtype=torch.float32)
        dimension_sequences = torch.zeros((self.num_dimension_types, self.max_seq_len), dtype=torch.float32)
        sentiment_sequences = torch.zeros((3, self.max_seq_len), dtype=torch.float32)

        # =========================
        # Valid position mask (ALIGN with predict.py):
        # - must be attention=1
        # - exclude PAD/SEP
        # - keep CLS=0 valid
        # =========================
        valid_pos = attention_mask.clone().bool()
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            valid_pos[input_ids == int(self.tokenizer.pad_token_id)] = False
        if getattr(self.tokenizer, "sep_token_id", None) is not None:
            valid_pos[input_ids == int(self.tokenizer.sep_token_id)] = False
        valid_pos[0] = True

        def _is_valid_pos(t: int) -> bool:
            return (0 <= t < self.max_seq_len) and bool(valid_pos[t].item())

        # =========================
        # Category supervision
        # =========================
        invalid_id = int(self.invalid_id)
        num_cats_with_invalid = int(self.num_categories)

        # priors (optional injection from train.py)
        pair2cat_prior = getattr(self, "pair2cat", {}) or {}
        asp2cat_prior = getattr(self, "asp2cat", {}) or {}
        global_cat = getattr(self, "global_cat", None) or self.default_category

        def _build_cand_mask(
            a_key: Optional[str],
            o_key: Optional[str],
            gold_cat: Optional[str] = None,
            max_cands: int = 4
        ) -> torch.Tensor:
            """
            - gold 一定在 candidates 里（不会被截断）
            - INVALID 永远可选
            """
            gold = canon_cat(gold_cat, default_category=self.default_category) if gold_cat else None
            priors: List[str] = []

            if a_key is not None and o_key is not None:
                c = pair2cat_prior.get((a_key, o_key))
                if c:
                    priors.append(canon_cat(c, default_category=self.default_category))
            if a_key is not None:
                c = asp2cat_prior.get(a_key)
                if c:
                    priors.append(canon_cat(c, default_category=self.default_category))

            priors.append(canon_cat(global_cat, default_category=self.default_category))

            uniq = []
            for c in priors:
                if c and c not in uniq:
                    uniq.append(c)

            # reserve one slot for gold if needed
            keep = uniq[:max(0, max_cands - 1)]
            if gold and gold in self.category2id and gold not in keep:
                keep.append(gold)

            m = torch.zeros((num_cats_with_invalid,), dtype=torch.bool)
            for c in keep:
                if c in self.category2id:
                    m[int(self.category2id[c])] = True

            m[invalid_id] = True
            return m

        pair2catid: Dict[Tuple[int, int], int] = {}
        pair2candmask: Dict[Tuple[int, int], torch.Tensor] = {}

        pos_pairs: List[Tuple[int, int]] = []
        a_starts: List[int] = []
        o_ends: List[int] = []

        # ===== positives =====
        for a in answers:
            a_start, a_end = int(a["a_start"]), int(a["a_end"])
            o_start, o_end = int(a["o_start"]), int(a["o_end"])
            sent_id = int(a["sent_id"])
            dim_key = a["dim_key"]

            if self.label_pattern == "raw":
                query = "BA-EO"
            elif self.label_pattern == "sentiment":
                query = f"BA-EO-{self.id2sentiment[sent_id]}"
            else:
                query = f"BA-EO-{dim_key}-{self.id2sentiment[sent_id]}"

            # bounds
            if not (0 <= a_start < self.max_seq_len and 0 <= a_end < self.max_seq_len and
                    0 <= o_start < self.max_seq_len and 0 <= o_end < self.max_seq_len):
                continue

            # horns labels
            matrix_ids[self.label_types.index("BA-BO"), a_start, o_start] = 1.0
            matrix_ids[self.label_types.index("EA-EO"), a_end, o_end] = 1.0
            matrix_ids[self.label_types.index(query), a_start, o_end] = 1.0

            # aux sequences (skip implicit=0 spans)
            if (a_start != 0 or a_end != 0) and a_end >= a_start:
                dimension_sequences[self.dimension2id[dim_key], a_start: a_end + 1] = 1.0
                sentiment_sequences[sent_id, a_start: a_end + 1] = 1.0
            if (o_start != 0 or o_end != 0) and o_end >= o_start:
                dimension_sequences[self.dimension2id[dim_key], o_start: o_end + 1] = 1.0
                sentiment_sequences[sent_id, o_start: o_end + 1] = 1.0

            dimension_ids[self.dimension2id[dim_key]] = 1.0

            # pair label: key = (a_start, o_end)
            cat = canon_cat(a.get("category", "NULL"), default_category=self.default_category)
            key = (a_start, o_end)

            # pair positions must be valid (exclude PAD/SEP)
            if not _is_valid_pos(a_start) or not _is_valid_pos(o_end):
                continue

            cid = int(self.category2id[cat]) if cat in self.category2id else invalid_id
            if key not in pair2catid:
                pair2catid[key] = cid

            a_key = norm_key(a.get("aspect", ""), apostrophe_norm=self.apostrophe_norm)
            o_key = norm_key(a.get("opinion", ""), apostrophe_norm=self.apostrophe_norm)
            pair2candmask[key] = _build_cand_mask(a_key, o_key, gold_cat=cat, max_cands=4)

            pos_pairs.append(key)
            a_starts.append(a_start)
            o_ends.append(o_end)

        # ===== negatives (INVALID) =====
        neg_budget = 0
        if len(pos_pairs) > 0:
            neg_budget = int(min(self.neg_max_per_sample, max(1, round(len(pos_pairs) * float(self.neg_ratio)))))

        neg_pairs_set: set = set()

        # (1) hard shift
        if neg_budget > 0 and self.neg_shift > 0 and len(pos_pairs) > 0:
            s = int(self.neg_shift)
            for (a0, o0) in pos_pairs:
                cands = [(a0 - s, o0), (a0 + s, o0), (a0, o0 - s), (a0, o0 + s)]
                for (aa, oo) in cands:
                    if (aa, oo) in pair2catid:
                        continue
                    if aa == 0 or oo == 0:
                        continue
                    if not _is_valid_pos(aa) or not _is_valid_pos(oo):
                        continue
                    neg_pairs_set.add((int(aa), int(oo)))
                    if len(neg_pairs_set) >= neg_budget:
                        break
                if len(neg_pairs_set) >= neg_budget:
                    break

        # (2) cross pairing
        if neg_budget > 0 and self.neg_include_cross and len(a_starts) >= 2 and len(o_ends) >= 2:
            for aa in a_starts:
                for oo in o_ends:
                    if (aa, oo) in pair2catid:
                        continue
                    if aa == 0 or oo == 0:
                        continue
                    if not _is_valid_pos(aa) or not _is_valid_pos(oo):
                        continue
                    neg_pairs_set.add((int(aa), int(oo)))
                    if len(neg_pairs_set) >= neg_budget:
                        break
                if len(neg_pairs_set) >= neg_budget:
                    break

        # (3) random pairs
        if neg_budget > 0 and self.neg_include_random:
            valid_tokens = [i for i in range(1, self.max_seq_len) if _is_valid_pos(i)]
            if len(valid_tokens) > 0:
                tries = 0
                while len(neg_pairs_set) < neg_budget and tries < neg_budget * 30:
                    tries += 1
                    aa = random.choice(valid_tokens)
                    oo = random.choice(valid_tokens)
                    if (aa, oo) in pair2catid:
                        continue
                    neg_pairs_set.add((int(aa), int(oo)))

        for (aa, oo) in neg_pairs_set:
            k = (int(aa), int(oo))
            if k in pair2catid:
                continue
            pair2catid[k] = invalid_id
            pair2candmask[k] = _build_cand_mask(None, None, gold_cat=None, max_cands=4)

        # ===== pack =====
        if len(pair2catid) == 0:
            cat_pair_indices = torch.zeros((1, 2), dtype=torch.long)
            cat_pair_labels = torch.zeros((1,), dtype=torch.long)
            cat_pair_mask = torch.zeros((1,), dtype=torch.bool)
            cat_pair_cand_mask = torch.zeros((1, num_cats_with_invalid), dtype=torch.bool)
            cat_pair_cand_mask[0, invalid_id] = True
        else:
            items = sorted(pair2catid.items(), key=lambda x: (x[0][0], x[0][1]))
            keys = [k for k, _ in items]
            labels = [v for _, v in items]

            cat_pair_indices = torch.tensor(keys, dtype=torch.long)          # [K,2]
            cat_pair_labels = torch.tensor(labels, dtype=torch.long)         # [K]
            cat_pair_mask = torch.ones((cat_pair_labels.size(0),), dtype=torch.bool)

            cand_list = []
            for k in keys:
                m = pair2candmask.get(k, None)
                if m is None:
                    m = torch.zeros((num_cats_with_invalid,), dtype=torch.bool)
                    m[invalid_id] = True
                cand_list.append(m)
            cat_pair_cand_mask = torch.stack(cand_list, dim=0).bool()        # [K,C]

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "matrix_ids": matrix_ids,
            "dimension_ids": dimension_ids,
            "dimension_sequences": dimension_sequences,
            "sentiment_sequences": sentiment_sequences,

            "cat_pair_indices": cat_pair_indices,          # [K,2]
            "cat_pair_labels": cat_pair_labels,            # [K]
            "cat_pair_mask": cat_pair_mask,                # [K]
            "cat_pair_cand_mask": cat_pair_cand_mask,      # [K,C]
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([x["input_ids"] for x in batch])
    token_type_ids = torch.stack([x["token_type_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    matrix_ids = torch.stack([x["matrix_ids"] for x in batch])
    dimension_ids = torch.stack([x["dimension_ids"] for x in batch])
    dimension_sequences = torch.stack([x["dimension_sequences"] for x in batch])
    sentiment_sequences = torch.stack([x["sentiment_sequences"] for x in batch])

    # -------- pair-level padding --------
    k_list = [int(x["cat_pair_indices"].size(0)) for x in batch]
    k_max = max(k_list) if len(k_list) else 1
    if k_max <= 0:
        k_max = 1

    B = len(batch)
    num_cat_all = batch[0]["cat_pair_cand_mask"].size(-1)

    cat_pair_indices = torch.zeros((B, k_max, 2), dtype=torch.long)
    cat_pair_labels = torch.zeros((B, k_max), dtype=torch.long)
    cat_pair_mask = torch.zeros((B, k_max), dtype=torch.bool)
    cat_pair_cand_mask = torch.zeros((B, k_max, num_cat_all), dtype=torch.bool)

    for i, x in enumerate(batch):
        k = int(x["cat_pair_indices"].size(0))
        if k <= 0:
            continue
        cat_pair_indices[i, :k] = x["cat_pair_indices"]
        cat_pair_labels[i, :k] = x["cat_pair_labels"]
        cat_pair_mask[i, :k] = x["cat_pair_mask"]
        cat_pair_cand_mask[i, :k] = x["cat_pair_cand_mask"]

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "matrix_ids": matrix_ids,
        "dimension_ids": dimension_ids,
        "dimension_sequences": dimension_sequences,
        "sentiment_sequences": sentiment_sequences,

        "cat_pair_indices": cat_pair_indices,        # [B,K,2]
        "cat_pair_labels": cat_pair_labels,          # [B,K]
        "cat_pair_mask": cat_pair_mask,              # [B,K]
        "cat_pair_cand_mask": cat_pair_cand_mask,    # [B,K,C]
    }
