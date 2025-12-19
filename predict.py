# -*- coding: utf-8 -*-
"""
DimASQP / SemEval Task3 - Predict
(Horns-correct + CategoryHead optional + Head-guided VA + Prior span refine + min_score gate)

关键点：
1) 保留 CLS=0（隐式 NULL）
2) horns 正确解码：
   - relation head: (a_start, o_end)
   - BA-BO: (a_start, o_start)
   - EA-EO: (a_end, o_end)
3) Category：cat_source 控制（prior/head/mix）
   - 自动从 ckpt 推断 category head 输出维度，避免 121/122 mismatch
   - 推理端也对 head logits 做 candidate mask（与训练一致）
4) VA：从 head 解析 dim/sent + train_stats 统计 cat×sent 的 valence + cat×dim 的 arousal
5) span refine：修复边界 bug；并加入“先验命中优先”（asp2cat/pair2cat）
6) 输出门控：--min_score 仅限制第2个及以后 quad（保证至少输出1个 quad）
7) 泛化：不写死 LAPTOP#GENERAL；默认类从 categories_path 中读取（首类，非 INVALID）

本版本修复点（你要求的“确保可运行”）：
- cand 统一为 4-tuple: (score, q, used_pair, used_cat_head_flag)
- prior 的来源 prior_src 放入 q["_prior_src"]，避免 stage2_refine 解包错误
- fallback cand 同样保持 4-tuple，并写入 _prior_src
- stage2_refine/conditional_va_refine 保持兼容（只处理 4-tuple）
- 去掉 pair_hit / asp_hit 重复累加（之前会翻倍）
"""

import os
import json
import re
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer

from dataset.dataset import AcqpDataset, collate_fn, load_categories
from models.model import QuadrupleModel
from utils.utils import set_seeds
from utils.config_loader import load_config, apply_cfg_defaults


ALNUM_RE = re.compile(r"[A-Za-z0-9]")
SPACE_AROUND_APOS_RE = re.compile(r"\s+'\s*|\s*'\s+")
NT_FIX_RE = re.compile(r"\b(n)\s*'\s*(t)\b", flags=re.IGNORECASE)

STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "i", "you", "we", "they",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "and", "or", "but", "if", "then", "than", "so", "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "not", "no", "yes", "do", "does", "did", "done", "have", "has", "had",
}

NULL_STR = "null"
INVALID_CAT = "INVALID"
PRIOR_SRC_KEY = "_prior_src"  # hidden field for internal stats only


# ======================
# I/O
# ======================
def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fw:
        for x in rows:
            fw.write(json.dumps(x, ensure_ascii=False) + "\n")


# ======================
# Text normalize (SINGLE SOURCE OF TRUTH)
# ======================
def clean_ws(s: str) -> str:
    return " ".join(str(s).strip().split()) if s is not None else ""


def fix_apostrophes(s: str) -> str:
    t = clean_ws(s)
    if not t:
        return t
    t = t.replace(" ' ", "'")
    t = t.replace(" '", "'").replace("' ", "'")
    t = NT_FIX_RE.sub(r"\1'\2", t)
    t = SPACE_AROUND_APOS_RE.sub("'", t)
    return clean_ws(t)


def norm_key(s: str, apostrophe_norm: bool = True) -> str:
    t = fix_apostrophes(s) if apostrophe_norm else clean_ws(s)
    t = t.lower()
    if not t:
        return "null"
    if t in {"null", "none"}:
        return "null"
    return t


def canon_cat_output(s: str, cat_case: str = "upper", default_category: str = None) -> str:
    """
    仅用于最终输出格式。内部所有 category 逻辑建议用 UPPER 处理。
    """
    if default_category is None or not str(default_category).strip():
        default_category = INVALID_CAT
    t = clean_ws(s)
    if (not t) or (t.lower() in {"null", "none", "n/a", "na"}):
        t = default_category
    return t.lower() if cat_case == "lower" else t.upper()


def canon_cat_internal(s: str, default_category: str) -> str:
    """
    内部一律使用 UPPER，保证与 ds.category2id / categories.txt 对齐。
    """
    if default_category is None or not str(default_category).strip():
        default_category = INVALID_CAT
    t = clean_ws(s)
    if (not t) or (t.lower() in {"null", "none", "n/a", "na"}):
        t = default_category
    return t.upper()


def is_bad_span(text: str, apostrophe_norm: bool = True) -> bool:
    if text is None:
        return True
    t = fix_apostrophes(text) if apostrophe_norm else clean_ws(text)
    if not t:
        return True
    if t.lower() == NULL_STR:
        return False
    if len(t) < 2:
        return True
    if not ALNUM_RE.search(t):
        return True
    if len(t.split()) == 1 and t.lower() in STOPWORDS:
        return True
    return False


def safe_decode(tokenizer, input_ids_1d, i, j):
    if i == 0 and j == 0:
        return NULL_STR
    if i == 0 and j > 0:
        return NULL_STR
    ids = input_ids_1d[i:j + 1].tolist()
    s = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return clean_ws(s)


# ======================
# VA utilities
# ======================
def parse_va(va_str: str):
    try:
        a, b = str(va_str).split("#")
        return float(a), float(b)
    except Exception:
        return None


def fmt_va(x: float, y: float) -> str:
    x = float(x)
    y = float(y)
    x = max(1.0, min(9.0, x))
    y = max(1.0, min(9.0, y))
    return f"{x:.2f}#{y:.2f}"


def va_to_sentiment(v: float) -> str:
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


def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    m = n // 2
    if n % 2 == 1:
        return xs[m]
    return 0.5 * (xs[m - 1] + xs[m])


# ======================
# Stage-2 refine
# ======================
def conditional_va_refine(cand, alpha=0.3):
    """
    轻量 VA 平滑，防止极端 VA
    cand: [(score, q, used_pair, used_head), ...]
    """
    out = []
    for score, q, up, uh in cand:
        try:
            v, a = map(float, q["VA"].split("#"))
            v = alpha * v + (1 - alpha) * 5.0
            a = alpha * a + (1 - alpha) * 5.0
            q = dict(q)
            q["VA"] = f"{v:.2f}#{a:.2f}"
        except Exception:
            pass
        out.append((score, q, up, uh))
    return out


def stage2_refine(cand, tokenizer, args):
    """
    二阶段 refine（只在 --stage2_refine 开启时生效）
    cand: [(score, q, used_pair, used_head), ...]
    """
    cand = conditional_va_refine(cand)

    # Aspect 级强去重：保留每个 aspect 的最高分
    best_by_asp = {}
    for item in cand:
        akey = norm_key(item[1]["Aspect"], apostrophe_norm=args.apostrophe_norm)
        if akey not in best_by_asp or best_by_asp[akey][0] < item[0]:
            best_by_asp[akey] = item
    cand = list(best_by_asp.values())

    # Opinion overlap 去重
    refined = []
    for score, q, up, uh in sorted(cand, key=lambda x: -x[0]):
        o = q["Opinion"]
        if o == NULL_STR:
            refined.append((score, q, up, uh))
            continue

        tok_o = tokenizer.encode(o, add_special_tokens=False)
        keep = True
        for _, q2, *_ in refined:
            o2 = q2["Opinion"]
            if o2 == NULL_STR:
                continue
            tok_o2 = tokenizer.encode(o2, add_special_tokens=False)
            if not tok_o or not tok_o2:
                continue
            overlap = len(set(tok_o) & set(tok_o2)) / max(1, min(len(tok_o), len(tok_o2)))
            if overlap > 0.5:
                keep = False
                break
        if keep:
            refined.append((score, q, up, uh))
    cand = refined

    # NULL opinion 约束（最多 1 个）
    nulls = [x for x in cand if x[1]["Opinion"] == NULL_STR]
    non_nulls = [x for x in cand if x[1]["Opinion"] != NULL_STR]
    if len(nulls) > 1:
        nulls = sorted(nulls, key=lambda x: -x[0])[:1]
    cand = sorted(non_nulls + nulls, key=lambda x: -x[0])

    return cand


# ======================
# Priors
# ======================
def build_priors(
    train_stats_path,
    va_stat="median",
    apostrophe_norm=True,
    default_category: str = None,
):
    """
    内部 category 一律用 UPPER，保证与 categories.txt / ds.category2id 一致。
    """
    if default_category is None or not str(default_category).strip():
        default_category = INVALID_CAT

    rows = read_jsonl(train_stats_path)

    cat_cnt = Counter()
    asp_cat_cnt = Counter()
    pair_cat_cnt = Counter()

    asp_freq = Counter()
    op_freq = Counter()
    pair_freq = Counter()

    cat_v = defaultdict(list)
    cat_a = defaultdict(list)
    cat_va_mode = defaultdict(Counter)

    cat_sent_v = defaultdict(list)  # (cat,sent) -> [v]
    cat_dim_a = defaultdict(list)   # (cat,dim) -> [a]

    for r in rows:
        qs = r.get("Quadruplet") or r.get("Quadruplets") or r.get("quadruplets") or []
        for q in qs:
            a_raw = q.get("Aspect", "null")
            o_raw = q.get("Opinion", "null")
            c_raw = q.get("Category", "")
            va = q.get("VA", "5.00#5.00")

            a = norm_key(a_raw, apostrophe_norm=apostrophe_norm)
            o = norm_key(o_raw, apostrophe_norm=apostrophe_norm)
            c = canon_cat_internal(c_raw, default_category=default_category)

            asp_freq[a] += 1
            op_freq[o] += 1
            pair_freq[(a, o)] += 1

            cat_cnt[c] += 1
            asp_cat_cnt[(a, c)] += 1
            pair_cat_cnt[(a, o, c)] += 1

            cat_va_mode[c][str(va)] += 1
            xy = parse_va(va)
            if xy is not None:
                v, ar = xy
                cat_v[c].append(v)
                cat_a[c].append(ar)
                sent = va_to_sentiment(v)
                dim = va_to_dimension(ar)
                cat_sent_v[(c, sent)].append(v)
                cat_dim_a[(c, dim)].append(ar)

    global_cat = cat_cnt.most_common(1)[0][0] if len(cat_cnt) else canon_cat_internal(default_category, default_category)

    # pair->cat mode
    pair2cat = {}
    best = {}
    for (a, o, c), cnt in pair_cat_cnt.items():
        key = (a, o)
        if key not in best or cnt > best[key][1]:
            best[key] = (c, cnt)
    for k, (c, _) in best.items():
        pair2cat[k] = c

    # asp->cat mode
    asp2cat = {}
    best2 = {}
    for (a, c), cnt in asp_cat_cnt.items():
        if a not in best2 or cnt > best2[a][1]:
            best2[a] = (c, cnt)
    for a, (c, _) in best2.items():
        asp2cat[a] = c

    asp2cats_top = defaultdict(list)  # a -> [(cat, cnt), ...]
    for (a, c), cnt in asp_cat_cnt.items():
        asp2cats_top[a].append((c, cnt))
    for a in list(asp2cats_top.keys()):
        asp2cats_top[a].sort(key=lambda x: x[1], reverse=True)

    pair2cats_top = defaultdict(list)  # (a,o) -> [(cat, cnt), ...]
    for (a, o, c), cnt in pair_cat_cnt.items():
        pair2cats_top[(a, o)].append((c, cnt))
    for k in list(pair2cats_top.keys()):
        pair2cats_top[k].sort(key=lambda x: x[1], reverse=True)

    def _stat(xs, how: str):
        if not xs:
            return None
        if how == "mean":
            return float(sum(xs) / max(1, len(xs)))
        return float(median(xs))

    how = str(va_stat).lower().strip()

    cat2va = {}
    cat2v = {}
    cat2a = {}

    for c in cat_cnt.keys():
        if how == "mode":
            va0 = cat_va_mode[c].most_common(1)[0][0] if len(cat_va_mode[c]) else fmt_va(5.0, 5.0)
            xy = parse_va(va0)
            if xy is not None:
                cat2v[c], cat2a[c] = xy
            else:
                cat2v[c], cat2a[c] = 5.0, 5.0
            cat2va[c] = fmt_va(cat2v[c], cat2a[c])
        else:
            mv = _stat(cat_v.get(c, []), how=how)
            ma = _stat(cat_a.get(c, []), how=how)
            if mv is None or ma is None:
                va0 = cat_va_mode[c].most_common(1)[0][0] if len(cat_va_mode[c]) else fmt_va(5.0, 5.0)
                xy = parse_va(va0)
                if xy is not None:
                    cat2v[c], cat2a[c] = xy
                else:
                    cat2v[c], cat2a[c] = 5.0, 5.0
                cat2va[c] = fmt_va(cat2v[c], cat2a[c])
            else:
                cat2v[c], cat2a[c] = float(mv), float(ma)
                cat2va[c] = fmt_va(mv, ma)

    cat_sent2v = {}
    for (c, s), vs in cat_sent_v.items():
        mv = _stat(vs, how=("median" if how == "mode" else how))
        if mv is not None:
            cat_sent2v[(c, s)] = float(mv)

    cat_dim2a = {}
    for (c, d), ars in cat_dim_a.items():
        ma = _stat(ars, how=("median" if how == "mode" else how))
        if ma is not None:
            cat_dim2a[(c, d)] = float(ma)

    # ensure fallbacks exist
    dc = canon_cat_internal(default_category, default_category)
    if dc not in cat2v:
        cat2v[dc], cat2a[dc] = 5.0, 5.0
        cat2va[dc] = fmt_va(5.0, 5.0)
    if global_cat not in cat2v:
        cat2v[global_cat], cat2a[global_cat] = 5.0, 5.0
        cat2va[global_cat] = fmt_va(5.0, 5.0)

    logger.info(
        f"[Priors] pair2cat={len(pair2cat)} asp2cat={len(asp2cat)} "
        f"cat2va={len(cat2va)} cat_sent2v={len(cat_sent2v)} cat_dim2a={len(cat_dim2a)} "
        f"global_cat={global_cat} va_stat={va_stat} default_category={default_category}"
    )
    return (pair2cat, asp2cat, cat2va, cat_sent2v, cat_dim2a, cat2v, cat2a,
            global_cat, asp_freq, op_freq, pair_freq, asp2cats_top, pair2cats_top)


# ======================
# Heads parsing
# ======================
def parse_head_name(head_name: str):
    if not head_name.startswith("BA-EO"):
        return None, None
    parts = head_name.split("-")
    if len(parts) == 4:
        dim = parts[2]
        sent = parts[3]
        return dim, sent
    if len(parts) == 3:
        return None, parts[2]
    return None, None


# ======================
# Selection helpers
# ======================
def topk_mask(scores_1d, valid_pos, k):
    if k is None or k <= 0:
        return valid_pos.clone()
    sc = scores_1d.clone()
    sc[~valid_pos] = -1e9
    n = int(valid_pos.sum().item())
    if n <= k:
        return valid_pos.clone()
    _, idx = torch.topk(sc, k=k, largest=True)
    m = torch.zeros_like(valid_pos)
    m[idx] = True
    return m & valid_pos


def extract_top_pairs(scores_LL, pair_mask, thr, topk, max_pair_dist):
    L = scores_LL.shape[0]
    device = scores_LL.device
    mask = pair_mask.clone()

    if max_pair_dist and max_pair_dist > 0:
        idx = torch.arange(L, device=device)
        dist = (idx[None, :] - idx[:, None]).abs()
        allow = (dist <= max_pair_dist) | (idx[:, None] == 0) | (idx[None, :] == 0)
        mask = mask & allow

    flat = scores_LL.masked_fill(~mask, -1e9).view(-1)
    cand = (flat >= thr)
    if not cand.any():
        return []

    idxs = torch.nonzero(cand, as_tuple=False).squeeze(-1)
    vals = flat[idxs]
    if topk is not None and topk > 0 and idxs.numel() > topk:
        topv, topi = torch.topk(vals, k=topk, largest=True)
        idxs = idxs[topi]
        vals = topv

    order = torch.argsort(vals, descending=True)
    idxs = idxs[order]
    vals = vals[order]

    out = []
    for p, v in zip(idxs.tolist(), vals.tolist()):
        i = int(p // L)
        j = int(p % L)
        out.append((i, j, float(v)))
    return out


def pick_o_start(ba_bo, valid_pos, a_start, o_end, max_span_len):
    if o_end == 0:
        return 0, float(ba_bo[a_start, 0].item())

    mask = valid_pos.clone()
    mask[(o_end + 1):] = False
    mask[0] = False
    if max_span_len and max_span_len > 0:
        lo = max(1, o_end - max_span_len + 1)
        mask[:lo] = False

    row = ba_bo[a_start].masked_fill(~mask, -1e9)
    k = int(torch.argmax(row).item())
    sc = float(row[k].item())
    if sc < -1e8:
        return 0, -1e9
    return k, sc


def pick_a_end(ea_eo, valid_pos, a_start, o_end, max_span_len):
    if a_start == 0:
        return 0, float(ea_eo[0, o_end].item())

    mask = valid_pos.clone()
    mask[:a_start] = False
    if max_span_len and max_span_len > 0:
        hi = min(ea_eo.shape[0] - 1, a_start + max_span_len - 1)
        if hi < ea_eo.shape[0] - 1:
            mask[(hi + 1):] = False

    col = ea_eo[:, o_end].masked_fill(~mask, -1e9)
    t = int(torch.argmax(col).item())
    sc = float(col[t].item())
    if sc < -1e8:
        return a_start, -1e9
    return t, sc


# ======================
# Prior span refine
# ======================
def _span_prior_score(txt, freq_dict, apostrophe_norm, asp2cat=None, pair2cat=None, other_key=None,
                      w_freq=1.0, w_asp_hit=0.9, w_pair_hit=1.2):
    k = norm_key(txt, apostrophe_norm)
    if k == "null":
        return -1e9
    freq = float(freq_dict.get(k, 0))
    sc = w_freq * math.log(1.0 + freq)

    if asp2cat is not None and k in asp2cat:
        sc += w_asp_hit

    if pair2cat is not None and other_key is not None:
        if (k, other_key) in pair2cat:
            sc += w_pair_hit
    return sc


def prior_refine_span(tokenizer, input_ids, valid_pos, i, j,
                      freq_dict,
                      max_seq_len,
                      max_span_len,
                      max_expand=2,
                      apostrophe_norm=False,
                      asp2cat=None,
                      pair2cat=None,
                      other_key=None,
                      w_freq=1.0, w_asp_hit=0.9, w_pair_hit=1.2,
                      min_gain=0.15):
    if i == 0 and j == 0:
        return i, j, NULL_STR

    base_txt = safe_decode(tokenizer, input_ids, i, j)
    if is_bad_span(base_txt, apostrophe_norm):
        return i, j, base_txt

    best_i, best_j, best_txt = i, j, base_txt
    best_sc = _span_prior_score(best_txt, freq_dict, apostrophe_norm, asp2cat, pair2cat, other_key,
                                w_freq=w_freq, w_asp_hit=w_asp_hit, w_pair_hit=w_pair_hit)

    for di in range(-max_expand, max_expand + 1):
        ni = i + di
        if ni <= 0 or ni >= max_seq_len:
            continue
        if not bool(valid_pos[ni].item()):
            continue
        for dj in range(-max_expand, max_expand + 1):
            nj = j + dj
            if nj <= 0 or nj >= max_seq_len:
                continue
            if nj < ni:
                continue
            if (nj - ni + 1) > max_span_len:
                continue
            if not bool(valid_pos[nj].item()):
                continue

            txt = safe_decode(tokenizer, input_ids, ni, nj)
            if is_bad_span(txt, apostrophe_norm):
                continue

            sc = _span_prior_score(txt, freq_dict, apostrophe_norm, asp2cat, pair2cat, other_key,
                                   w_freq=w_freq, w_asp_hit=w_asp_hit, w_pair_hit=w_pair_hit)
            if sc > best_sc + min_gain:
                best_sc, best_i, best_j, best_txt = sc, ni, nj, txt

    return best_i, best_j, best_txt


# ======================
# CKPT category dim inference
# ======================
def _infer_ckpt_num_categories(state_dict: dict) -> int:
    for k in [
        "category_mlp.3.weight", "module.category_mlp.3.weight",
        "category_mlp.2.weight", "module.category_mlp.2.weight",
    ]:
        if k in state_dict and hasattr(state_dict[k], "shape"):
            return int(state_dict[k].shape[0])
    return 0


def load_model_auto_catdim(ckpt_path, model_name, num_label_types, num_dimension_types, max_len, device, ds_num_categories):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    ckpt_num_categories = _infer_ckpt_num_categories(state)
    if ckpt_num_categories <= 0:
        ckpt_num_categories = int(ds_num_categories)

    model = QuadrupleModel(
        num_label_types=num_label_types,
        num_dimension_types=num_dimension_types,
        max_seq_len=max_len,
        pretrain_model_path=model_name,
        num_categories=int(ckpt_num_categories),
    ).to(device)

    model_state = model.state_dict()
    filtered = {}
    skipped = []

    for k, v in state.items():
        if k not in model_state:
            skipped.append((k, "missing_key_in_model"))
            continue
        if tuple(v.shape) != tuple(model_state[k].shape):
            skipped.append((k, f"shape_mismatch ckpt={tuple(v.shape)} model={tuple(model_state[k].shape)}"))
            continue
        filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if skipped:
        logger.warning("[load_model] skipped {} params due to key/shape mismatch. Examples: {}", len(skipped), skipped[:5])
    if missing:
        logger.warning("[load_model] missing keys (not loaded): {}", len(missing))
    if unexpected:
        logger.warning("[load_model] unexpected keys (ignored): {}", len(unexpected))

    logger.info(f"[load_model] ckpt={ckpt_path} ds_num_categories={ds_num_categories} ckpt_num_categories={ckpt_num_categories}")

    id2category_pred = {}
    for i in range(int(ckpt_num_categories)):
        id2category_pred[i] = None if i < int(ds_num_categories) else INVALID_CAT

    model.eval()
    return model, id2category_pred


@torch.no_grad()
def classify_pairs_batch(
    model,
    sequence_output_1: torch.Tensor,  # [1,L,H]
    pair_indices_1: torch.Tensor,     # [N,3] with b=0
    cat_head_batch: int
):
    N = int(pair_indices_1.size(0))
    logits_all = []
    for st in range(0, N, int(cat_head_batch)):
        ed = min(N, st + int(cat_head_batch))
        sub = pair_indices_1[st:ed]
        logits = model.classify_pairs(sequence_output_1, sub)
        logits_all.append(logits)
    return torch.cat(logits_all, dim=0) if len(logits_all) else None


def build_cand_mask_for_pair(
    a_key: str,
    o_key: str,
    pair2cat: dict,
    asp2cat: dict,
    global_cat: str,
    ds_category2id: dict,
    num_categories: int,
    invalid_id_ds: int,
    max_cands: int = 12,
):
    """
    推理端 cand mask：与 dataset/train 的逻辑一致（prior->cands + global + INVALID）
    内部 lookup 一律用 UPPER。
    """
    cands = []
    if a_key != "null" and o_key != "null":
        c = pair2cat.get((a_key, o_key))
        if c:
            cands.append(str(c).upper())
    if a_key != "null":
        c = asp2cat.get(a_key)
        if c:
            cands.append(str(c).upper())

    cands.append(str(global_cat).upper())

    uniq = []
    for c in cands:
        if c not in uniq:
            uniq.append(c)
    uniq = uniq[:max_cands]

    m = torch.zeros((int(num_categories),), dtype=torch.bool)
    for c in uniq:
        if c in ds_category2id:
            idx = int(ds_category2id[c])
            if idx < int(num_categories):
                m[idx] = True

    # ensure INVALID always candidate
    if 0 <= int(invalid_id_ds) < int(num_categories):
        m[int(invalid_id_ds)] = True
    if int(num_categories) > 0:
        m[int(num_categories) - 1] = True
    return m


def main():
    import argparse
    import os
    from pathlib import Path
    from collections import Counter

    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from loguru import logger

    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str, default=None, help="Path to configs/<lang>-<domain>/data.yaml")
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--train_stats", type=str, default=None)

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--label_pattern", type=str, default="sentiment_dim", choices=["raw", "sentiment", "sentiment_dim"])

    ap.add_argument("--thr_aux", type=float, default=0.05)
    ap.add_argument("--topk_aux", type=int, default=80)
    ap.add_argument("--max_span_len", type=int, default=12)

    ap.add_argument("--thr_rel", type=float, default=0.12)
    ap.add_argument("--topk_rel", type=int, default=800)
    ap.add_argument("--max_pair_dist", type=int, default=120)

    ap.add_argument("--max_quads", type=int, default=2)

    # 注意：只门控第2个及以后 quad
    ap.add_argument("--min_score", type=float, default=-1.0)

    # Stage-2 refine (A/B switch)
    ap.add_argument("--stage2_refine", action="store_true", help="Enable Stage-2 refinement (A/B switch)")

    ap.add_argument("--null_thr_o", type=float, default=0.10)

    ap.add_argument("--va_stat", type=str, default="median", choices=["mode", "median", "mean"])
    ap.add_argument("--cat_case", type=str, default="upper", choices=["upper", "lower"])

    # 默认 apostrophe_norm=True，同时提供关闭开关
    ap.add_argument("--no_apostrophe_norm", action="store_true", help="Disable apostrophe normalization (default ON).")

    ap.add_argument("--no_pair2cat_when_op_null", action="store_true")
    ap.add_argument("--dedup_by_aspect", action="store_true")

    ap.add_argument("--refine_span", action="store_true")
    ap.add_argument("--span_expand", type=int, default=2)
    ap.add_argument("--span_min_gain", type=float, default=0.15)

    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--output", required=True)

    ap.add_argument("--categories_path", type=str, default=None)

    ap.add_argument("--cat_source", type=str, default="prior", choices=["prior", "head", "mix"])
    ap.add_argument("--cat_head_min_conf", type=float, default=0.55)
    ap.add_argument("--global_cat_min_conf", type=float, default=0.98)
    ap.add_argument("--cat_head_batch", type=int, default=2048)

    # anti-collapse 相关：为了“冲分”，默认不放宽触发条件；如果你未来要做E1再改这里
    ap.add_argument("--ac_head2_min_conf", type=float, default=0.15, help="min prob for head-top2 in anti-collapse")
    ap.add_argument("--ac_enable", action="store_true", help="Enable anti-collapse (safe version).")

    args = ap.parse_args()

    # deterministic
    set_seeds(int(args.seed))

    # derive apostrophe_norm
    args.apostrophe_norm = (not bool(args.no_apostrophe_norm))

    # repo_root：保证从任意目录启动都稳定
    repo_root = Path(__file__).resolve().parent

    # config fill
    if args.config:
        cfg = load_config(args.config, repo_root=str(repo_root))
        args = apply_cfg_defaults(args, cfg, {
            "dev": "input",
            "train_all": "train_stats",
            "categories": "categories_path",
        })

    if not args.input:
        raise ValueError("Missing --input (or set 'dev:' in --config).")
    if not args.train_stats:
        raise ValueError("Missing --train_stats (or set 'train_all:' in --config).")
    if not args.categories_path:
        raise ValueError("Missing --categories_path (or set 'categories:' in --config).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Device] {device}")
    if args.config:
        logger.info(f"[Config] {os.path.abspath(args.config)}")
    logger.info(
        f"[Paths] input={os.path.abspath(args.input)} train_stats={os.path.abspath(args.train_stats)} "
        f"categories={os.path.abspath(args.categories_path)}"
    )
    logger.info(f"[AposNorm] apostrophe_norm={args.apostrophe_norm}")

    # categories + default category
    cats = load_categories(args.categories_path)  # includes INVALID last
    non_invalid = [c for c in cats if c != INVALID_CAT]
    default_category = non_invalid[0] if len(non_invalid) else INVALID_CAT
    logger.info(f"[Categories] default_category={default_category} (from {args.categories_path})")

    (pair2cat, asp2cat, cat2va, cat_sent2v, cat_dim2a, cat2v, cat2a,
     global_cat, asp_freq, op_freq, pair_freq, asp2cats_top, pair2cats_top) = build_priors(
        args.train_stats,
        va_stat=args.va_stat,
        apostrophe_norm=args.apostrophe_norm,
        default_category=default_category,
    )

    # ✅ 统一：内部 global 都用 UPPER 字符串
    global_cat_u = str(global_cat).upper()

    # hidden keys (must never leak to output)
    PRIOR_SRC_KEY = "_prior_src"
    AC_USED_KEY = "_ac_used"  # None / "head2" / "asp2"

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    ds = AcqpDataset(
        "PredictSet",
        args.input,
        args.max_len,
        tok,
        label_pattern=args.label_pattern,
        categories_path=args.categories_path,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    if "BA-BO" not in ds.label_types or "EA-EO" not in ds.label_types:
        raise RuntimeError(f"label_types missing BA-BO/EA-EO: {ds.label_types}")

    H_BA_BO = ds.label_types.index("BA-BO")
    H_EA_EO = ds.label_types.index("EA-EO")
    rel_heads = [i for i, nm in enumerate(ds.label_types) if nm not in ["BA-BO", "EA-EO"]]

    model, id2category_pred = load_model_auto_catdim(
        args.ckpt,
        args.model_name,
        len(ds.label_types),
        len(ds.dimension2id),
        args.max_len,
        device,
        ds_num_categories=ds.num_categories
    )

    # ✅ id2category_pred 里 None 的位置补齐为 ds 的 category（防止后面 upper(None)）
    for i in list(id2category_pred.keys()):
        if id2category_pred[i] is None:
            id2category_pred[i] = ds.id2category.get(i, INVALID_CAT)

    preds = []
    base = 0

    # diag counters (only count FINAL emitted quads)
    total_quads = 0
    global_cnt = 0
    null_asp_cnt = 0
    null_op_cnt = 0
    asp_hit = 0
    pair_hit = 0
    cat_counter = Counter()
    skipped_by_min_score = 0

    used_cat_head = 0
    used_prior_cat = 0
    prior_src_pair = 0
    prior_src_asp = 0
    prior_src_global = 0

    used_head_top2 = 0
    used_asp_top2 = 0

    invalid_id_ds = ds.category2id.get(INVALID_CAT, ds.num_categories - 1)

    for batch in tqdm(dl, desc="Predict", ncols=90):
        for k in batch:
            batch[k] = batch[k].to(device)

        with torch.no_grad():
            out = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])
            mat = torch.sigmoid(out["matrix"])     # [B, heads, L, L]
            seq_out = out["sequence_output"]       # [B, L, H]

        B, _, L, _ = mat.shape

        for bi in range(B):
            idx = base + bi
            sid = ds.df.iloc[idx]["id"]

            input_ids = batch["input_ids"][bi]
            attn = batch["attention_mask"][bi].bool()

            # valid positions
            valid_pos = attn.clone()
            if tok.sep_token_id is not None:
                valid_pos[input_ids == tok.sep_token_id] = False
            if tok.pad_token_id is not None:
                valid_pos[input_ids == tok.pad_token_id] = False
            valid_pos[0] = True  # CLS always valid

            mask_2d = valid_pos[:, None] & valid_pos[None, :]

            # NOTE: use large negative that is safe for fp16 later paths
            ba_bo = mat[bi, H_BA_BO].masked_fill(~mask_2d, -1e4)
            ea_eo = mat[bi, H_EA_EO].masked_fill(~mask_2d, -1e4)

            a_start_score = torch.max(ba_bo, dim=1).values
            o_end_score = torch.max(ea_eo, dim=0).values

            a_ok = topk_mask(a_start_score, valid_pos, args.topk_aux) & (a_start_score >= args.thr_aux)
            o_ok = topk_mask(o_end_score, valid_pos, args.topk_aux) & (o_end_score >= args.thr_aux)
            a_ok[0] = True
            o_ok[0] = True

            pair_mask = a_ok[:, None] & o_ok[None, :]

            # collect relation candidates
            pair_cands = []
            for h in rel_heads:
                pairs = extract_top_pairs(mat[bi, h], pair_mask, args.thr_rel, args.topk_rel, args.max_pair_dist)
                head_name = ds.label_types[h]
                dim_pred, sent_pred = parse_head_name(head_name)
                for (a_start, o_end, rel_sc) in pairs:
                    pair_cands.append((rel_sc, h, a_start, o_end, dim_pred, sent_pred))
            pair_cands.sort(key=lambda x: x[0], reverse=True)

            decoded = []
            pair_key2keys = {}

            for rel_sc, h, a_start, o_end, dim_pred, sent_pred in pair_cands:
                o_start, sc_bo = pick_o_start(ba_bo, valid_pos, a_start, o_end, args.max_span_len)
                a_end, sc_eo = pick_a_end(ea_eo, valid_pos, a_start, o_end, args.max_span_len)
                if sc_bo < -1e3 or sc_eo < -1e3:
                    continue

                if a_start == 0:
                    a_end = 0
                if o_end == 0:
                    o_start = 0

                a_txt = safe_decode(tok, input_ids, a_start, a_end)
                o_txt = safe_decode(tok, input_ids, o_start, o_end)

                if is_bad_span(a_txt, apostrophe_norm=args.apostrophe_norm):
                    continue
                if is_bad_span(o_txt, apostrophe_norm=args.apostrophe_norm):
                    o_txt = NULL_STR

                a_key = norm_key(a_txt, apostrophe_norm=args.apostrophe_norm)
                o_key = norm_key(o_txt, apostrophe_norm=args.apostrophe_norm)

                # optional span refine (kept as-is)
                if args.refine_span and o_txt != NULL_STR:
                    oi2, oj2, o_txt2 = prior_refine_span(
                        tok, input_ids, valid_pos, o_start, o_end,
                        freq_dict=op_freq,
                        max_seq_len=args.max_len,
                        max_span_len=args.max_span_len,
                        max_expand=args.span_expand,
                        apostrophe_norm=args.apostrophe_norm,
                        asp2cat=None,
                        pair2cat=None,
                        other_key=None,
                        w_freq=1.0, w_asp_hit=0.0, w_pair_hit=0.0,
                        min_gain=args.span_min_gain
                    )
                    if o_txt2 and not is_bad_span(o_txt2, apostrophe_norm=args.apostrophe_norm):
                        o_start, o_end, o_txt = oi2, oj2, o_txt2
                        o_key = norm_key(o_txt, apostrophe_norm=args.apostrophe_norm)

                if args.refine_span and a_txt != NULL_STR:
                    ai2, aj2, a_txt2 = prior_refine_span(
                        tok, input_ids, valid_pos, a_start, a_end,
                        freq_dict=asp_freq,
                        max_seq_len=args.max_len,
                        max_span_len=args.max_span_len,
                        max_expand=args.span_expand,
                        apostrophe_norm=args.apostrophe_norm,
                        asp2cat=asp2cat,
                        pair2cat=(pair2cat if o_key != "null" else None),
                        other_key=(o_key if o_key != "null" else None),
                        w_freq=1.0, w_asp_hit=0.9, w_pair_hit=1.2,
                        min_gain=args.span_min_gain
                    )
                    if a_txt2 and not is_bad_span(a_txt2, apostrophe_norm=args.apostrophe_norm):
                        a_start, a_end, a_txt = ai2, aj2, a_txt2
                        a_key = norm_key(a_txt, apostrophe_norm=args.apostrophe_norm)

                # null opinion gate
                if (o_txt != NULL_STR) and (float(rel_sc) < float(args.null_thr_o)) and (pair_freq.get((a_key, o_key), 0) == 0):
                    o_txt = NULL_STR
                    o_key = "null"
                    o_start, o_end = 0, 0

                score = float(rel_sc) + 0.6 * float(sc_bo) + 0.6 * float(sc_eo)

                key_pair = (int(a_start), int(o_end))
                pair_key2keys[key_pair] = (a_key, o_key)

                decoded.append({
                    "score": score,
                    "rel_sc": float(rel_sc),
                    "a_start": int(a_start), "a_end": int(a_end),
                    "o_start": int(o_start), "o_end": int(o_end),
                    "a_txt": a_txt, "o_txt": o_txt,
                    "a_key": a_key, "o_key": o_key,
                    "dim_pred": dim_pred, "sent_pred": sent_pred,
                })

            # ---- category head inference (pair-level) ----
            cat_by_pair = {}

            if args.cat_source != "prior" and getattr(model, "category_mlp", None) is not None and len(decoded) > 0:
                uniq = []
                seen_pair = set()
                for d in decoded:
                    key = (int(d["a_start"]), int(d["o_end"]))
                    if key not in seen_pair:
                        seen_pair.add(key)
                        uniq.append(key)

                pair_idx = torch.tensor([[0, a, o] for (a, o) in uniq], dtype=torch.long, device=device)
                logits = classify_pairs_batch(model, seq_out[bi:bi + 1], pair_idx, args.cat_head_batch)

                if logits is not None:
                    num_cat = int(logits.size(-1))

                    masks = []
                    for (a, o) in uniq:
                        a_key, o_key = pair_key2keys.get((a, o), ("null", "null"))
                        m = build_cand_mask_for_pair(
                            a_key=a_key,
                            o_key=o_key,
                            pair2cat=pair2cat,
                            asp2cat=asp2cat,
                            global_cat=global_cat_u,
                            ds_category2id=ds.category2id,
                            num_categories=num_cat,
                            invalid_id_ds=invalid_id_ds,
                            max_cands=12,
                        )
                        masks.append(m)

                    cand_mask = torch.stack(masks, dim=0).to(device)
                    logits = logits.masked_fill(~cand_mask, -1e4)

                    prob = F.softmax(logits, dim=-1)
                    topv, topi = torch.topk(prob, k=2, dim=-1)  # [N,2]

                    for ii, (a, o) in enumerate(uniq):
                        pid1 = int(topi[ii, 0].item())
                        pid2 = int(topi[ii, 1].item())

                        cat1 = id2category_pred.get(pid1, INVALID_CAT)
                        cat2 = id2category_pred.get(pid2, INVALID_CAT)

                        c1 = float(topv[ii, 0].item())
                        c2 = float(topv[ii, 1].item())

                        cat1_raw = str(cat1).upper() if cat1 is not None else INVALID_CAT
                        cat2_raw = str(cat2).upper() if cat2 is not None else INVALID_CAT
                        g = global_cat_u

                        head_used_raw = (c1 >= float(args.cat_head_min_conf) and cat1_raw != INVALID_CAT)

                        suppressed_global = False
                        head_used_eff = head_used_raw
                        if cat1_raw == g and c1 < float(args.global_cat_min_conf):
                            suppressed_global = True
                            # head-only / mix: 都不把它算作“可覆盖 prior 的 head_used”
                            head_used_eff = False

                        cat_by_pair[(a, o)] = (cat1_raw, c1, cat2_raw, c2, head_used_eff, suppressed_global)

            # ---- build candidates (score + quad dict) ----
            cand = []
            for d in decoded:
                a_txt = d["a_txt"]
                o_txt = d["o_txt"]
                a_key = d["a_key"]
                o_key = d["o_key"]

                used_pair_flag = False
                cat_prior = None
                prior_src = None

                # prior: pair
                if not (args.no_pair2cat_when_op_null and o_txt == NULL_STR):
                    cat_prior = pair2cat.get((a_key, o_key))
                    if cat_prior is not None:
                        used_pair_flag = True
                        prior_src = "pair"

                # prior: asp
                if cat_prior is None:
                    cat_prior = asp2cat.get(a_key)
                    if cat_prior is not None:
                        prior_src = "asp"

                # prior: global
                if cat_prior is None:
                    cat_prior = global_cat_u
                    prior_src = "global"

                used_cat_head_flag = False
                ac_used = None
                cat_internal = str(cat_prior).upper()

                key_pair = (int(d["a_start"]), int(d["o_end"]))
                head_pack = cat_by_pair.get(key_pair, None)

                # head/mix logic
                if args.cat_source != "prior" and getattr(model, "category_mlp", None) is not None and head_pack is not None:
                    cat1_raw, c1, cat2_raw, c2, head_used_eff, suppressed_global = head_pack
                    g = global_cat_u

                    if args.cat_source == "head":
                        # head-only：直接用 top1（即使 suppressed_global，也输出；但不算 used_head 覆盖）
                        cat_internal = cat1_raw
                        used_cat_head_flag = True
                    else:
                        # mix：只有 head_used_eff 才覆盖 prior
                        if head_used_eff and cat1_raw != INVALID_CAT:
                            cat_internal = cat1_raw
                            used_cat_head_flag = True
                        else:
                            cat_internal = str(cat_prior).upper()
                            used_cat_head_flag = False

                    # after you have decided cat_internal (either from head_used_eff or from prior)
                    if args.cat_source == "mix" and prior_src != "pair":
                        if str(cat_internal).upper() == global_cat_u:
                            # 1) try head top2
                            if head_pack is not None:
                                cat1_raw, c1, cat2_raw, c2, head_used_eff, suppressed_global = head_pack
                                if (cat2_raw != INVALID_CAT) and (cat2_raw != global_cat_u) and (c2 >= 0.15):
                                    cat_internal = cat2_raw
                                    ac_used = "head2"
                            # 2) else try asp top2
                            if str(cat_internal).upper() == global_cat_u:
                                tops = asp2cats_top.get(a_key, [])
                                if len(tops) >= 2:
                                    alt = str(tops[1][0]).upper()
                                    if alt != global_cat_u and alt != INVALID_CAT:
                                        cat_internal = alt
                                        ac_used = "asp2"

                # VA backfill / calibration
                v0 = cat2v.get(cat_internal, 5.0)
                a0 = cat2a.get(cat_internal, 5.0)
                if d["sent_pred"] is not None:
                    v0 = cat_sent2v.get((cat_internal, d["sent_pred"]), v0)
                if d["dim_pred"] is not None:
                    a0 = cat_dim2a.get((cat_internal, d["dim_pred"]), a0)
                va = fmt_va(v0, a0)

                cat_out = canon_cat_output(cat_internal, cat_case=args.cat_case, default_category=default_category)

                q = {
                    "Aspect": (fix_apostrophes(a_txt) if args.apostrophe_norm else clean_ws(a_txt)),
                    "Category": cat_out,
                    "Opinion": (NULL_STR if o_txt == NULL_STR else (fix_apostrophes(o_txt) if args.apostrophe_norm else clean_ws(o_txt))),
                    "VA": va
                }
                q[PRIOR_SRC_KEY] = prior_src
                if ac_used is not None:
                    q[AC_USED_KEY] = ac_used

                cand.append((float(d["score"]), q, used_pair_flag, used_cat_head_flag))

            # fallback: ensure at least 1 quad
            if not cand:
                i = int(torch.argmax(a_start_score).item())
                o_end = 0
                o_start, _ = pick_o_start(ba_bo, valid_pos, i, o_end, args.max_span_len)
                a_end, _ = pick_a_end(ea_eo, valid_pos, i, o_end, args.max_span_len)
                a_txt = safe_decode(tok, input_ids, i, a_end)
                if is_bad_span(a_txt, apostrophe_norm=args.apostrophe_norm):
                    a_txt = NULL_STR
                a_key = norm_key(a_txt, apostrophe_norm=args.apostrophe_norm)

                cat_internal = asp2cat.get(a_key, global_cat_u)
                cat_internal = str(cat_internal).upper()

                v0 = cat2v.get(cat_internal, 5.0)
                a0 = cat2a.get(cat_internal, 5.0)
                va = fmt_va(v0, a0)

                cat_out = canon_cat_output(cat_internal, cat_case=args.cat_case, default_category=default_category)

                q = {
                    "Aspect": (fix_apostrophes(a_txt) if args.apostrophe_norm else clean_ws(a_txt)),
                    "Category": cat_out,
                    "Opinion": NULL_STR,
                    "VA": va
                }
                q[PRIOR_SRC_KEY] = "fallback"
                cand = [(float(a_start_score[i].item()), q, False, False)]

            cand.sort(key=lambda x: x[0], reverse=True)
            if args.stage2_refine:
                cand = stage2_refine(cand, tok, args)

            # ---- select final quads (THIS PART MUST NOT DOUBLE COUNT) ----
            quads = []
            seen = set()

            for score, q, used_pair_flag, used_cat_head_flag in cand:
                prior_src = q.get(PRIOR_SRC_KEY, "unknown")
                ac_used_now = q.get(AC_USED_KEY, None)

                a_key2 = norm_key(q["Aspect"], apostrophe_norm=args.apostrophe_norm)
                o_key2 = norm_key(q["Opinion"], apostrophe_norm=args.apostrophe_norm)

                dedup_key = a_key2 if args.dedup_by_aspect else (a_key2, q["Category"], o_key2)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # min_score only gates 2nd+ quad
                if args.min_score is not None and float(args.min_score) > 0 and len(quads) >= 1:
                    if float(score) < float(args.min_score):
                        skipped_by_min_score += 1
                        continue

                # hit stats (only once for emitted quads)
                if used_pair_flag:
                    pair_hit += 1
                elif a_key2 in asp2cat:
                    asp_hit += 1

                # prior_src stats (only emitted quads)
                if prior_src == "pair":
                    prior_src_pair += 1
                elif prior_src == "asp":
                    prior_src_asp += 1
                else:
                    prior_src_global += 1

                # anti-collapse stats (only emitted quads)
                if ac_used_now == "head2":
                    used_head_top2 += 1
                elif ac_used_now == "asp2":
                    used_asp_top2 += 1

                # strip hidden fields BEFORE output
                q.pop(PRIOR_SRC_KEY, None)
                q.pop(AC_USED_KEY, None)

                quads.append(q)
                total_quads += 1

                cat_counter[q["Category"]] += 1

                if used_cat_head_flag:
                    used_cat_head += 1
                else:
                    used_prior_cat += 1

                if q["Category"].upper() == global_cat_u:
                    global_cnt += 1
                if a_key2 == "null":
                    null_asp_cnt += 1
                if o_key2 == "null":
                    null_op_cnt += 1

                if len(quads) >= int(args.max_quads):
                    break

            if not quads:
                q0 = dict(cand[0][1])
                q0.pop(PRIOR_SRC_KEY, None)
                q0.pop(AC_USED_KEY, None)
                quads = [q0]

            preds.append({"ID": sid, "Quadruplet": quads})

        base += B

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_jsonl(args.output, preds)
    logger.success(f"Saved -> {args.output}")

    if args.diag:
        global_rate = global_cnt / max(1, total_quads)
        null_asp_rate = null_asp_cnt / max(1, total_quads)
        null_op_rate = null_op_cnt / max(1, total_quads)
        logger.info(
            f"[Diag] total_quads={total_quads} global_rate={global_rate:.3f} "
            f"null_aspect_rate={null_asp_rate:.3f} null_opinion_rate={null_op_rate:.3f}"
        )
        logger.info(f"[Diag] asp_hit={asp_hit} pair_hit={pair_hit}")
        logger.info(f"[Diag] top10_cats={cat_counter.most_common(10)}")
        logger.info(f"[Diag] min_score={float(args.min_score):.3f} skipped_quads(2nd+ only)={skipped_by_min_score}")
        logger.info(f"[Diag] used_cat_head={used_cat_head} used_prior_cat={used_prior_cat}")
        logger.info(f"[Diag] prior_src pair={prior_src_pair} asp={prior_src_asp} global={prior_src_global}")
        logger.info(f"[Diag] anti_collapse used_head_top2={used_head_top2} used_asp_top2={used_asp_top2}")




if __name__ == "__main__":
    main()
