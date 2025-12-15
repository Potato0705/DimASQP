# -*- coding: utf-8 -*-
"""
DimASQP / SemEval Task3 - Predict (Horns-correct + Head-guided VA + Prior span refine + optional Retrieval cat/VA)

关键点：
1) 保留 CLS=0（隐式 NULL）
2) horns 正确解码：
   - relation head: (a_start, o_end)
   - BA-BO: (a_start, o_start)
   - EA-EO: (a_end, o_end)
3) VA：head 指示 dim/sent + cat 条件中位数
4) refine_span：保守扩展（可选）
5) max_quads 用 min_score 控制（强烈建议）：2nd+ quadruplet 若 score<min_score 则不输出（降 FP）
6) 检索只做 cat/VA 校正（可选）：encoder 同源 embedding
"""

import os
import json
import re
import math
import argparse
from collections import Counter, defaultdict
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer

from dataset.dataset import AcqpDataset, collate_fn
from models.model import QuadrupleModel
from utils.utils import set_seeds

set_seeds(42)

ALNUM_RE = re.compile(r"[A-Za-z0-9]")

STOPWORDS = {
    "a","an","the","this","that","these","those","it","its","i","you","we","they",
    "is","are","was","were","be","been","being","am",
    "and","or","but","if","then","than","so","to","of","in","on","for","with","as","at","by",
    "not","no","yes","do","does","did","done","have","has","had",
}

NULL_STR = "null"


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


def clean_ws(s: str) -> str:
    return " ".join(str(s).strip().split()) if s is not None else ""


def norm_key(s: str) -> str:
    t = clean_ws(s).lower()
    if not t or t in {"null", "none"}:
        return "null"
    return t


def is_bad_span(text: str) -> bool:
    if text is None:
        return True
    t = clean_ws(text)
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
    ids = input_ids_1d[i:j+1].tolist()
    s = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return clean_ws(s)


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
    return 0.5 * (xs[m-1] + xs[m])


def canon_cat(s: str, cat_case: str = "upper") -> str:
    t = clean_ws(s) or "LAPTOP#GENERAL"
    return t.lower() if cat_case == "lower" else t.upper()


def build_priors(train_stats_path, cat_case="upper", va_stat="median"):
    rows = read_jsonl(train_stats_path)

    cat_cnt = Counter()
    asp_cat_cnt = Counter()
    pair_cat_cnt = Counter()

    asp_freq = Counter()
    op_freq = Counter()
    pair_freq = Counter()

    # VA stats
    cat_v = defaultdict(list)
    cat_a = defaultdict(list)
    cat_va_mode = defaultdict(Counter)
    cat_sent_v = defaultdict(list)   # (cat,sent)->[v]
    cat_dim_a = defaultdict(list)    # (cat,dim)->[a]

    for r in rows:
        qs = r.get("Quadruplet") or r.get("Quadruplets") or r.get("quadruplets") or []
        for q in qs:
            a_raw = q.get("Aspect", "null")
            o_raw = q.get("Opinion", "null")
            c_raw = q.get("Category", "LAPTOP#GENERAL")
            va = q.get("VA", "5.00#5.00")

            a = norm_key(a_raw)
            o = norm_key(o_raw)
            c = canon_cat(c_raw, cat_case=cat_case)

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
                cat_sent_v[(c, va_to_sentiment(v))].append(v)
                cat_dim_a[(c, va_to_dimension(ar))].append(ar)

    global_cat = cat_cnt.most_common(1)[0][0] if len(cat_cnt) else canon_cat("LAPTOP#GENERAL", cat_case=cat_case)

    # pair->cat (mode)
    pair2cat = {}
    best = {}
    for (a, o, c), cnt in pair_cat_cnt.items():
        key = (a, o)
        if key not in best or cnt > best[key][1]:
            best[key] = (c, cnt)
    for k, (c, _) in best.items():
        pair2cat[k] = c

    # asp->cat (mode)
    asp2cat = {}
    best2 = {}
    for (a, c), cnt in asp_cat_cnt.items():
        if a not in best2 or cnt > best2[a][1]:
            best2[a] = (c, cnt)
    for a, (c, _) in best2.items():
        asp2cat[a] = c

    # cat overall va
    cat2v, cat2a = {}, {}
    for c in cat_cnt.keys():
        if va_stat == "mode":
            va0 = cat_va_mode[c].most_common(1)[0][0] if len(cat_va_mode[c]) else fmt_va(5.0, 5.0)
            xy = parse_va(va0) or (5.0, 5.0)
            cat2v[c], cat2a[c] = xy
        else:
            mv = median(cat_v.get(c, []))
            ma = median(cat_a.get(c, []))
            if mv is None or ma is None:
                va0 = cat_va_mode[c].most_common(1)[0][0] if len(cat_va_mode[c]) else fmt_va(5.0, 5.0)
                xy = parse_va(va0) or (5.0, 5.0)
                cat2v[c], cat2a[c] = xy
            else:
                cat2v[c], cat2a[c] = float(mv), float(ma)

    # cat×sent v
    cat_sent2v = {}
    for (c, s), vs in cat_sent_v.items():
        mv = median(vs)
        if mv is not None:
            cat_sent2v[(c, s)] = float(mv)

    # cat×dim a
    cat_dim2a = {}
    for (c, d), ars in cat_dim_a.items():
        ma = median(ars)
        if ma is not None:
            cat_dim2a[(c, d)] = float(ma)

    logger.info(
        f"[Priors] pair2cat={len(pair2cat)} asp2cat={len(asp2cat)} "
        f"cat_sent2v={len(cat_sent2v)} cat_dim2a={len(cat_dim2a)} "
        f"global_cat={global_cat} va_stat={va_stat}"
    )
    return pair2cat, asp2cat, cat_sent2v, cat_dim2a, cat2v, cat2a, global_cat, asp_freq, op_freq, pair_freq


def parse_head_name(head_name: str):
    # BA-EO-{dim}-{sent} OR BA-EO-{sent}
    if not head_name.startswith("BA-EO"):
        return None, None
    parts = head_name.split("-")
    if len(parts) == 4:
        return parts[2], parts[3]
    if len(parts) == 3:
        return None, parts[2]
    return None, None


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
    if max_span_len and max_span_len > 0:
        lo = max(0, o_end - max_span_len + 1)
        mask[:lo] = False
    row = ba_bo[a_start].masked_fill(~mask, -1e9)
    k = int(torch.argmax(row).item())
    return k, float(row[k].item())


def pick_a_end(ea_eo, valid_pos, a_start, o_end, max_span_len):
    mask = valid_pos.clone()
    mask[:a_start] = False
    if max_span_len and max_span_len > 0:
        hi = min(ea_eo.shape[0] - 1, a_start + max_span_len - 1)
        if hi < ea_eo.shape[0] - 1:
            mask[(hi + 1):] = False
    col = ea_eo[:, o_end].masked_fill(~mask, -1e9)
    t = int(torch.argmax(col).item())
    return t, float(col[t].item())


def prior_refine_span(tokenizer, input_ids, valid_pos, i, j, freq_dict, max_span_len, max_expand=2):
    if i == 0 and j == 0:
        return i, j, NULL_STR

    best_i, best_j = i, j
    best_txt = safe_decode(tokenizer, input_ids, i, j)
    if is_bad_span(best_txt):
        return i, j, best_txt

    def score(txt):
        k = norm_key(txt)
        return math.log(1.0 + float(freq_dict.get(k, 0)))

    best_sc = score(best_txt)

    for d in range(1, max_expand + 1):
        nj = j + d
        if nj < max_span_len and bool(valid_pos[nj].item()):
            txt = safe_decode(tokenizer, input_ids, i, nj)
            if not is_bad_span(txt):
                sc = score(txt)
                if sc > best_sc + 0.3:
                    best_sc, best_i, best_j, best_txt = sc, i, nj, txt

        ni = i - d
        if ni > 0 and bool(valid_pos[ni].item()):
            txt = safe_decode(tokenizer, input_ids, ni, j)
            if not is_bad_span(txt):
                sc = score(txt)
                if sc > best_sc + 0.3:
                    best_sc, best_i, best_j, best_txt = sc, ni, j, txt

    return best_i, best_j, best_txt


def load_model(ckpt, model_name, num_label_types, num_dims, max_len, device):
    model = QuadrupleModel(num_label_types, num_dims, max_len, model_name).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# -------- Retrieval (optional) --------
def load_retrieval_index(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    emb = data["emb"].astype(np.float32)  # [N,H], L2 normalized
    meta = data["meta"].tolist()         # list[dict]
    return emb, meta


def retrieval_correct_cat_va(
    query_emb: np.ndarray,
    index_emb: np.ndarray,
    meta: List[Dict[str, Any]],
    topk: int = 16,
    min_sim: float = 0.35,
    vote_margin: float = 0.05,
):
    """
    只做 cat/VA 校正：
      - cat: 相似度加权投票
      - VA: topk 的 valence/arousal 加权中位数（更稳），这里用加权均值简化也可
    """
    sims = index_emb @ query_emb  # [N]
    if sims.size == 0:
        return None
    k = min(int(topk), int(sims.size))
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    if sims[idx[0]] < min_sim:
        return None

    # cat vote
    cat_scores = defaultdict(float)
    va_v = []
    va_a = []
    wts = []
    for j in idx:
        w = float(sims[j])
        m = meta[j]
        cat = m.get("Category", None)
        v = m.get("V", None)
        a = m.get("A", None)
        if cat is not None:
            cat_scores[cat] += w
        if v is not None and a is not None:
            va_v.append(float(v))
            va_a.append(float(a))
            wts.append(w)

    if not cat_scores:
        return None
    best_cat, best_sc = max(cat_scores.items(), key=lambda x: x[1])
    # margin check：防止不稳
    top2 = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    if len(top2) == 2 and (top2[0][1] - top2[1][1]) < vote_margin:
        return None

    # VA：加权均值（足够稳定，后续可升级加权中位数）
    if wts:
        wsum = max(1e-9, float(sum(wts)))
        v_hat = float(sum(w * v for w, v in zip(wts, va_v)) / wsum)
        a_hat = float(sum(w * a for w, a in zip(wts, va_a)) / wsum)
    else:
        v_hat, a_hat = None, None

    return {"Category": best_cat, "V": v_hat, "A": a_hat, "sim": float(sims[idx[0]])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train_stats", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)

    ap.add_argument("--label_pattern", type=str, default="sentiment_dim",
                    choices=["raw", "sentiment", "sentiment_dim"])

    ap.add_argument("--thr_aux", type=float, default=0.05)
    ap.add_argument("--topk_aux", type=int, default=80)
    ap.add_argument("--max_span_len", type=int, default=12)

    ap.add_argument("--thr_rel", type=float, default=0.12)
    ap.add_argument("--topk_rel", type=int, default=800)
    ap.add_argument("--max_pair_dist", type=int, default=120)

    ap.add_argument("--max_quads", type=int, default=2)
    ap.add_argument("--min_score", type=float, default=0.0)  # 仅用于 2nd+
    ap.add_argument("--null_thr_o", type=float, default=0.10)

    ap.add_argument("--va_stat", type=str, default="median", choices=["mode", "median", "mean"])
    ap.add_argument("--cat_case", type=str, default="upper", choices=["upper", "lower"])

    ap.add_argument("--refine_span", action="store_true")

    # retrieval options
    ap.add_argument("--retrieval_index", type=str, default="")
    ap.add_argument("--ret_topk", type=int, default=16)
    ap.add_argument("--ret_min_sim", type=float, default=0.35)
    ap.add_argument("--ret_vote_margin", type=float, default=0.05)

    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Device] {device}")

    pair2cat, asp2cat, cat_sent2v, cat_dim2a, cat2v, cat2a, global_cat, asp_freq, op_freq, pair_freq = build_priors(
        args.train_stats, cat_case=args.cat_case, va_stat=args.va_stat
    )

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    ds = AcqpDataset("PredictSet", args.input, args.max_len, tok, label_pattern=args.label_pattern)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    if "BA-BO" not in ds.label_types or "EA-EO" not in ds.label_types:
        raise RuntimeError(f"label_types missing BA-BO/EA-EO: {ds.label_types}")
    H_BA_BO = ds.label_types.index("BA-BO")
    H_EA_EO = ds.label_types.index("EA-EO")
    rel_heads = [i for i, nm in enumerate(ds.label_types) if nm not in ["BA-BO", "EA-EO"]]

    model = load_model(args.ckpt, args.model_name, len(ds.label_types), len(ds.dimension2id), args.max_len, device)

    # retrieval index
    index_emb, index_meta = None, None
    if args.retrieval_index:
        index_emb, index_meta = load_retrieval_index(args.retrieval_index)
        logger.info(f"[Retrieval] loaded index: N={index_emb.shape[0]} dim={index_emb.shape[1]}")

    preds = []
    base = 0

    total_quads = 0
    global_cnt = 0
    null_asp_cnt = 0
    null_op_cnt = 0
    asp_hit = 0
    pair_hit = 0
    cat_counter = Counter()

    skipped_2p = 0

    for batch in tqdm(dl, desc="Predict", ncols=90):
        for k in batch:
            batch[k] = batch[k].to(device)

        with torch.no_grad():
            out = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])
            mat = torch.sigmoid(out["matrix"])  # [B,C,L,L]

        B, _, L, _ = mat.shape

        for bi in range(B):
            idx = base + bi
            sid = ds.df.iloc[idx]["id"]
            text = ds.df.iloc[idx]["text"]

            input_ids = batch["input_ids"][bi]
            attn = batch["attention_mask"][bi].bool()

            # valid positions: keep CLS, mask SEP/PAD
            valid_pos = attn.clone()
            valid_pos[input_ids == tok.sep_token_id] = False
            valid_pos[input_ids == tok.pad_token_id] = False
            valid_pos[0] = True

            mask_2d = valid_pos[:, None] & valid_pos[None, :]

            ba_bo = mat[bi, H_BA_BO].masked_fill(~mask_2d, -1e9)  # (a_start, o_start)
            ea_eo = mat[bi, H_EA_EO].masked_fill(~mask_2d, -1e9)  # (a_end, o_end)

            a_start_score = torch.max(ba_bo, dim=1).values
            o_end_score = torch.max(ea_eo, dim=0).values

            a_ok = topk_mask(a_start_score, valid_pos, args.topk_aux) & (a_start_score >= args.thr_aux)
            o_ok = topk_mask(o_end_score, valid_pos, args.topk_aux) & (o_end_score >= args.thr_aux)
            a_ok[0] = True
            o_ok[0] = True

            pair_mask = a_ok[:, None] & o_ok[None, :]

            pair_cands = []
            for h in rel_heads:
                pairs = extract_top_pairs(mat[bi, h], pair_mask, args.thr_rel, args.topk_rel, args.max_pair_dist)
                head_name = ds.label_types[h]
                dim_pred, sent_pred = parse_head_name(head_name)
                for (a_start, o_end, rel_sc) in pairs:
                    pair_cands.append((rel_sc, a_start, o_end, dim_pred, sent_pred))

            pair_cands.sort(key=lambda x: x[0], reverse=True)

            cand = []  # (score, quad, used_pair)

            for rel_sc, a_start, o_end, dim_pred, sent_pred in pair_cands:
                o_start, sc_bo = pick_o_start(ba_bo, valid_pos, a_start, o_end, args.max_span_len)
                a_end, sc_eo = pick_a_end(ea_eo, valid_pos, a_start, o_end, args.max_span_len)

                if a_start == 0:
                    a_end = 0
                if o_end == 0:
                    o_start = 0

                a_txt = safe_decode(tok, input_ids, a_start, a_end)
                o_txt = safe_decode(tok, input_ids, o_start, o_end)

                if args.refine_span:
                    a_start2, a_end2, a_txt2 = prior_refine_span(tok, input_ids, valid_pos, a_start, a_end, asp_freq, args.max_span_len, 2)
                    if a_txt2 and not is_bad_span(a_txt2):
                        a_start, a_end, a_txt = a_start2, a_end2, a_txt2
                    if o_txt != NULL_STR:
                        o_start2, o_end2, o_txt2 = prior_refine_span(tok, input_ids, valid_pos, o_start, o_end, op_freq, args.max_span_len, 2)
                        if o_txt2 and not is_bad_span(o_txt2):
                            o_start, o_end, o_txt = o_start2, o_end2, o_txt2

                if is_bad_span(a_txt):
                    continue
                if is_bad_span(o_txt):
                    o_txt = NULL_STR

                a_key = norm_key(a_txt)
                o_key = norm_key(o_txt)

                # opinion null rule
                if (o_txt != NULL_STR) and (rel_sc < args.null_thr_o) and (pair_freq.get((a_key, o_key), 0) == 0):
                    o_txt = NULL_STR
                    o_key = "null"

                # category mapping
                used_pair = False
                cat = pair2cat.get((a_key, o_key))
                if cat is not None:
                    used_pair = True
                else:
                    cat = asp2cat.get(a_key, None)
                if cat is None:
                    cat = global_cat

                # VA head-guided
                v0 = cat2v.get(cat, 5.0)
                a0 = cat2a.get(cat, 5.0)
                if sent_pred is not None:
                    v0 = cat_sent2v.get((cat, sent_pred), v0)
                if dim_pred is not None:
                    a0 = cat_dim2a.get((cat, dim_pred), a0)

                # optional retrieval correction: only cat/VA
                if index_emb is not None and index_meta is not None:
                    # query = text [SEP] aspect [SEP] opinion
                    qtext = f"{clean_ws(text)} [SEP] {clean_ws(a_txt)} [SEP] {clean_ws(o_txt)}"
                    enc = tok(qtext, truncation=True, max_length=args.max_len, padding="max_length", return_tensors="pt")
                    enc = {k: v.to(device) for k, v in enc.items()}
                    qemb = model.encode_embeddings(enc["input_ids"], enc.get("token_type_ids", None), enc["attention_mask"], pooling="mean", normalize=True)
                    qemb = qemb[0].detach().cpu().numpy().astype(np.float32)

                    corr = retrieval_correct_cat_va(
                        qemb, index_emb, index_meta,
                        topk=args.ret_topk,
                        min_sim=args.ret_min_sim,
                        vote_margin=args.ret_vote_margin
                    )
                    if corr is not None:
                        cat = corr["Category"]
                        if corr["V"] is not None and corr["A"] is not None:
                            v0, a0 = corr["V"], corr["A"]

                va = fmt_va(v0, a0)

                score = float(rel_sc) + 0.6 * float(sc_bo) + 0.6 * float(sc_eo)

                cand.append((score, {
                    "Aspect": clean_ws(a_txt),
                    "Category": cat,
                    "Opinion": clean_ws(o_txt) if o_txt != NULL_STR else NULL_STR,
                    "VA": va
                }, used_pair))

            if not cand:
                # fallback: pick best a_start and output null opinion
                i = int(torch.argmax(a_start_score).item())
                o_end = 0
                o_start, sc_bo = pick_o_start(ba_bo, valid_pos, i, o_end, args.max_span_len)
                a_end, sc_eo = pick_a_end(ea_eo, valid_pos, i, o_end, args.max_span_len)
                a_txt = safe_decode(tok, input_ids, i, a_end)
                if is_bad_span(a_txt):
                    a_txt = NULL_STR
                a_key = norm_key(a_txt)
                cat = asp2cat.get(a_key, global_cat)
                v0 = cat2v.get(cat, 5.0)
                a0 = cat2a.get(cat, 5.0)
                va = fmt_va(v0, a0)
                cand = [(float(a_start_score[i].item()), {
                    "Aspect": clean_ws(a_txt),
                    "Category": cat,
                    "Opinion": NULL_STR,
                    "VA": va
                }, False)]

            cand.sort(key=lambda x: x[0], reverse=True)

            quads = []
            seen = set()

            for rank, (score, q, used_pair) in enumerate(cand):
                # min_score 只用于 2nd+
                if rank >= 1 and args.min_score > 0 and float(score) < float(args.min_score):
                    skipped_2p += 1
                    continue

                a_key = norm_key(q["Aspect"])
                o_key = norm_key(q["Opinion"])
                dedup_key = (a_key, q["Category"], o_key)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                quads.append(q)

                total_quads += 1
                cat_counter[q["Category"]] += 1
                if q["Category"] == global_cat:
                    global_cnt += 1
                if a_key == "null":
                    null_asp_cnt += 1
                if o_key == "null":
                    null_op_cnt += 1
                if used_pair:
                    pair_hit += 1
                elif a_key in asp2cat:
                    asp_hit += 1

                if len(quads) >= args.max_quads:
                    break

            preds.append({"ID": sid, "Quadruplet": quads})

        base += B

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_jsonl(args.output, preds)
    logger.success(f"Saved -> {args.output}")

    if args.diag:
        global_rate = global_cnt / max(1, total_quads)
        null_asp_rate = null_asp_cnt / max(1, total_quads)
        null_op_rate = null_op_cnt / max(1, total_quads)
        logger.info(f"[Diag] total_quads={total_quads} global_rate={global_rate:.3f} null_aspect_rate={null_asp_rate:.3f} null_opinion_rate={null_op_rate:.3f}")
        logger.info(f"[Diag] asp_hit={asp_hit} pair_hit={pair_hit}")
        logger.info(f"[Diag] top10_cats={cat_counter.most_common(10)}")
        logger.info(f"[Diag] min_score={args.min_score:.3f} skipped_quads(2nd+ only)={skipped_2p}")


if __name__ == "__main__":
    main()
