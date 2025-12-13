# -*- coding: utf-8 -*-
"""
SemEval-2026 Task3 (DimASQP) - Predict Script (V3)
目标：
- 降低 global fallback (LAPTOP#GENERAL + 默认 VA) 的比例
- 修复 span end 解码容易把多余词粘进去的问题
- VA 使用更稳健的统计量（默认 median），避免被 5.00#5.00 统治

核心策略：
1) relation heads 提名 pair (a_start, o_start)
2) span heads (BA-BO / EA-EO) 在局部窗口内搜索最优 end，并加 prior bonus
3) Category 用 pair2cat -> asp2cat -> global 的 canonical 映射
4) VA 用 cat->median(V,A)（或 mean）并格式化成 "x.xx#y.yy"
"""

import os
import re
import json
import math
import argparse
from collections import Counter, defaultdict

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer

from dataset.dataset import AcqpDataset, collate_fn
from models.model import QuadrupleModel
from utils.utils import set_seeds

set_seeds(42)

ALNUM_RE = re.compile(r"[A-Za-z0-9]")

# ----------------------------
# I/O
# ----------------------------
def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fw:
        for x in rows:
            fw.write(json.dumps(x, ensure_ascii=False) + "\n")

# ----------------------------
# Text utils
# ----------------------------
def clean_ws(s: str) -> str:
    return " ".join(str(s).strip().split()) if s is not None else ""

def norm_key(s: str) -> str:
    """用于 priors key：小写 + 去多余空白"""
    t = clean_ws(s)
    return t.lower() if t else "null"

def canon_cat(s: str, case: str = "upper") -> str:
    """Category canonical：upper/lower"""
    t = clean_ws(s)
    if not t:
        t = "LAPTOP#GENERAL"
    if case == "lower":
        return t.lower()
    return t.upper()

def is_bad_span(text: str) -> bool:
    """过滤：纯标点/太短/无字母数字"""
    if text is None:
        return True
    t = clean_ws(text)
    if len(t) < 2:
        return True
    if not ALNUM_RE.search(t):
        return True
    return False

def safe_decode(tokenizer, input_ids_1d, i, j):
    ids = input_ids_1d[i : j + 1].tolist()
    s = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return clean_ws(s)

# ----------------------------
# Priors
# ----------------------------
def build_priors(train_stats_path, cat_case="upper", va_stat="median"):
    """
    返回：
    - pair2cat: (a_key,o_key)->cat_canon(众数)
    - asp2cat : a_key->cat_canon(众数)
    - cat2va  : cat_canon->(V,A) 使用 mean/median
    - global_cat
    - asp_freq / op_freq / pair_freq
    """
    rows = read_jsonl(train_stats_path)

    cat_cnt = Counter()
    asp_cat_cnt = Counter()
    pair_cat_cnt = Counter()

    asp_freq = Counter()
    op_freq = Counter()
    pair_freq = Counter()

    cat_vs = defaultdict(list)  # cat -> list of V
    cat_as = defaultdict(list)  # cat -> list of A

    for r in rows:
        qs = r.get("Quadruplet") or r.get("Quadruplets") or r.get("quadruplets") or []
        for q in qs:
            a_raw = q.get("Aspect", "null")
            o_raw = q.get("Opinion", "null")
            c_raw = q.get("Category", "LAPTOP#GENERAL")
            va = q.get("VA", "5.00#5.00")

            a = norm_key(a_raw)
            o = norm_key(o_raw)
            c = canon_cat(c_raw, case=cat_case)

            asp_freq[a] += 1
            op_freq[o] += 1
            pair_freq[(a, o)] += 1

            cat_cnt[c] += 1
            asp_cat_cnt[(a, c)] += 1
            pair_cat_cnt[(a, o, c)] += 1

            try:
                v_str, a_str = str(va).split("#")
                v, ar = float(v_str), float(a_str)
                cat_vs[c].append(v)
                cat_as[c].append(ar)
            except:
                pass

    global_cat = cat_cnt.most_common(1)[0][0] if len(cat_cnt) else canon_cat("LAPTOP#GENERAL", case=cat_case)

    # pair -> cat (mode)
    pair2cat = {}
    best = {}
    for (a, o, c), cnt in pair_cat_cnt.items():
        key = (a, o)
        if key not in best or cnt > best[key][1]:
            best[key] = (c, cnt)
    for k, (c, _) in best.items():
        pair2cat[k] = c

    # asp -> cat (mode)
    asp2cat = {}
    best2 = {}
    for (a, c), cnt in asp_cat_cnt.items():
        if a not in best2 or cnt > best2[a][1]:
            best2[a] = (c, cnt)
    for a, (c, _) in best2.items():
        asp2cat[a] = c

    # cat -> VA (mean/median)
    cat2va = {}
    for c in cat_cnt.keys():
        vs = cat_vs.get(c, [])
        ars = cat_as.get(c, [])
        if len(vs) == 0 or len(ars) == 0:
            continue
        if va_stat == "mean":
            v = float(np.mean(vs))
            ar = float(np.mean(ars))
        else:
            v = float(np.median(vs))
            ar = float(np.median(ars))
        # 保守 clamp（避免异常）
        v = max(0.0, min(10.0, v))
        ar = max(0.0, min(10.0, ar))
        cat2va[c] = (v, ar)

    logger.info(
        f"[Priors] pair2cat={len(pair2cat)} asp2cat={len(asp2cat)} cat2va={len(cat2va)} "
        f"global_cat={global_cat} va_stat={va_stat}"
    )
    return pair2cat, asp2cat, cat2va, global_cat, asp_freq, op_freq, pair_freq

# ----------------------------
# Pair extraction from relation heads
# ----------------------------
def extract_top_pairs(scores_LL, valid_pos, thr, topk, max_pair_dist=None):
    """
    scores_LL: [L,L] sigmoid后（关系头）
    返回 (i,j,score) 按 score 降序
    """
    L = scores_LL.shape[0]
    device = scores_LL.device

    vp = valid_pos.to(device)
    ii = vp[:, None].expand(L, L)
    jj = vp[None, :].expand(L, L)
    mask = ii & jj

    tri = torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=0)
    mask = mask & tri

    if max_pair_dist is not None and max_pair_dist > 0:
        idx = torch.arange(L, device=device)
        dist = (idx[None, :] - idx[:, None])  # j-i
        mask = mask & (dist >= 0) & (dist <= max_pair_dist)

    flat = scores_LL.masked_fill(~mask, -1e9).view(-1)
    cand = (flat >= thr)
    if not cand.any():
        return []

    idxs = torch.nonzero(cand, as_tuple=False).squeeze(-1)
    vals = flat[idxs]
    if idxs.numel() > topk:
        topv, topi = torch.topk(vals, k=topk, largest=True)
        idxs = idxs[topi]
        vals = topv

    order = torch.argsort(vals, descending=True)
    idxs = idxs[order]
    vals = vals[order]

    out = []
    for k in range(idxs.numel()):
        p = int(idxs[k].item())
        i = p // L
        j = p % L
        out.append((i, j, float(vals[k].item())))
    return out

# ----------------------------
# Span end search with prior bonus
# ----------------------------
def best_span_end(
    tokenizer,
    input_ids,
    valid_pos,
    span_mat,          # [L,L] for BA-BO or EA-EO
    start_i: int,
    max_span_len: int,
    priors_freq: Counter = None,
    priors_hit: dict = None,
    prior_alpha: float = 0.7,
    len_penalty: float = 0.08,
):
    """
    在 [start_i, start_i+max_span_len) 搜索 end：
    objective = span_score + alpha*log(freq+1) - len_penalty*(len-1)
    若 priors_hit 提供，则只有命中才给 freq 奖励（更稳健）
    返回 (best_end, best_text, raw_span_score, best_obj, key_norm)
    """
    L = span_mat.shape[0]
    jmax = min(L - 1, start_i + max_span_len - 1)
    best = None

    for end_j in range(start_i, jmax + 1):
        if not bool(valid_pos[end_j].item()):
            continue
        sc = float(span_mat[start_i, end_j].item())
        txt = safe_decode(tokenizer, input_ids, start_i, end_j)
        if is_bad_span(txt):
            continue

        key = norm_key(txt)
        freq = 0
        if priors_freq is not None:
            freq = int(priors_freq.get(key, 0))

        hit_ok = True
        if priors_hit is not None:
            hit_ok = (key in priors_hit)

        bonus = prior_alpha * math.log(freq + 1.0) if hit_ok else 0.0
        length = max(1, len(txt.split()))
        obj = sc + bonus - len_penalty * (length - 1)

        if (best is None) or (obj > best[0]):
            best = (obj, end_j, txt, sc, key)

    if best is None:
        # fallback：只取 start 自己
        txt = safe_decode(tokenizer, input_ids, start_i, start_i)
        key = norm_key(txt)
        sc = float(span_mat[start_i, start_i].item())
        return start_i, txt, sc, sc, key

    _, end_j, txt, sc, key = best
    return end_j, txt, sc, best[0], key

# ----------------------------
# Model load
# ----------------------------
def load_model(ckpt, model_name, num_label_types, num_dims, max_len, device):
    model = QuadrupleModel(num_label_types, num_dims, max_len, model_name).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train_stats", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)

    # span heads（BA-BO / EA-EO）
    ap.add_argument("--thr_aux", type=float, default=0.05)
    ap.add_argument("--topk_aux", type=int, default=80)
    ap.add_argument("--max_span_len", type=int, default=12)

    # relation heads（BA-EO-...）
    ap.add_argument("--thr_rel", type=float, default=0.15)
    ap.add_argument("--topk_rel", type=int, default=800)
    ap.add_argument("--max_pair_dist", type=int, default=160)

    ap.add_argument("--max_quads", type=int, default=2)
    ap.add_argument("--null_thr_o", type=float, default=0.12)   # opinion 判空阈值（结合 rel + prior）
    ap.add_argument("--cat_case", type=str, default="upper", choices=["upper", "lower"])
    ap.add_argument("--va_stat", type=str, default="median", choices=["median", "mean"])

    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Device] {device}")

    pair2cat, asp2cat, cat2va, global_cat, asp_freq, op_freq, pair_freq = build_priors(
        args.train_stats, cat_case=args.cat_case, va_stat=args.va_stat
    )

    tok = AutoTokenizer.from_pretrained(args.model_name)
    ds = AcqpDataset("PredictSet", args.input, args.max_len, tok, label_pattern="sentiment_dim")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    # heads
    AC = ds.label_types.index("BA-BO")
    EO = ds.label_types.index("EA-EO")
    rel_heads = [i for i, nm in enumerate(ds.label_types) if nm not in ["BA-BO", "EA-EO"]]

    model = load_model(args.ckpt, args.model_name, len(ds.label_types), len(ds.dimension2id), args.max_len, device)

    preds = []
    base = 0

    # diagnostics
    diag_total_quads = 0
    diag_global = 0
    diag_null_a = 0
    diag_null_o = 0
    diag_asp_hit = 0
    diag_pair_hit = 0
    diag_cat = Counter()

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

            input_ids = batch["input_ids"][bi]
            attn = batch["attention_mask"][bi].bool()

            # valid positions: attn=1 & not special
            valid_pos = attn.clone()
            specials = torch.tensor([tok.cls_token_id, tok.sep_token_id, tok.pad_token_id], device=input_ids.device)
            is_special = (input_ids[..., None] == specials[None, ...]).any(dim=-1)
            valid_pos = valid_pos & (~is_special)

            # ----- 1) relation heads 提名 pairs
            pair_map = {}  # (i,j) -> best_rel_score
            for h in rel_heads:
                pairs = extract_top_pairs(
                    mat[bi, h],
                    valid_pos,
                    thr=args.thr_rel,
                    topk=args.topk_rel,
                    max_pair_dist=args.max_pair_dist,
                )
                for (i, j, sc) in pairs:
                    key = (i, j)
                    if (key not in pair_map) or (sc > pair_map[key]):
                        pair_map[key] = sc

            pair_cands = [(sc, i, j) for (i, j), sc in pair_map.items()]
            pair_cands.sort(key=lambda x: x[0], reverse=True)

            quads = []
            used = set()

            # ----- 2) 对每个 pair：局部搜索 best aspect/opinion end（带 prior bonus）
            for (rel_sc, i, j) in pair_cands:
                if len(quads) >= args.max_quads:
                    break

                # aspect end search
                ac_mat = mat[bi, AC]  # [L,L]
                # 将无效位置 mask 掉
                mask_2d = (valid_pos[:, None] & valid_pos[None, :])
                ac_mat_m = ac_mat.masked_fill(~mask_2d, -1e9)

                a_end, a_text, a_span_sc, a_obj, a_key = best_span_end(
                    tok, input_ids, valid_pos, ac_mat_m, i, args.max_span_len,
                    priors_freq=asp_freq, priors_hit=asp2cat, prior_alpha=0.9, len_penalty=0.10
                )
                if is_bad_span(a_text) or a_key == "null":
                    continue

                # opinion end search
                eo_mat = mat[bi, EO]
                eo_mat_m = eo_mat.masked_fill(~mask_2d, -1e9)

                o_end, o_text, o_span_sc, o_obj, o_key = best_span_end(
                    tok, input_ids, valid_pos, eo_mat_m, j, args.max_span_len,
                    priors_freq=op_freq, priors_hit=None, prior_alpha=0.4, len_penalty=0.08
                )
                if is_bad_span(o_text):
                    o_text = "null"
                    o_key = "null"

                # opinion 判空：rel 低 + pair 未见过 + opinion span 低
                pf = int(pair_freq.get((a_key, o_key), 0))
                if (o_key != "null") and (rel_sc < args.null_thr_o) and (pf == 0) and (o_span_sc < args.thr_aux):
                    o_text = "null"
                    o_key = "null"

                # category：pair -> asp -> global
                cat = pair2cat.get((a_key, o_key))
                if cat is None:
                    cat = asp2cat.get(a_key)
                if cat is None:
                    cat = global_cat

                # VA：cat 的 mean/median
                if cat in cat2va:
                    v, ar = cat2va[cat]
                    va = f"{v:.2f}#{ar:.2f}"
                else:
                    va = "5.00#5.00"

                tup = (a_key, o_key, cat)
                if tup in used:
                    continue
                used.add(tup)

                quads.append({
                    "Aspect": clean_ws(a_text),
                    "Category": cat,
                    "Opinion": "null" if o_key == "null" else clean_ws(o_text),
                    "VA": va
                })

            # ----- 3) 没有 pair：退化输出一个最可信 aspect（仍走 end 搜索 + asp2cat）
            if len(quads) == 0:
                ac_mat = mat[bi, AC]
                mask_2d = (valid_pos[:, None] & valid_pos[None, :])
                ac_mat_m = ac_mat.masked_fill(~mask_2d, -1e9)
                flat = ac_mat_m.view(-1)
                p = int(torch.argmax(flat).item())
                ai = p // L
                aj = p % L
                if aj < ai:
                    aj = ai
                # 用 ai 作为 start，再用 V3 的 end 搜索（带 prior）
                a_end, a_text, a_span_sc, a_obj, a_key = best_span_end(
                    tok, input_ids, valid_pos, ac_mat_m, ai, args.max_span_len,
                    priors_freq=asp_freq, priors_hit=asp2cat, prior_alpha=0.9, len_penalty=0.10
                )
                if is_bad_span(a_text):
                    a_text = "null"
                    a_key = "null"

                cat = asp2cat.get(a_key, global_cat)
                if cat in cat2va:
                    v, ar = cat2va[cat]
                    va = f"{v:.2f}#{ar:.2f}"
                else:
                    va = "5.00#5.00"

                quads = [{
                    "Aspect": clean_ws(a_text),
                    "Category": cat,
                    "Opinion": "null",
                    "VA": va
                }]

            preds.append({"ID": sid, "Quadruplet": quads})

            # ----- diag
            if args.diag:
                for q in quads:
                    diag_total_quads += 1
                    cat = q.get("Category", "")
                    diag_cat[cat] += 1
                    if cat == global_cat:
                        diag_global += 1
                    a = norm_key(q.get("Aspect", "null"))
                    o = norm_key(q.get("Opinion", "null"))
                    if a == "null":
                        diag_null_a += 1
                    if o == "null":
                        diag_null_o += 1
                    if a in asp2cat:
                        diag_asp_hit += 1
                    if (a, o) in pair2cat:
                        diag_pair_hit += 1

        base += B

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_jsonl(args.output, preds)
    logger.success(f"Saved -> {args.output}")

    if args.diag and diag_total_quads > 0:
        global_rate = diag_global / diag_total_quads
        null_a_rate = diag_null_a / diag_total_quads
        null_o_rate = diag_null_o / diag_total_quads
        logger.info(
            f"[Diag] total_quads={diag_total_quads}  global_rate={global_rate:.3f}  "
            f"null_aspect_rate={null_a_rate:.3f}  null_opinion_rate={null_o_rate:.3f}"
        )
        logger.info(f"[Diag] asp_hit={diag_asp_hit} pair_hit={diag_pair_hit}")
        logger.info(f"[Diag] top10_cats={diag_cat.most_common(10)}")

if __name__ == "__main__":
    main()
