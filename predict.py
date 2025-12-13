# -*- coding: utf-8 -*-
"""
SemEval-2026 Task3 (DimASQP) - Predict Script (V3)
目标：
- 纠正 head 语义：BA-BO/EA-EO/BA-EO-(dim,sent) 是“跨实体边”，不是 span 头
- 显著降低无意义 global fallback，同时避免把 opinion 过度置 null
- VA：优先使用 (Category, dim, sent) 条件统计（dim/sent 来自 BA-EO head 名）

兼容你现有命令行参数风格（thr_aux/topk_aux/max_span_len/thr_rel/topk_rel/max_pair_dist/max_quads/null_thr_o 等）
"""

import os
import re
import json
import math
import argparse
import statistics
from collections import Counter, defaultdict

import torch
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer

from dataset.dataset import AcqpDataset, collate_fn
from models.model import QuadrupleModel
from utils.utils import set_seeds

set_seeds(42)

ALNUM_RE = re.compile(r"[A-Za-z0-9]")

# ----------------------------
# IO
# ----------------------------
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

# ----------------------------
# text helpers
# ----------------------------
def clean_ws(s: str) -> str:
    return " ".join(str(s).strip().split()) if s is not None else ""

def norm_key(s: str) -> str:
    t = clean_ws(s)
    return t.lower() if t else "null"

def is_bad_span(text: str) -> bool:
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
# VA parse + coarse mapping (与 dataset.py 的划分一致)
# sentiment by Valence (V), dimension by Arousal (A)
# ----------------------------
def parse_va(va_str: str, default=(5.0, 5.0)):
    try:
        s = str(va_str).strip()
        if "#" not in s:
            return default
        v, a = s.split("#", 1)
        return float(v), float(a)
    except Exception:
        return default

def va_to_sent(v: float) -> str:
    # dataset.py: V>=6 positive, V<=4 negative else neutral
    if v >= 6.0:
        return "positive"
    if v <= 4.0:
        return "negative"
    return "neutral"

def va_to_dim(a: float) -> str:
    # dataset.py: A>=6 performance, A<=4 usability else quality
    if a >= 6.0:
        return "performance"
    if a <= 4.0:
        return "usability"
    return "quality"

def fmt_va(v: float, a: float) -> str:
    return f"{v:.2f}#{a:.2f}"

# ----------------------------
# Category case
# ----------------------------
def canon_cat(s: str, cat_case: str, default="LAPTOP#GENERAL") -> str:
    t = clean_ws(s)
    if not t:
        t = default
    if cat_case == "upper":
        return t.upper()
    if cat_case == "lower":
        return t.lower()
    return t

# ----------------------------
# Priors
# ----------------------------
def build_priors(train_stats_path, cat_case="upper", va_stat="median"):
    """
    返回：
    - pair2cats: (a_key,o_key)->Counter(cat)
    - asp2cats : a_key->Counter(cat)
    - global_cat
    - pair_freq, asp_freq
    - cat_va_vals: cat -> list[(v,a)]
    - cat_dim_sent_va_vals: (cat,dim,sent) -> list[(v,a)]
    - cat_dim_cnt: cat -> Counter(dim)
    """
    rows = read_jsonl(train_stats_path)

    pair2cats = defaultdict(Counter)
    asp2cats = defaultdict(Counter)
    cat_cnt = Counter()

    pair_freq = Counter()
    asp_freq = Counter()

    cat_va_vals = defaultdict(list)
    cat_dim_sent_va_vals = defaultdict(list)
    cat_dim_cnt = defaultdict(Counter)

    for r in rows:
        qs = r.get("Quadruplet") or r.get("Quadruplets") or r.get("quadruplets") or []
        for q in qs:
            a_raw = q.get("Aspect", "null")
            o_raw = q.get("Opinion", "null")
            c_raw = q.get("Category", "LAPTOP#GENERAL")
            va_raw = q.get("VA", "5.00#5.00")

            a = norm_key(a_raw)
            o = norm_key(o_raw)
            c = canon_cat(c_raw, cat_case=cat_case)

            pair_freq[(a, o)] += 1
            asp_freq[a] += 1

            pair2cats[(a, o)][c] += 1
            asp2cats[a][c] += 1
            cat_cnt[c] += 1

            v, ar = parse_va(va_raw)
            cat_va_vals[c].append((v, ar))

            dim = va_to_dim(ar)
            sent = va_to_sent(v)
            cat_dim_cnt[c][dim] += 1
            cat_dim_sent_va_vals[(c, dim, sent)].append((v, ar))

    global_cat = cat_cnt.most_common(1)[0][0] if len(cat_cnt) else canon_cat("", cat_case, "LAPTOP#GENERAL")

    logger.info(
        f"[Priors] pair2cats={len(pair2cats)} asp2cats={len(asp2cats)} "
        f"global_cat={global_cat} va_stat={va_stat}"
    )

    return pair2cats, asp2cats, global_cat, pair_freq, asp_freq, cat_va_vals, cat_dim_sent_va_vals, cat_dim_cnt

def stat_va(vals, va_stat="median"):
    if not vals:
        return (5.0, 5.0)
    vs = [x[0] for x in vals]
    ars = [x[1] for x in vals]
    if va_stat == "mean":
        return (sum(vs)/len(vs), sum(ars)/len(ars))
    # median default
    return (statistics.median(vs), statistics.median(ars))

def choose_va(cat, dim, sent, cat_va_vals, cat_dim_sent_va_vals, global_cat, va_stat="median"):
    # 优先 (cat,dim,sent)
    key = (cat, dim, sent)
    if key in cat_dim_sent_va_vals and len(cat_dim_sent_va_vals[key]) > 0:
        v, a = stat_va(cat_dim_sent_va_vals[key], va_stat=va_stat)
        return fmt_va(v, a)
    # 退化 cat
    if cat in cat_va_vals and len(cat_va_vals[cat]) > 0:
        v, a = stat_va(cat_va_vals[cat], va_stat=va_stat)
        return fmt_va(v, a)
    # 退化 global
    if global_cat in cat_va_vals and len(cat_va_vals[global_cat]) > 0:
        v, a = stat_va(cat_va_vals[global_cat], va_stat=va_stat)
        return fmt_va(v, a)
    return "5.00#5.00"

def choose_cat(a_key, o_key, dim, pair2cats, asp2cats, cat_dim_cnt, global_cat):
    # candidates from pair -> asp -> global
    if (a_key, o_key) in pair2cats and len(pair2cats[(a_key, o_key)]) > 0:
        cand = pair2cats[(a_key, o_key)]
    elif a_key in asp2cats and len(asp2cats[a_key]) > 0:
        cand = asp2cats[a_key]
    else:
        return global_cat, "global"

    # 如果有 dim，则优先 dim 一致性（count 大），再按原频次
    if dim is not None:
        best_cat = None
        best_tuple = None
        for c, cnt in cand.items():
            dcnt = cat_dim_cnt.get(c, Counter()).get(dim, 0)
            tup = (dcnt, cnt)
            if best_tuple is None or tup > best_tuple:
                best_tuple = tup
                best_cat = c
        if best_cat is not None:
            return best_cat, "prior_dim"

    # 无 dim 直接 mode
    return cand.most_common(1)[0][0], "prior"

# ----------------------------
# Head parsing
# ----------------------------
def parse_rel_head_name(name: str):
    # "BA-EO-{dim}-{sent}"
    # return (dim, sent) or (None, None)
    parts = str(name).split("-")
    if len(parts) >= 4 and parts[0] == "BA" and parts[1] == "EO":
        dim = parts[2]
        sent = parts[3]
        return dim, sent
    return None, None

# ----------------------------
# Candidate extraction for BA-EO
# ----------------------------
def extract_top_aStart_oEnd(scores, valid_a, valid_o, thr, topk, max_pair_dist=None):
    """
    scores: [L,L] after sigmoid, index [a_start, o_end]
    """
    L = scores.shape[0]
    device = scores.device

    va = valid_a.to(device)
    vo = valid_o.to(device)
    mask = va[:, None] & vo[None, :]

    if max_pair_dist is not None and max_pair_dist > 0:
        idx = torch.arange(L, device=device)
        dist = (idx[None, :] - idx[:, None]).abs()   # |o_end - a_start|
        mask = mask & (dist <= max_pair_dist)

    flat = scores.masked_fill(~mask, -1e9).view(-1)
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
        a_start = p // L
        o_end = p % L
        out.append((a_start, o_end, float(vals[k].item())))
    return out

# ----------------------------
# model loader
# ----------------------------
def load_model(ckpt, model_name, num_label_types, num_dims, max_len, device):
    model = QuadrupleModel(num_label_types, num_dims, max_len, model_name).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--train_stats", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)

    # compatibility knobs
    ap.add_argument("--thr_aux", type=float, default=0.05)   # used for BA-BO / EA-EO gate
    ap.add_argument("--topk_aux", type=int, default=80)      # reserved
    ap.add_argument("--max_span_len", type=int, default=12)  # used as max_as_len & max_op_len

    ap.add_argument("--thr_rel", type=float, default=0.15)
    ap.add_argument("--topk_rel", type=int, default=800)
    ap.add_argument("--max_pair_dist", type=int, default=120)

    ap.add_argument("--max_quads", type=int, default=2)

    ap.add_argument("--null_thr_o", type=float, default=0.12)     # BA-BO gate
    ap.add_argument("--null_thr_rel", type=float, default=None)   # BA-EO gate (None->thr_rel)

    ap.add_argument("--va_stat", type=str, default="median", choices=["median", "mean"])
    ap.add_argument("--cat_case", type=str, default="upper", choices=["upper", "lower", "raw"])

    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--output", required=True)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Device] {device}")

    # priors
    (
        pair2cats, asp2cats, global_cat,
        pair_freq, asp_freq,
        cat_va_vals, cat_dim_sent_va_vals, cat_dim_cnt
    ) = build_priors(args.train_stats, cat_case=args.cat_case, va_stat=args.va_stat)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    ds = AcqpDataset("PredictSet", args.input, args.max_len, tok, label_pattern="sentiment_dim")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    # heads
    try:
        h_ba_bo = ds.label_types.index("BA-BO")
        h_ea_eo = ds.label_types.index("EA-EO")
    except ValueError:
        raise RuntimeError(f"label_types missing BA-BO/EA-EO: {ds.label_types}")

    rel_heads = []
    for i, nm in enumerate(ds.label_types):
        if nm in ("BA-BO", "EA-EO"):
            continue
        dim, sent = parse_rel_head_name(nm)
        rel_heads.append((i, dim, sent))

    null_thr_rel = args.null_thr_rel if args.null_thr_rel is not None else args.thr_rel
    max_as_len = max(1, int(args.max_span_len))
    max_op_len = max(1, int(args.max_span_len))

    model = load_model(args.ckpt, args.model_name, len(ds.label_types), len(ds.dimension2id), args.max_len, device)

    preds = []
    base = 0

    # diagnostics
    diag_total = 0
    diag_global = 0
    diag_null_a = 0
    diag_null_o = 0
    diag_asp_hit = 0
    diag_pair_hit = 0
    cat_counter = Counter()

    for batch in tqdm(dl, desc="Predict", ncols=90):
        for k in batch:
            batch[k] = batch[k].to(device)

        with torch.no_grad():
            out = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])
            mat = torch.sigmoid(out["matrix"])   # [B,C,L,L]

        B, _, L, _ = mat.shape

        for bi in range(B):
            idx = base + bi
            # robust id fetch
            try:
                sid = ds.df.iloc[idx]["id"]
            except Exception:
                sid = ds.df[idx].get("id", str(idx))

            input_ids = batch["input_ids"][bi]
            attn = batch["attention_mask"][bi].bool()

            # valid positions: allow CLS(0) + normal tokens, exclude PAD/SEP
            valid = attn.clone()
            sep_id = tok.sep_token_id
            pad_id = tok.pad_token_id
            if sep_id is not None:
                valid = valid & (input_ids != sep_id)
            if pad_id is not None:
                valid = valid & (input_ids != pad_id)
            # ensure CLS allowed if present
            valid[0] = True

            # -------- 1) BA-EO candidates: (a_start, o_end) --------
            cand_list = []
            for (h, dim, sent) in rel_heads:
                pairs = extract_top_aStart_oEnd(
                    mat[bi, h], valid, valid,
                    thr=args.thr_rel,
                    topk=args.topk_rel,
                    max_pair_dist=args.max_pair_dist
                )
                for (a_start, o_end, rel_sc) in pairs:
                    cand_list.append((rel_sc, h, dim, sent, a_start, o_end))

            cand_list.sort(key=lambda x: x[0], reverse=True)

            # -------- 2) decode spans using BA-BO and EA-EO (correct semantics) --------
            quads = []
            used = set()

            for (rel_sc, h, dim, sent, a_start, o_end) in cand_list:
                if len(quads) >= args.max_quads:
                    break

                # aspect implicit
                if a_start == 0:
                    a_end = 0
                    a_text = "null"
                else:
                    # choose a_end using EA-EO[:, o_end] with constraint a_end>=a_start and len<=max_as_len
                    col = mat[bi, h_ea_eo][:, o_end].clone()
                    col = col.masked_fill(~valid, -1e9)

                    lo = a_start
                    hi = min(L - 1, a_start + max_as_len - 1)
                    if lo > hi:
                        lo, hi = a_start, a_start

                    best_sc = -1e9
                    best_end = a_start
                    for ae in range(lo, hi + 1):
                        sc = float(col[ae].item())
                        if sc > best_sc:
                            best_sc = sc
                            best_end = ae
                    a_end = best_end
                    a_text = safe_decode(tok, input_ids, a_start, a_end)
                    if is_bad_span(a_text):
                        a_text = "null"
                        a_start, a_end = 0, 0

                # opinion implicit
                if o_end == 0:
                    o_start = 0
                    o_text = "null"
                else:
                    # choose o_start using BA-BO[a_start, :] if a_start!=0 else choose global best start->o_start
                    row = mat[bi, h_ba_bo][a_start].clone() if a_start != 0 else mat[bi, h_ba_bo][0].clone()
                    row = row.masked_fill(~valid, -1e9)

                    lo = 1
                    hi = min(o_end, L - 1)
                    lo2 = max(lo, hi - max_op_len + 1)
                    if lo2 > hi:
                        lo2 = lo
                    best_sc = -1e9
                    best_st = lo2
                    for os_ in range(lo2, hi + 1):
                        sc = float(row[os_].item())
                        if sc > best_sc:
                            best_sc = sc
                            best_st = os_
                    o_start = best_st

                    o_text = safe_decode(tok, input_ids, o_start, o_end)
                    if is_bad_span(o_text):
                        o_text = "null"
                        o_start, o_end = 0, 0

                a_key = norm_key(a_text)
                o_key = norm_key(o_text)

                # 3) opinion null gate: 只有当 rel 也低 + pair 未见过 + BA-BO 也低 才置空
                pf = pair_freq.get((a_key, o_key), 0)
                bo_sc = float(mat[bi, h_ba_bo][a_start, o_start].item()) if (a_start < L and o_start < L) else 0.0

                if o_text != "null":
                    if (bo_sc < args.null_thr_o) and (rel_sc < null_thr_rel) and (pf == 0):
                        o_text = "null"
                        o_key = "null"

                # 4) category mapping (pair->asp->global) with dim tie-break
                cat, how = choose_cat(a_key, o_key, dim, pair2cats, asp2cats, cat_dim_cnt, global_cat)

                # 5) VA choose (cat,dim,sent) -> cat -> global
                va = choose_va(cat, dim, sent, cat_va_vals, cat_dim_sent_va_vals, global_cat, va_stat=args.va_stat)

                key = (a_key, o_key, cat)
                if key in used:
                    continue
                used.add(key)

                if how == "global":
                    diag_global += 1
                if a_text == "null":
                    diag_null_a += 1
                if o_text == "null":
                    diag_null_o += 1

                if (a_key in asp2cats) and len(asp2cats[a_key]) > 0:
                    diag_asp_hit += 1
                if ((a_key, o_key) in pair2cats) and len(pair2cats[(a_key, o_key)]) > 0:
                    diag_pair_hit += 1

                cat_counter[cat] += 1
                diag_total += 1

                quads.append({
                    "Aspect": clean_ws(a_text),
                    "Category": cat,
                    "Opinion": "null" if o_text == "null" else clean_ws(o_text),
                    "VA": va
                })

            # -------- 6) fallback: no BA-EO candidates -> output one null-safe quad --------
            if len(quads) == 0:
                # pick strongest BA-BO (a_start,o_start), then pick o_end by best BA-EO head under this a_start
                bo = mat[bi, h_ba_bo].clone()
                mask = valid[:, None] & valid[None, :]
                bo = bo.masked_fill(~mask, -1e9)
                p = int(torch.argmax(bo.view(-1)).item())
                a_start = p // L
                o_start = p % L

                # pick o_end using best BA-EO head for this a_start
                best = (-1e9, 0, "quality", "neutral")  # (score, o_end, dim, sent)
                for (h, dim, sent) in rel_heads:
                    row = mat[bi, h][a_start].clone()
                    row = row.masked_fill(~valid, -1e9)
                    o_end = int(torch.argmax(row).item())
                    sc = float(row[o_end].item())
                    if sc > best[0]:
                        best = (sc, o_end, dim, sent)

                rel_sc, o_end, dim, sent = best

                # aspect end from EA-EO[:,o_end]
                if a_start == 0:
                    a_end = 0
                    a_text = "null"
                else:
                    col = mat[bi, h_ea_eo][:, o_end].clone()
                    col = col.masked_fill(~valid, -1e9)
                    lo = a_start
                    hi = min(L - 1, a_start + max_as_len - 1)
                    a_end = int(torch.argmax(col[lo:hi+1]).item() + lo)
                    a_text = safe_decode(tok, input_ids, a_start, a_end)
                    if is_bad_span(a_text):
                        a_text = "null"

                # opinion span
                if o_end == 0 or o_start == 0:
                    o_text = "null"
                else:
                    if o_start > o_end:
                        o_start = o_end
                    o_text = safe_decode(tok, input_ids, o_start, o_end)
                    if is_bad_span(o_text):
                        o_text = "null"

                a_key = norm_key(a_text)
                o_key = norm_key(o_text)

                cat, _ = choose_cat(a_key, o_key, dim, pair2cats, asp2cats, cat_dim_cnt, global_cat)
                va = choose_va(cat, dim, sent, cat_va_vals, cat_dim_sent_va_vals, global_cat, va_stat=args.va_stat)

                quads = [{
                    "Aspect": clean_ws(a_text),
                    "Category": cat,
                    "Opinion": "null" if o_text == "null" else clean_ws(o_text),
                    "VA": va
                }]

                cat_counter[cat] += 1
                diag_total += 1
                if cat == global_cat:
                    diag_global += 1
                if a_text == "null":
                    diag_null_a += 1
                if o_text == "null":
                    diag_null_o += 1

            preds.append({"ID": sid, "Quadruplet": quads})

        base += B

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_jsonl(args.output, preds)
    logger.success(f"Saved -> {args.output}")

    if args.diag:
        total_quads = max(1, diag_total)
        logger.info(
            f"[Diag] total_quads={diag_total}  global_rate={diag_global/total_quads:.3f}  "
            f"null_aspect_rate={diag_null_a/total_quads:.3f}  null_opinion_rate={diag_null_o/total_quads:.3f}"
        )
        logger.info(f"[Diag] asp_hit={diag_asp_hit} pair_hit={diag_pair_hit}")
        logger.info(f"[Diag] top10_cats={cat_counter.most_common(10)}")

if __name__ == "__main__":
    main()
