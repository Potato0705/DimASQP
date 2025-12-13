# -*- coding: utf-8 -*-
"""
DimASQP / SemEval Task3 - Predict (V4, conservative & stable)

目标：在你当前 0.2183 解码策略基础上，进一步降低 FP 并提升 priors 命中率，避免退化为 global fallback。
"""

import os
import json
import re
import math
import argparse
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
SPACE_AROUND_APOS_RE = re.compile(r"\s+'\s*|\s*'\s+")
NT_FIX_RE = re.compile(r"\b(n)\s*'\s*(t)\b", flags=re.IGNORECASE)

STOPWORDS = {
    "a","an","the","this","that","these","those","it","its","i","you","we","they",
    "is","are","was","were","be","been","being","am",
    "and","or","but","if","then","than","so","to","of","in","on","for","with","as","at","by",
    "not","no","yes","do","does","did","done","have","has","had",
}

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

def fix_apostrophes(s: str) -> str:
    t = clean_ws(s)
    if not t:
        return t
    t = t.replace(" ' ", "'")
    t = t.replace(" '", "'").replace("' ", "'")
    t = NT_FIX_RE.sub(r"\1'\2", t)                 # n ' t -> n't
    t = SPACE_AROUND_APOS_RE.sub("'", t)           # 通用收紧
    return clean_ws(t)

def norm_key(s: str, apostrophe_norm: bool = True) -> str:
    t = fix_apostrophes(s) if apostrophe_norm else clean_ws(s)
    t = t.lower()
    return t if t else "null"

def canon_cat(s: str, cat_case: str = "upper") -> str:
    t = clean_ws(s) or "LAPTOP#GENERAL"
    return t.lower() if cat_case == "lower" else t.upper()

def is_bad_span(text: str, apostrophe_norm: bool = True) -> bool:
    if text is None:
        return True
    t = fix_apostrophes(text) if apostrophe_norm else clean_ws(text)
    if len(t) < 2:
        return True
    if not ALNUM_RE.search(t):
        return True
    if len(t.split()) == 1 and t.lower() in STOPWORDS:
        return True
    return False

def safe_decode(tokenizer, input_ids_1d, i, j):
    ids = input_ids_1d[i:j+1].tolist()
    s = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return clean_ws(s)

def parse_va(va_str: str):
    try:
        a, b = va_str.split("#")
        return float(a), float(b)
    except Exception:
        return None

def fmt_va(x: float, y: float) -> str:
    return f"{x:.2f}#{y:.2f}"

def build_priors(train_stats_path, cat_case="upper", va_stat="median", apostrophe_norm=True):
    rows = read_jsonl(train_stats_path)

    cat_cnt = Counter()
    asp_cat_cnt = Counter()
    pair_cat_cnt = Counter()

    asp_freq = Counter()
    op_freq = Counter()
    pair_freq = Counter()

    cat_va_list = defaultdict(list)
    cat_va_mode = defaultdict(Counter)

    for r in rows:
        qs = r.get("Quadruplet") or r.get("Quadruplets") or r.get("quadruplets") or []
        for q in qs:
            a_raw = q.get("Aspect", "null")
            o_raw = q.get("Opinion", "null")
            c_raw = q.get("Category", "LAPTOP#GENERAL")
            va = q.get("VA", "5.00#5.00")

            a = norm_key(a_raw, apostrophe_norm=apostrophe_norm)
            o = norm_key(o_raw, apostrophe_norm=apostrophe_norm)
            c = canon_cat(c_raw, cat_case=cat_case)

            asp_freq[a] += 1
            op_freq[o] += 1
            pair_freq[(a, o)] += 1

            cat_cnt[c] += 1
            asp_cat_cnt[(a, c)] += 1
            pair_cat_cnt[(a, o, c)] += 1

            cat_va_mode[c][va] += 1
            xy = parse_va(va)
            if xy is not None:
                cat_va_list[c].append(xy)

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

    # cat->va
    cat2va = {}
    for c in cat_cnt.keys():
        if va_stat == "mode":
            cat2va[c] = cat_va_mode[c].most_common(1)[0][0] if len(cat_va_mode[c]) else fmt_va(5.0, 5.0)
        else:
            vals = cat_va_list.get(c, [])
            if not vals:
                cat2va[c] = cat_va_mode[c].most_common(1)[0][0] if len(cat_va_mode[c]) else fmt_va(5.0, 5.0)
            else:
                xs = sorted([x for x, _ in vals])
                ys = sorted([y for _, y in vals])
                if va_stat == "mean":
                    cat2va[c] = fmt_va(sum(xs)/len(xs), sum(ys)/len(ys))
                else:
                    mid = len(xs)//2
                    if len(xs) % 2 == 1:
                        cat2va[c] = fmt_va(xs[mid], ys[mid])
                    else:
                        cat2va[c] = fmt_va((xs[mid-1]+xs[mid])/2.0, (ys[mid-1]+ys[mid])/2.0)

    logger.info(f"[Priors] pair2cat={len(pair2cat)} asp2cat={len(asp2cat)} cat2va={len(cat2va)} global_cat={global_cat} va_stat={va_stat}")
    return pair2cat, asp2cat, cat2va, global_cat, asp_freq, op_freq, pair_freq

def topk_mask(scores_1d, valid_pos, k):
    """在 valid_pos 里保留 top-k 位置（k<=0 表示不限制）"""
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
    """scores_LL: [L,L]，pair_mask: [L,L]"""
    L = scores_LL.shape[0]
    device = scores_LL.device

    mask = pair_mask.clone()

    if max_pair_dist and max_pair_dist > 0:
        idx = torch.arange(L, device=device)
        dist = (idx[None, :] - idx[:, None]).abs()
        mask = mask & (dist <= max_pair_dist)

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
    for p, v in zip(idxs.tolist(), vals.tolist()):
        i = int(p // L)
        j = int(p % L)
        out.append((i, j, float(v)))
    return out

def best_span_end(span_mat, valid_end_mask, tokenizer, input_ids, start_i, max_span_len, priors_freq=None, prior_alpha=0.0, apostrophe_norm=True):
    L = span_mat.shape[0]
    j_max = min(L - 1, start_i + max_span_len - 1) if (max_span_len and max_span_len > 0) else (L - 1)

    mask = valid_end_mask.clone()
    mask[:start_i] = False
    if j_max < L - 1:
        mask[(j_max + 1):] = False

    row = span_mat[start_i].masked_fill(~mask, -1e9)
    k = min(16, int(mask.sum().item())) if int(mask.sum().item()) > 0 else 0
    if k == 0:
        return start_i, 0.0, "null"

    topv, topi = torch.topk(row, k=k, largest=True)

    best = (start_i, -1e18, "null")
    for sc, j in zip(topv.tolist(), topi.tolist()):
        j = int(j)
        txt = safe_decode(tokenizer, input_ids, start_i, j)
        if is_bad_span(txt, apostrophe_norm=apostrophe_norm):
            continue
        bonus = 0.0
        if priors_freq is not None and prior_alpha > 0:
            bonus = prior_alpha * math.log(1.0 + float(priors_freq.get(norm_key(txt, apostrophe_norm), 0)))
        sc2 = float(sc) + bonus
        if sc2 > best[1]:
            best = (j, sc2, txt)

    if best[2] == "null":
        j = int(torch.argmax(row).item())
        return j, float(row[j].item()), safe_decode(tokenizer, input_ids, start_i, j)
    return best

def refine_start(span_mat, valid_pos, tokenizer, input_ids, base_i, window, max_span_len, priors_freq=None, prior_alpha=0.0, apostrophe_norm=True):
    L = span_mat.shape[0]
    lo = max(0, base_i - window)
    hi = min(L - 1, base_i + window)
    best = (base_i, base_i, -1e18, "null")  # (i,j,sc,txt)
    for i in range(lo, hi + 1):
        if not bool(valid_pos[i].item()):
            continue
        j, sc, txt = best_span_end(span_mat, valid_pos, tokenizer, input_ids, i, max_span_len, priors_freq, prior_alpha, apostrophe_norm)
        if is_bad_span(txt, apostrophe_norm=apostrophe_norm):
            continue
        if sc > best[2]:
            best = (i, j, sc, txt)
    return best

def load_model(ckpt, model_name, num_label_types, num_dims, max_len, device):
    model = QuadrupleModel(num_label_types, num_dims, max_len, model_name).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train_stats", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)

    ap.add_argument("--thr_aux", type=float, default=0.05)
    ap.add_argument("--topk_aux", type=int, default=80)
    ap.add_argument("--max_span_len", type=int, default=12)

    ap.add_argument("--thr_rel", type=float, default=0.15)
    ap.add_argument("--topk_rel", type=int, default=800)
    ap.add_argument("--max_pair_dist", type=int, default=120)

    ap.add_argument("--max_quads", type=int, default=2)
    ap.add_argument("--null_thr_o", type=float, default=0.12)

    ap.add_argument("--va_stat", type=str, default="median", choices=["mode","median","mean"])
    ap.add_argument("--cat_case", type=str, default="upper", choices=["upper","lower"])

    ap.add_argument("--apostrophe_norm", action="store_true")
    ap.add_argument("--start_refine", type=int, default=2)
    ap.add_argument("--prior_alpha", type=float, default=0.12)

    ap.add_argument("--no_pair2cat_when_op_null", action="store_true")
    ap.add_argument("--dedup_by_aspect", action="store_true")

    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Device] {device}")

    pair2cat, asp2cat, cat2va, global_cat, asp_freq, op_freq, pair_freq = build_priors(
        args.train_stats,
        cat_case=args.cat_case,
        va_stat=args.va_stat,
        apostrophe_norm=args.apostrophe_norm
    )

    tok = AutoTokenizer.from_pretrained(args.model_name)
    ds = AcqpDataset("PredictSet", args.input, args.max_len, tok, label_pattern="sentiment_dim")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    if "BA-BO" not in ds.label_types or "EA-EO" not in ds.label_types:
        raise RuntimeError(f"label_types missing BA-BO/EA-EO: {ds.label_types}")
    AC = ds.label_types.index("BA-BO")
    EO = ds.label_types.index("EA-EO")
    rel_heads = [i for i, nm in enumerate(ds.label_types) if nm not in ["BA-BO", "EA-EO"]]

    model = load_model(args.ckpt, args.model_name, len(ds.label_types), len(ds.dimension2id), args.max_len, device)

    preds = []
    base = 0

    total_quads = 0
    global_cnt = 0
    null_asp_cnt = 0
    null_op_cnt = 0
    asp_hit = 0
    pair_hit = 0
    cat_counter = Counter()

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

            valid_pos = attn.clone()
            specials = torch.tensor([tok.cls_token_id, tok.sep_token_id, tok.pad_token_id], device=input_ids.device)
            is_special = (input_ids[..., None] == specials[None, ...]).any(dim=-1)
            valid_pos = valid_pos & (~is_special)

            mask_2d = (valid_pos[:, None] & valid_pos[None, :])
            ac_mat = mat[bi, AC].masked_fill(~mask_2d, -1e9)
            eo_mat = mat[bi, EO].masked_fill(~mask_2d, -1e9)

            # aux gating：用 BA-BO / EA-EO 的 start 置信度筛 start
            a_start = torch.max(ac_mat, dim=1).values
            o_start = torch.max(eo_mat, dim=1).values

            a_ok = topk_mask(a_start, valid_pos, args.topk_aux) & (a_start >= args.thr_aux)
            o_ok = topk_mask(o_start, valid_pos, args.topk_aux) & (o_start >= args.thr_aux)

            pair_mask = (a_ok[:, None] & o_ok[None, :])

            pair_cands = []
            for h in rel_heads:
                pairs = extract_top_pairs(mat[bi, h], pair_mask, args.thr_rel, args.topk_rel, args.max_pair_dist)
                for (i, j, sc) in pairs:
                    pair_cands.append((sc, h, i, j))
            pair_cands.sort(key=lambda x: x[0], reverse=True)

            cand = []  # (score, quad, used_pair)

            for rel_sc, h, i, j in pair_cands:
                # aspect decode
                if args.start_refine and args.start_refine > 0:
                    ai, aj, a_sc, a_txt = refine_start(ac_mat, valid_pos, tok, input_ids, i, args.start_refine, args.max_span_len, asp_freq, args.prior_alpha, args.apostrophe_norm)
                else:
                    aj, a_sc, a_txt = best_span_end(ac_mat, valid_pos, tok, input_ids, i, args.max_span_len, asp_freq, args.prior_alpha, args.apostrophe_norm)
                    ai = i

                if is_bad_span(a_txt, apostrophe_norm=args.apostrophe_norm):
                    continue

                # opinion decode
                if args.start_refine and args.start_refine > 0:
                    oi, oj, o_sc, o_txt = refine_start(eo_mat, valid_pos, tok, input_ids, j, args.start_refine, args.max_span_len, op_freq, 0.5*args.prior_alpha, args.apostrophe_norm)
                else:
                    oj, o_sc, o_txt = best_span_end(eo_mat, valid_pos, tok, input_ids, j, args.max_span_len, op_freq, 0.5*args.prior_alpha, args.apostrophe_norm)
                    oi = j

                if is_bad_span(o_txt, apostrophe_norm=args.apostrophe_norm):
                    o_txt = "null"

                a_key = norm_key(a_txt, apostrophe_norm=args.apostrophe_norm)
                o_key = norm_key(o_txt, apostrophe_norm=args.apostrophe_norm)

                # opinion null rule：关系弱 + 训练没见过 -> null
                if (o_txt != "null") and (rel_sc < args.null_thr_o) and (pair_freq.get((a_key, o_key), 0) == 0):
                    o_txt = "null"
                    o_key = "null"

                # cat mapping
                used_pair = False
                cat = None
                if not (args.no_pair2cat_when_op_null and o_txt == "null"):
                    cat = pair2cat.get((a_key, o_key))
                    if cat is not None:
                        used_pair = True
                if cat is None:
                    cat = asp2cat.get(a_key)
                if cat is None:
                    cat = global_cat

                va = cat2va.get(cat, fmt_va(5.0, 5.0))

                score = float(rel_sc) + 0.6*float(a_sc) + 0.4*float(o_sc)
                cand.append((score, {
                    "Aspect": fix_apostrophes(a_txt) if args.apostrophe_norm else clean_ws(a_txt),
                    "Category": cat,
                    "Opinion": "null" if o_txt == "null" else (fix_apostrophes(o_txt) if args.apostrophe_norm else clean_ws(o_txt)),
                    "VA": va
                }, used_pair))

            if not cand:
                # fallback
                flat = ac_mat.view(-1)
                p = int(torch.argmax(flat).item())
                ai, aj = p // L, p % L
                if aj < ai:
                    aj = ai
                a_txt = safe_decode(tok, input_ids, ai, aj)
                if is_bad_span(a_txt, apostrophe_norm=args.apostrophe_norm):
                    a_txt = "null"
                a_key = norm_key(a_txt, apostrophe_norm=args.apostrophe_norm)
                cat = asp2cat.get(a_key, global_cat)
                va = cat2va.get(cat, fmt_va(5.0, 5.0))
                cand = [(float(torch.max(ac_mat[ai]).item()), {"Aspect": clean_ws(a_txt), "Category": cat, "Opinion": "null", "VA": va}, False)]

            cand.sort(key=lambda x: x[0], reverse=True)

            quads = []
            seen = set()
            for score, q, used_pair in cand:
                a_key = norm_key(q["Aspect"], apostrophe_norm=args.apostrophe_norm)
                o_key = norm_key(q["Opinion"], apostrophe_norm=args.apostrophe_norm)

                dedup_key = a_key if args.dedup_by_aspect else (a_key, q["Category"])
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

if __name__ == "__main__":
    main()
