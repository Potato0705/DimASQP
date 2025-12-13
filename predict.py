# -*- coding: utf-8 -*-
"""
SemEval-2026 Task3 (DimASQP) - Predict Script (V5)
核心修复（针对你现在 global_rate=1.0 的现象）：
1) Span rescue：BA-BO / EA-EO 在阈值下无候选时，强制取 masked top1，避免大量 Aspect='null'
2) Category backoff：除了 pair2cat / asp2cat 精确匹配，增加 token->category 投票映射（train_stats 构建）
3) Span 轻量长度惩罚 + 去重：减少 "wireless signals weakly" 这类把 opinion 吃进 aspect 的情况
4) 输出 Category 大小写可控（--cat_case upper/lower），默认 upper（内部 priors 用 upper）
"""

import os
import json
import re
import math
import argparse
import collections
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer

from dataset.dataset import AcqpDataset, collate_fn
from models.model import QuadrupleModel
from utils.utils import set_seeds

set_seeds(42)

ALNUM_RE = re.compile(r"[A-Za-z0-9]")
WORD_RE  = re.compile(r"[A-Za-z0-9]+")


# -----------------------------
# IO
# -----------------------------
def read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fw:
        for x in rows:
            fw.write(json.dumps(x, ensure_ascii=False) + "\n")


# -----------------------------
# text utils
# -----------------------------
def clean_ws(s: str) -> str:
    return " ".join(str(s).strip().split()) if s is not None else ""


def trim_edge_punct(s: str) -> str:
    """只裁首尾无用符号，尽量不破坏实体词"""
    if s is None:
        return ""
    t = clean_ws(s)
    if not t:
        return t
    i = 0
    while i < len(t) and (not t[i].isalnum()) and t[i] not in ["#"]:
        i += 1
    j = len(t) - 1
    while j >= 0 and (not t[j].isalnum()) and t[j] not in ["#"]:
        j -= 1
    if i <= j:
        t = t[i : j + 1]
    return clean_ws(t)


def norm_key(s: str) -> str:
    t = trim_edge_punct(s)
    return t.lower() if t else "null"


def canon_cat(s: str) -> str:
    t = clean_ws(s)
    return t.upper() if t else "LAPTOP#GENERAL"


def cat_out(cat_upper: str, cat_case: str) -> str:
    if cat_case == "lower":
        return cat_upper.lower()
    return cat_upper


def is_bad_span(text: str) -> bool:
    if text is None:
        return True
    t = trim_edge_punct(text)
    if len(t) < 2:
        return True
    if not ALNUM_RE.search(t):
        return True
    return False


def safe_decode(tokenizer, input_ids_1d, i: int, j: int) -> str:
    ids = input_ids_1d[i : j + 1].tolist()
    s = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return trim_edge_punct(s)


def to_tokens_lower(s: str) -> List[str]:
    if not s:
        return []
    return [w.lower() for w in WORD_RE.findall(s)]


# -----------------------------
# priors
# -----------------------------
def build_priors(train_stats_path: str):
    """
    返回：
    - pair2cat: (a_key,o_key)->CAT_UPPER (mode)
    - asp2cat : a_key->CAT_UPPER (mode)
    - cat2va_mode: CAT_UPPER->va_str (mode)
    - global_cat: CAT_UPPER
    - asp_freq: a_key->count
    - pair_freq: (a_key,o_key)->count
    - token2cat: token(lower)->Counter(CAT_UPPER)
    """
    rows = read_jsonl(train_stats_path)

    cat_cnt = {}
    asp_cat_cnt = {}
    pair_cat_cnt = {}
    asp_freq = {}
    pair_freq = {}
    cat_va_cnt: Dict[str, Dict[str, int]] = {}
    token2cat: Dict[str, Dict[str, int]] = {}

    for r in rows:
        qs = r.get("Quadruplet") or r.get("Quadruplets") or r.get("quadruplets") or []
        for q in qs:
            a_raw = q.get("Aspect", "null")
            o_raw = q.get("Opinion", "null")
            c_raw = q.get("Category", "LAPTOP#GENERAL")
            va = q.get("VA", "5.00#5.00")

            a = norm_key(a_raw)
            o = norm_key(o_raw)
            c = canon_cat(c_raw)

            asp_freq[a] = asp_freq.get(a, 0) + 1
            pair_freq[(a, o)] = pair_freq.get((a, o), 0) + 1

            cat_cnt[c] = cat_cnt.get(c, 0) + 1
            asp_cat_cnt[(a, c)] = asp_cat_cnt.get((a, c), 0) + 1
            pair_cat_cnt[(a, o, c)] = pair_cat_cnt.get((a, o, c), 0) + 1

            if c not in cat_va_cnt:
                cat_va_cnt[c] = {}
            cat_va_cnt[c][va] = cat_va_cnt[c].get(va, 0) + 1

            # token->cat votes (主要从 aspect tokens，必要时也加一点 opinion tokens)
            for tk in to_tokens_lower(a_raw):
                token2cat.setdefault(tk, {})
                token2cat[tk][c] = token2cat[tk].get(c, 0) + 1
            for tk in to_tokens_lower(o_raw):
                token2cat.setdefault(tk, {})
                token2cat[tk][c] = token2cat[tk].get(c, 0) + 1

    global_cat = max(cat_cnt.items(), key=lambda x: x[1])[0] if cat_cnt else "LAPTOP#GENERAL"

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

    # cat->va mode
    cat2va_mode = {}
    for c, d in cat_va_cnt.items():
        va_mode = max(d.items(), key=lambda x: x[1])[0]
        cat2va_mode[c] = va_mode

    logger.info(
        f"[Priors] pair2cat={len(pair2cat)} asp2cat={len(asp2cat)} token2cat={len(token2cat)} cat2va={len(cat2va_mode)} global_cat={global_cat}"
    )
    return pair2cat, asp2cat, cat2va_mode, global_cat, asp_freq, pair_freq, token2cat


def guess_cat_by_tokens(a_text: str, o_text: str, token2cat: Dict[str, Dict[str, int]], min_votes: int = 3):
    """
    词典投票：把 aspect/opinion 的 token 映射到类别计数，选最高票
    min_votes：过低会引入噪声，建议 3~6
    """
    votes = collections.Counter()
    toks = to_tokens_lower(a_text) + to_tokens_lower(o_text)
    for tk in toks:
        if tk in token2cat:
            votes.update(token2cat[tk])
    if not votes:
        return None
    cat, v = votes.most_common(1)[0]
    if v < min_votes:
        return None
    return cat


# -----------------------------
# span extraction (with rescue)
# -----------------------------
def extract_spans(scores_LL: torch.Tensor,
                  valid_pos: torch.Tensor,
                  thr: float,
                  topk: int,
                  max_span_len: int,
                  rescue_top1: bool = True):
    """
    scores_LL: [L,L] sigmoid 后
    return (i,j,score) desc; if none >=thr and rescue_top1=True -> return masked top1
    """
    L = scores_LL.shape[0]
    device = scores_LL.device

    vp = valid_pos.to(device)
    mask = vp[:, None].expand(L, L) & vp[None, :].expand(L, L)

    idx = torch.arange(L, device=device)
    ii = idx[:, None].expand(L, L)
    jj = idx[None, :].expand(L, L)

    mask = mask & (jj >= ii)
    if max_span_len and max_span_len > 0:
        mask = mask & ((jj - ii) < max_span_len)

    flat = scores_LL.masked_fill(~mask, -1e9).view(-1)
    cand = (flat >= thr)

    if not cand.any():
        if not rescue_top1:
            return []
        bestv, besti = torch.max(flat, dim=0)
        if bestv.item() <= -1e8:
            return []
        idxs = besti.view(1)
        vals = bestv.view(1)
    else:
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


def dedup_terms(terms: List[dict], key_name: str = "key"):
    """
    同 key 去重：保留 score 更高的；如分数相近则保留更短的 span（更像纯 aspect/opinion）
    """
    best = {}
    for t in terms:
        k = t[key_name]
        if k not in best:
            best[k] = t
        else:
            # compare by score, then by length
            if t["score"] > best[k]["score"] + 1e-6:
                best[k] = t
            elif abs(t["score"] - best[k]["score"]) <= 1e-6:
                if t["len_w"] < best[k]["len_w"]:
                    best[k] = t
    return list(best.values())


def load_model(ckpt: str, model_name: str, num_label_types: int, num_dims: int, max_len: int, device: str):
    model = QuadrupleModel(num_label_types, num_dims, max_len, model_name).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# -----------------------------
# main
# -----------------------------
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
    ap.add_argument("--max_span_len", type=int, default=16)

    ap.add_argument("--thr_rel", type=float, default=0.55)
    ap.add_argument("--topk_rel", type=int, default=800)
    ap.add_argument("--max_pair_dist", type=int, default=160)

    ap.add_argument("--max_quads", type=int, default=4)
    ap.add_argument("--null_thr_o", type=float, default=0.12)

    # scoring weights
    ap.add_argument("--w_rel", type=float, default=1.0)
    ap.add_argument("--w_a", type=float, default=0.25)
    ap.add_argument("--w_o", type=float, default=0.25)
    ap.add_argument("--w_dist", type=float, default=0.02)
    ap.add_argument("--w_pf", type=float, default=0.80)
    ap.add_argument("--w_af", type=float, default=0.35)
    ap.add_argument("--len_pen", type=float, default=0.03)  # 每多1个词扣一点，鼓励短 span

    # token vote backoff
    ap.add_argument("--token_min_votes", type=int, default=3)
    ap.add_argument("--token_override_global", action="store_true")  # 若 cat=global 且 token vote 有更强建议，则覆盖

    ap.add_argument("--cat_case", choices=["upper", "lower"], default="upper")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Device] {device}")

    pair2cat, asp2cat, cat2va_mode, global_cat, asp_freq, pair_freq, token2cat = build_priors(args.train_stats)
    default_va = cat2va_mode.get(global_cat, "5.00#5.00")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    ds = AcqpDataset("PredictSet", args.input, args.max_len, tok, label_pattern="sentiment_dim")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    H_ASP = ds.label_types.index("BA-BO")
    H_OPN = ds.label_types.index("EA-EO")
    rel_heads = [i for i, nm in enumerate(ds.label_types) if nm not in ["BA-BO", "EA-EO"]]

    model = load_model(args.ckpt, args.model_name, len(ds.label_types), len(ds.dimension2id), args.max_len, device)

    # diag
    total_quads = 0
    global_cnt = 0
    null_a_cnt = 0
    null_o_cnt = 0
    asp_hit = 0
    pair_hit = 0
    token_used = 0
    cat_counter = collections.Counter()

    preds = []
    base = 0

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

            # 1) extract spans with rescue
            a_spans = extract_spans(mat[bi, H_ASP], valid_pos, args.thr_aux, args.topk_aux, args.max_span_len, rescue_top1=True)
            o_spans = extract_spans(mat[bi, H_OPN], valid_pos, args.thr_aux, args.topk_aux, args.max_span_len, rescue_top1=True)

            aspects = []
            for (ai, aj, sc) in a_spans:
                t = safe_decode(tok, input_ids, ai, aj)
                if is_bad_span(t):
                    continue
                len_w = max(1, len(t.split()))
                score = sc - args.len_pen * (len_w - 1)
                akey = norm_key(t)
                score = score + args.w_af * math.log(asp_freq.get(akey, 0) + 1.0)
                aspects.append({"bi": ai, "bj": aj, "raw_sc": sc, "score": score, "text": t, "key": akey, "len_w": len_w})

            opinions = []
            for (oi, oj, sc) in o_spans:
                t = safe_decode(tok, input_ids, oi, oj)
                if is_bad_span(t):
                    continue
                len_w = max(1, len(t.split()))
                score = sc - args.len_pen * (len_w - 1)
                okey = norm_key(t)
                opinions.append({"ei": oi, "ej": oj, "raw_sc": sc, "score": score, "text": t, "key": okey, "len_w": len_w})

            aspects = dedup_terms(aspects, "key")
            opinions = dedup_terms(opinions, "key")

            aspects.sort(key=lambda x: x["score"], reverse=True)
            opinions.sort(key=lambda x: x["score"], reverse=True)

            # 永远加入一个 null opinion 候选，便于出“有类别的 null”
            opinions.append({"ei": -1, "ej": -1, "raw_sc": 0.0, "score": 0.0, "text": "null", "key": "null", "len_w": 1})

            # 如果 aspect 真的一个都没抽出来（极少），做硬兜底
            if len(aspects) == 0:
                quads = [{"Aspect": "null", "Category": cat_out(global_cat, args.cat_case), "Opinion": "null", "VA": default_va}]
                preds.append({"ID": sid, "Quadruplet": quads})
                total_quads += 1
                global_cnt += 1
                null_a_cnt += 1
                null_o_cnt += 1
                cat_counter[global_cat] += 1
                continue

            # 截断候选数量，控复杂度
            aspects = aspects[: min(len(aspects), max(20, args.max_quads * 8))]
            opinions_real = [o for o in opinions if o["key"] != "null"][: min(40, args.max_quads * 12)]
            has_real_op = len(opinions_real) > 0

            pair_scores = []

            # 2) relation pairing（只对 real opinions）
            if has_real_op:
                ba = torch.tensor([a["bi"] for a in aspects], device=device, dtype=torch.long)
                eo = torch.tensor([o["ej"] for o in opinions_real], device=device, dtype=torch.long)

                rel_stack = mat[bi, rel_heads]  # [H,L,L]
                rel_vals = rel_stack[:, ba[:, None], eo[None, :]]  # [H,na,no]
                rel_max, rel_arg = rel_vals.max(dim=0)             # [na,no]

                if args.max_pair_dist and args.max_pair_dist > 0:
                    dist = (ba[:, None] - eo[None, :]).abs()
                    rel_max = rel_max.masked_fill(dist >= args.max_pair_dist, -1e9)

                a_sc = torch.tensor([a["score"] for a in aspects], device=device).view(-1, 1)
                o_sc = torch.tensor([o["score"] for o in opinions_real], device=device).view(1, -1)
                distf = (ba[:, None] - eo[None, :]).abs().float()

                comb = args.w_rel * rel_max + args.w_a * a_sc + args.w_o * o_sc - args.w_dist * distf
                comb = comb.masked_fill(rel_max < args.thr_rel, -1e9)

                flat = comb.view(-1)
                if (flat > -1e8).any():
                    k = min(args.topk_rel, flat.numel())
                    topv, topi = torch.topk(flat, k=k, largest=True)
                    for vv, pp in zip(topv.tolist(), topi.tolist()):
                        if vv <= -1e8:
                            break
                        ia = pp // len(opinions_real)
                        io = pp % len(opinions_real)
                        r = float(rel_max[ia, io].item())
                        pair_scores.append((float(vv), ia, io, r))

            # 3) add null-opinion pairs（保证有输出，且能用 asp2cat/token2cat 决类目）
            null_idx = len(opinions_real)  # 虚拟索引
            for ia, a in enumerate(aspects):
                # null pair score：主要看 aspect
                vv = a["score"]
                pair_scores.append((vv, ia, null_idx, 0.0))

            pair_scores.sort(key=lambda x: x[0], reverse=True)

            used = set()
            quads = []

            for (vv, ia, io, relv) in pair_scores:
                if len(quads) >= args.max_quads:
                    break

                a = aspects[ia]
                a_text, a_key = a["text"], a["key"]

                if io == null_idx:
                    o_text, o_key, o_raw_sc = "null", "null", 0.0
                else:
                    o = opinions_real[io]
                    o_text, o_key, o_raw_sc = o["text"], o["key"], o["raw_sc"]

                # opinion null gate（只对 real opinion）
                if o_key != "null":
                    pf = pair_freq.get((a_key, o_key), 0)
                    if (o_raw_sc < args.null_thr_o) and (pf == 0):
                        o_text, o_key = "null", "null"

                # 4) category mapping：pair->asp->token_vote->global
                cat = None
                if o_key != "null":
                    cat = pair2cat.get((a_key, o_key))
                    if cat is not None:
                        pair_hit += 1

                if cat is None:
                    cat = asp2cat.get(a_key)
                    if cat is not None:
                        asp_hit += 1

                if cat is None:
                    g = guess_cat_by_tokens(a_text, "" if o_key == "null" else o_text, token2cat, min_votes=args.token_min_votes)
                    if g is not None:
                        cat = g
                        token_used += 1

                if cat is None:
                    cat = global_cat

                # 可选：若最终仍是 global，但 token vote 很确定，则覆盖
                if args.token_override_global and cat == global_cat:
                    g = guess_cat_by_tokens(a_text, "" if o_key == "null" else o_text, token2cat, min_votes=max(args.token_min_votes, 5))
                    if g is not None and g != global_cat:
                        cat = g
                        token_used += 1

                va = cat2va_mode.get(cat, default_va)

                key = (a_key, o_key, cat)
                if key in used:
                    continue
                used.add(key)

                quads.append({
                    "Aspect": clean_ws(a_text) if a_key != "null" else "null",
                    "Category": cat_out(cat, args.cat_case),
                    "Opinion": "null" if o_key == "null" else clean_ws(o_text),
                    "VA": va
                })

            if len(quads) == 0:
                quads = [{"Aspect": "null", "Category": cat_out(global_cat, args.cat_case), "Opinion": "null", "VA": default_va}]

            preds.append({"ID": sid, "Quadruplet": quads})

            # diag update
            for q in quads:
                total_quads += 1
                cat_u = q["Category"].upper() if args.cat_case == "lower" else q["Category"]
                cat_counter[cat_u] += 1
                if cat_u == global_cat:
                    global_cnt += 1
                if q["Aspect"] == "null":
                    null_a_cnt += 1
                if q["Opinion"] == "null":
                    null_o_cnt += 1

        base += B

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_jsonl(args.output, preds)

    logger.success(f"Saved -> {args.output}")
    if total_quads > 0:
        logger.info(
            f"[Diag] total_quads={total_quads}  global_rate={global_cnt/total_quads:.3f}  null_aspect_rate={null_a_cnt/total_quads:.3f}  null_opinion_rate={null_o_cnt/total_quads:.3f}"
        )
        logger.info(f"[Diag] asp_hit={asp_hit} pair_hit={pair_hit} token_used={token_used}")
        logger.info(f"[Diag] top10_cats={cat_counter.most_common(10)}")


if __name__ == "__main__":
    main()
