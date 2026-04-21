"""ISR (Implicit Sentiment Reasoning) — Recovery variant.

Reads gold training data, identifies quadruplets with NULL aspect and/or opinion,
calls an LLM to recover surrogate spans, and writes a new training file where
NULL spans are replaced with the recovered surrogates.

The key insight: 34.9% of gold training quads have NULL aspect or opinion.
These map to [SEP] position during training, polluting the encoder's [SEP]
representation with mixed implicit signals. By recovering explicit surrogate
spans, we give the model actual token positions to attend to.

Pipeline:
    1. Load gold JSONL
    2. For each quad with NULL aspect / opinion / both:
       - Call LLM with ISR recovery prompt
       - Validate that returned surrogate is an exact substring
       - Replace NULL with surrogate span indices
    3. Write augmented .txt + .jsonl + _sidecar.json + _stats.json

Usage:
    python data/isr_generator.py \
        --task_domain eng_restaurant \
        --gold_jsonl data/v2/eng/eng_restaurant_train.jsonl \
        --gold_txt data/v2/eng/eng_restaurant_train.txt \
        --out_prefix data/v2/eng/eng_restaurant_train_isr__llama31-70b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from llm import (  # noqa: E402
    DEFAULTS,
    OpenRouterClient,
    OpenRouterError,
    PROMPT_VERSION,
    build_isr_recovery_prompt,
    build_isr_both_null_prompt,
)
from llm.config import ISR_DEFAULTS  # noqa: E402
from llm.openrouter_client import parse_json_content  # noqa: E402


def _load_jsonl(path: str) -> List[Dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                entries.append(json.loads(ln))
    return entries


def _load_txt_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def _find_span(text: str, span_text: str) -> Tuple[int, int]:
    if not span_text or span_text == "NULL":
        return -1, -1
    idx = text.find(span_text)
    if idx != -1:
        return idx, idx + len(span_text)
    idx = text.lower().find(span_text.lower())
    if idx != -1:
        return idx, idx + len(span_text)
    norm_text = re.sub(r"\s+", " ", text)
    norm_span = re.sub(r"\s+", " ", span_text)
    idx = norm_text.find(norm_span)
    if idx != -1:
        return idx, idx + len(norm_span)
    return -1, -1


def _cache_key(prefix: str, model: str, content_hash: str) -> str:
    h = hashlib.sha256()
    h.update(PROMPT_VERSION.encode("utf-8"))
    h.update(b"\x00")
    h.update(prefix.encode("utf-8"))
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(content_hash.encode("utf-8"))
    return h.hexdigest()


def _cache_load(cache_dir: str, key: str) -> Optional[Dict]:
    path = os.path.join(cache_dir, key[:2], key + ".json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _cache_save(cache_dir: str, key: str, payload: Dict) -> None:
    path = os.path.join(cache_dir, key[:2], key + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def _llm_call(
    client: Optional[OpenRouterClient],
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    cache_dir: str,
    cache_prefix: str,
    content_hash: str,
    dry_run: bool,
    stats: Dict[str, Any],
    verbose: bool,
) -> Optional[str]:
    key = _cache_key(cache_prefix, model, content_hash)
    cached = _cache_load(cache_dir, key)
    if cached is not None:
        stats["n_cache_hits"] += 1
        return cached.get("content", "")
    if dry_run:
        return None
    if client is None:
        return None
    try:
        resp = client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except OpenRouterError as e:
        stats["n_http_fail"] += 1
        if verbose:
            print(f"  [http_fail] {e}")
        return None
    content = resp.get("content", "") or ""
    usage = resp.get("usage") or {}
    stats["n_llm_calls"] += 1
    stats["tokens_prompt"] += int(usage.get("prompt_tokens") or 0)
    stats["tokens_completion"] += int(usage.get("completion_tokens") or 0)
    stats["tokens_total"] += int(usage.get("total_tokens") or 0)
    _cache_save(cache_dir, key, {
        "content": content, "usage": usage,
        "model_served": resp.get("model"),
    })
    return content


def _recover_single_null(
    client: Optional[OpenRouterClient],
    sentence: str,
    category: str,
    slot_type: str,
    other_slot_type: str,
    other_slot_value: str,
    valence: float,
    arousal: float,
    model: str,
    cache_dir: str,
    dry_run: bool,
    stats: Dict[str, Any],
    verbose: bool,
) -> Optional[str]:
    """Call LLM to recover a surrogate span for a single NULL slot."""
    msgs = build_isr_recovery_prompt(
        sentence=sentence,
        category=category,
        slot_type=slot_type,
        other_slot_type=other_slot_type,
        other_slot_value=other_slot_value,
        valence=valence,
        arousal=arousal,
    )
    content_hash = hashlib.sha256(
        f"{sentence}:{category}:{slot_type}:{other_slot_value}:{valence}:{arousal}".encode()
    ).hexdigest()
    raw = _llm_call(
        client, msgs, model,
        ISR_DEFAULTS["temperature"],
        ISR_DEFAULTS["max_tokens"],
        cache_dir, "isr_recovery", content_hash,
        dry_run, stats, verbose,
    )
    if raw is None:
        return None
    parsed = parse_json_content(raw)
    if parsed is None:
        stats["n_parse_fail"] += 1
        return None
    surrogate = parsed.get("surrogate", "NULL")
    if not surrogate or surrogate == "NULL":
        return None
    s0, _ = _find_span(sentence, surrogate)
    if s0 == -1:
        stats["n_span_miss"] += 1
        if verbose:
            print(f"  [span_miss] surrogate={surrogate!r} not in sentence")
        return None
    return surrogate


def _recover_both_null(
    client: Optional[OpenRouterClient],
    sentence: str,
    category: str,
    valence: float,
    arousal: float,
    model: str,
    cache_dir: str,
    dry_run: bool,
    stats: Dict[str, Any],
    verbose: bool,
) -> Tuple[Optional[str], Optional[str]]:
    """Call LLM to recover surrogate spans when both aspect and opinion are NULL."""
    msgs = build_isr_both_null_prompt(
        sentence=sentence,
        category=category,
        valence=valence,
        arousal=arousal,
    )
    content_hash = hashlib.sha256(
        f"{sentence}:{category}:both:{valence}:{arousal}".encode()
    ).hexdigest()
    raw = _llm_call(
        client, msgs, model,
        ISR_DEFAULTS["temperature"],
        ISR_DEFAULTS["max_tokens"],
        cache_dir, "isr_both_null", content_hash,
        dry_run, stats, verbose,
    )
    if raw is None:
        return None, None
    parsed = parse_json_content(raw)
    if parsed is None:
        stats["n_parse_fail"] += 1
        return None, None

    asp_surr = parsed.get("aspect_surrogate", "NULL")
    opi_surr = parsed.get("opinion_surrogate", "NULL")

    if asp_surr and asp_surr != "NULL":
        s0, _ = _find_span(sentence, asp_surr)
        if s0 == -1:
            stats["n_span_miss"] += 1
            asp_surr = None
    else:
        asp_surr = None

    if opi_surr and opi_surr != "NULL":
        s0, _ = _find_span(sentence, opi_surr)
        if s0 == -1:
            stats["n_span_miss"] += 1
            opi_surr = None
    else:
        opi_surr = None

    return asp_surr, opi_surr


def main():
    parser = argparse.ArgumentParser(
        description="ISR-Recovery: recover surrogate spans for NULL aspect/opinion in gold data."
    )
    parser.add_argument("--task_domain", required=True)
    parser.add_argument("--gold_jsonl", default=None)
    parser.add_argument("--gold_txt", default=None)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--llm_model_name", default=DEFAULTS["model"])
    parser.add_argument("--cache_dir", default=ISR_DEFAULTS["cache_dir"])
    parser.add_argument("--nrows", type=int, default=None,
                        help="Limit to first N gold rows (for smoke tests).")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    lang = args.task_domain.split("_", 1)[0]
    gold_jsonl = args.gold_jsonl or os.path.join(
        _REPO_ROOT, "data", "v2", lang, f"{args.task_domain}_train.jsonl"
    )
    gold_txt = args.gold_txt or os.path.join(
        _REPO_ROOT, "data", "v2", lang, f"{args.task_domain}_train.txt"
    )

    entries = _load_jsonl(gold_jsonl)
    txt_lines = _load_txt_lines(gold_txt)

    if len(entries) != len(txt_lines):
        parser.error(f"JSONL ({len(entries)}) and TXT ({len(txt_lines)}) row count mismatch")

    if args.nrows is not None:
        entries = entries[:args.nrows]
        txt_lines = txt_lines[:args.nrows]

    stats: Dict[str, Any] = {
        "task_domain": args.task_domain,
        "model": args.llm_model_name,
        "prompt_version": PROMPT_VERSION,
        "pipeline": "ISR-Recovery",
        "n_gold_rows": len(entries),
        "n_total_quads": 0,
        "n_null_aspect": 0,
        "n_null_opinion": 0,
        "n_both_null": 0,
        "n_recovered_aspect": 0,
        "n_recovered_opinion": 0,
        "n_parse_fail": 0,
        "n_span_miss": 0,
        "n_http_fail": 0,
        "n_cache_hits": 0,
        "n_llm_calls": 0,
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "tokens_total": 0,
        "wallclock_s": 0.0,
    }

    client: Optional[OpenRouterClient] = None
    if not args.dry_run:
        try:
            client = OpenRouterClient()
        except OpenRouterError as e:
            parser.error(str(e))

    print(f"[ISR] task={args.task_domain}  model={args.llm_model_name}  "
          f"gold_rows={len(entries)}")

    out_txt: List[str] = []
    out_jsonl: List[Dict] = []
    out_sidecar: List[Dict] = []

    t0 = time.time()

    for row_idx, (entry, orig_txt_line) in enumerate(zip(entries, txt_lines)):
        sentence = entry["Text"]
        quads = entry.get("Quadruplet", [])
        new_quads_txt = []
        new_quads_jsonl = []
        sidecar_quads = []

        for q in quads:
            stats["n_total_quads"] += 1
            aspect = q.get("Aspect", "NULL")
            opinion = q.get("Opinion", "NULL")
            category = q.get("Category", "")
            va_str = q.get("VA", "5.0#5.0")
            try:
                v_str, a_str = va_str.split("#")
                valence, arousal = float(v_str), float(a_str)
            except (ValueError, AttributeError):
                valence, arousal = 5.0, 5.0

            asp_is_null = (aspect == "NULL")
            opi_is_null = (opinion == "NULL")

            recovered_asp = aspect
            recovered_opi = opinion

            if asp_is_null and opi_is_null:
                stats["n_both_null"] += 1
                asp_surr, opi_surr = _recover_both_null(
                    client, sentence, category, valence, arousal,
                    args.llm_model_name, args.cache_dir,
                    args.dry_run, stats, args.verbose,
                )
                if asp_surr:
                    recovered_asp = asp_surr
                    stats["n_recovered_aspect"] += 1
                if opi_surr:
                    recovered_opi = opi_surr
                    stats["n_recovered_opinion"] += 1
            elif asp_is_null:
                stats["n_null_aspect"] += 1
                surr = _recover_single_null(
                    client, sentence, category,
                    "aspect", "opinion", opinion,
                    valence, arousal,
                    args.llm_model_name, args.cache_dir,
                    args.dry_run, stats, args.verbose,
                )
                if surr:
                    recovered_asp = surr
                    stats["n_recovered_aspect"] += 1
            elif opi_is_null:
                stats["n_null_opinion"] += 1
                surr = _recover_single_null(
                    client, sentence, category,
                    "opinion", "aspect", aspect,
                    valence, arousal,
                    args.llm_model_name, args.cache_dir,
                    args.dry_run, stats, args.verbose,
                )
                if surr:
                    recovered_opi = surr
                    stats["n_recovered_opinion"] += 1

            a0, a1 = _find_span(sentence, recovered_asp)
            o0, o1 = _find_span(sentence, recovered_opi)

            new_quads_txt.append([
                category,
                f"{a0},{a1}",
                f"{o0},{o1}",
                f"{valence:.2f}#{arousal:.2f}",
            ])
            new_quads_jsonl.append({
                "Aspect": recovered_asp if a0 != -1 else "NULL",
                "Opinion": recovered_opi if o0 != -1 else "NULL",
                "Category": category,
                "VA": f"{valence:.2f}#{arousal:.2f}",
            })
            sidecar_quads.append({
                "orig_aspect": aspect,
                "orig_opinion": opinion,
                "recovered_aspect": recovered_asp if a0 != -1 else "NULL",
                "recovered_opinion": recovered_opi if o0 != -1 else "NULL",
                "category": category,
                "VA": f"{valence:.2f}#{arousal:.2f}",
                "aspect_span": [a0, a1],
                "opinion_span": [o0, o1],
            })

        out_txt.append(f"{sentence}####{json.dumps(new_quads_txt, ensure_ascii=False)}")
        out_jsonl.append({
            "ID": entry.get("ID", f"isr_{row_idx:04d}"),
            "Text": sentence,
            "Quadruplet": new_quads_jsonl,
        })
        out_sidecar.append({
            "ID": entry.get("ID", f"isr_{row_idx:04d}"),
            "Text": sentence,
            "line_index": row_idx,
            "quads": sidecar_quads,
        })

        if args.verbose and (row_idx + 1) % 100 == 0:
            print(f"  [{row_idx+1}/{len(entries)}] recovered_asp={stats['n_recovered_aspect']}  "
                  f"recovered_opi={stats['n_recovered_opinion']}")

    stats["wallclock_s"] = round(time.time() - t0, 2)

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_prefix + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out_txt) + ("\n" if out_txt else ""))
    with open(args.out_prefix + ".jsonl", "w", encoding="utf-8") as f:
        for g in out_jsonl:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    with open(args.out_prefix + "_sidecar.json", "w", encoding="utf-8") as f:
        json.dump(out_sidecar, f, ensure_ascii=False, indent=2)
    with open(args.out_prefix + "_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    n_null_total = stats["n_null_aspect"] + stats["n_null_opinion"] + stats["n_both_null"]
    n_recovered = stats["n_recovered_aspect"] + stats["n_recovered_opinion"]
    rate = n_recovered / max(1, n_null_total + stats["n_both_null"]) * 100
    print(f"\n[ISR] Done. quads={stats['n_total_quads']}  "
          f"null_asp={stats['n_null_aspect']}  null_opi={stats['n_null_opinion']}  "
          f"both_null={stats['n_both_null']}")
    print(f"  recovered: asp={stats['n_recovered_aspect']}  opi={stats['n_recovered_opinion']}  "
          f"rate={rate:.1f}%")
    print(f"  parse_fail={stats['n_parse_fail']}  span_miss={stats['n_span_miss']}  "
          f"http_fail={stats['n_http_fail']}")
    print(f"  cache_hits={stats['n_cache_hits']}  llm_calls={stats['n_llm_calls']}  "
          f"tokens={stats['tokens_total']}")
    print(f"  wallclock={stats['wallclock_s']}s")
    print(f"  TXT     : {args.out_prefix}.txt")
    print(f"  JSONL   : {args.out_prefix}.jsonl")
    print(f"  sidecar : {args.out_prefix}_sidecar.json")
    print(f"  stats   : {args.out_prefix}_stats.json")


if __name__ == "__main__":
    main()
