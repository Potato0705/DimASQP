"""Offline LLM pseudo-labeler for DimASQP (scheme 3 from deep-research-report.md).

Reads a TRAIN sentence source (gold .txt, gold .jsonl, or raw .txt), calls an
OpenRouter-hosted LLM per sentence with a few-shot prompt, validates the
returned quadruplets against the task's category list and [1,9] VA range, and
writes the same four files that tools/convert_v2_all_languages.py produces so
the result can be fed to train.py without any loader changes.

Outputs (when --out_prefix=PATH):
    PATH.txt              TXT for train.py   (####-separated)
    PATH.jsonl            Gold-style JSONL   (for inspection / merging)
    PATH_sidecar.json     Sidecar JSON
    PATH_stats.json       Run statistics (total, dropped, LLM calls, tokens)

Hard safety rails:
    * The source path must NOT contain '_dev' / '_test' / '/dev.txt' / '/test.txt'.
      Pass --i_really_mean_it_not_a_leak to bypass (only for ablation *within*
      dev/test; NEVER merge such output into training data).
    * Category labels must be a subset of the task's train-set category list.
    * aspect / opinion must be either "NULL" or a substring of the sentence.

Typical usage (Phase A curve fit — cheap models):
    export OPENROUTER_API_KEY=sk-or-...
    python data/llm_pseudo_labeler.py \
        --task_domain eng_restaurant \
        --source_file data/v2/eng/eng_restaurant_train.jsonl \
        --llm_model_name meta-llama/llama-3.1-70b-instruct \
        --out_prefix data/v2/eng/eng_restaurant_train_pseudo__llama31-70b

    python data/llm_pseudo_labeler.py \
        --task_domain eng_restaurant \
        --source_file data/v2/eng/eng_restaurant_train.jsonl \
        --llm_model_name openai/gpt-4o-mini \
        --out_prefix data/v2/eng/eng_restaurant_train_pseudo__gpt4o-mini

Phase B (冲分) — swap the model name:
    --llm_model_name anthropic/claude-3.5-sonnet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

# Make `llm` importable when run as `python data/llm_pseudo_labeler.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from llm import (  # noqa: E402
    DEFAULTS,
    OpenRouterClient,
    OpenRouterError,
    PROMPT_VERSION,
    build_pseudo_label_prompt,
)
from llm.openrouter_client import parse_json_content  # noqa: E402


# --------------------------------------------------------------- I/O helpers
def _read_source_entries(path: str) -> List[Dict]:
    """Return a list of {"ID": str, "Text": str} regardless of source format.

    Supported formats:
      * JSONL  (keys: ID, Text)       — preferred, preserves IDs
      * TXT    (DimASQP ####-format)  — falls back to synthetic IDs
      * raw    (one sentence per line, set via --source_raw)
    """
    entries: List[Dict] = []
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    if path.endswith(".jsonl"):
        for i, ln in enumerate(lines):
            obj = json.loads(ln)
            entries.append({
                "ID": obj.get("ID", f"src_{i:06d}"),
                "Text": obj["Text"],
            })
        return entries

    if path.endswith(".txt"):
        for i, ln in enumerate(lines):
            text = ln.split("####", 1)[0]
            entries.append({"ID": f"src_{i:06d}", "Text": text})
        return entries

    # Generic fallback: assume one sentence per line.
    for i, ln in enumerate(lines):
        entries.append({"ID": f"src_{i:06d}", "Text": ln})
    return entries


def _read_gold_shots(gold_jsonl_path: str) -> List[Dict]:
    """Load gold JSONL entries to use as few-shot examples."""
    shots: List[Dict] = []
    if not os.path.exists(gold_jsonl_path):
        return shots
    with open(gold_jsonl_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            # Keep only examples with at least one fully non-NULL quadruplet for quality.
            if not obj.get("Quadruplet"):
                continue
            shots.append({"Text": obj["Text"], "Quadruplet": obj["Quadruplet"]})
    return shots


def _collect_category_list(gold_jsonl_path: str) -> List[str]:
    """Extract the set of category labels present in the gold training set."""
    if not os.path.exists(gold_jsonl_path):
        raise FileNotFoundError(
            f"Expected gold JSONL at {gold_jsonl_path} to harvest the category list.\n"
            "Run tools/convert_v2_all_languages.py first."
        )
    cats = set()
    with open(gold_jsonl_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            for q in obj.get("Quadruplet", []):
                cat = q.get("Category")
                if cat:
                    cats.add(cat)
    return sorted(cats)


# ----------------------------------------------------------- leak protection
def _looks_like_eval_split(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    return (
        "_dev" in p
        or "_test" in p
        or p.endswith("/dev.txt")
        or p.endswith("/test.txt")
        or p.endswith("/dev.jsonl")
        or p.endswith("/test.jsonl")
    )


# ---------------------------------------------------------- span validation
def _find_span(text: str, span_text: str) -> Tuple[int, int]:
    """Mirror of tools/convert_v2_all_languages.find_span (same normalization)."""
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


def _clean_quadruplet(q: Dict, text: str, allowed_categories: set) -> Optional[Dict]:
    """Validate one LLM-generated quadruplet. Returns the canonical dict or None."""
    aspect = q.get("aspect", "NULL")
    opinion = q.get("opinion", "NULL")
    category = q.get("category")
    valence = q.get("valence")
    arousal = q.get("arousal")

    if not isinstance(aspect, str) or not isinstance(opinion, str):
        return None
    if category not in allowed_categories:
        return None
    try:
        v = float(valence)
        a = float(arousal)
    except (TypeError, ValueError):
        return None
    if not (1.0 <= v <= 9.0 and 1.0 <= a <= 9.0):
        return None

    if aspect != "NULL":
        a0, a1 = _find_span(text, aspect)
        if a0 == -1:
            aspect = "NULL"
    if opinion != "NULL":
        o0, o1 = _find_span(text, opinion)
        if o0 == -1:
            opinion = "NULL"

    return {
        "Aspect": aspect,
        "Opinion": opinion,
        "Category": category,
        "VA": f"{v:.2f}#{a:.2f}",
    }


# -------------------------------------------------------------- disk cache
def _cache_key(sentence: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(PROMPT_VERSION.encode("utf-8"))
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(sentence.encode("utf-8"))
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


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Offline LLM pseudo-labeler for DimASQP (scheme 3)."
    )
    parser.add_argument("--task_domain", required=True,
                        help="Used to locate the gold JSONL for categories + few-shot, e.g., eng_restaurant.")
    parser.add_argument("--gold_jsonl", default=None,
                        help="Path to the task's gold train JSONL. "
                             "Default: data/v2/{lang}/{task_domain}_train.jsonl")
    parser.add_argument("--source_file", default=None,
                        help="Sentences to annotate. Defaults to the gold train JSONL. "
                             "Accepts .jsonl (ID+Text), .txt (#### format), or raw one-per-line.")
    parser.add_argument("--source_raw", action="store_true",
                        help="Force raw-text interpretation (one sentence per line) regardless of extension.")
    parser.add_argument("--out_prefix", required=True,
                        help="Output path prefix (without extension). Produces .txt / .jsonl / _sidecar.json / _stats.json.")
    parser.add_argument("--llm_model_name", default=DEFAULTS["model"],
                        help=f"OpenRouter model ID (default: {DEFAULTS['model']}).")
    parser.add_argument("--llm_temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--llm_top_p", type=float, default=DEFAULTS["top_p"])
    parser.add_argument("--llm_max_tokens", type=int, default=DEFAULTS["max_tokens"])
    parser.add_argument("--num_shots", type=int, default=DEFAULTS["num_shots"])
    parser.add_argument("--min_valid_quads", type=int, default=DEFAULTS["min_valid_quads"])
    parser.add_argument("--cache_dir", default=DEFAULTS["cache_dir"])
    parser.add_argument("--nrows", type=int, default=None,
                        help="Only process the first N rows (for smoke tests).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for few-shot sampling.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Do not call the LLM; emit cached results only (and skip uncached sentences).")
    parser.add_argument("--i_really_mean_it_not_a_leak", action="store_true",
                        help="Bypass the dev/test path guard. Do not use for training data!")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    lang = args.task_domain.split("_", 1)[0]
    gold_jsonl = args.gold_jsonl or os.path.join(
        "data", "v2", lang, f"{args.task_domain}_train.jsonl"
    )
    source_file = args.source_file or gold_jsonl

    # ---- leak guard ------------------------------------------------------
    if _looks_like_eval_split(source_file) and not args.i_really_mean_it_not_a_leak:
        parser.error(
            f"Refusing to run: source_file={source_file!r} looks like a dev/test split. "
            "Pseudo-labeling eval data and merging it into train.txt causes label leakage. "
            "Pass --i_really_mean_it_not_a_leak only if you are building a non-training ablation."
        )

    # ---- category list + few-shot sampling -------------------------------
    allowed_categories = _collect_category_list(gold_jsonl)
    if not allowed_categories:
        parser.error(f"No categories found in {gold_jsonl}. Is the file non-empty?")
    shot_pool = _read_gold_shots(gold_jsonl)
    if args.source_raw:
        entries = _read_source_entries(source_file) if os.path.exists(source_file) else []
        # Re-read as raw if extension would otherwise trigger JSONL path.
        with open(source_file, encoding="utf-8") as f:
            entries = [{"ID": f"raw_{i:06d}", "Text": ln.strip()}
                       for i, ln in enumerate(f) if ln.strip()]
    else:
        entries = _read_source_entries(source_file)
    if args.nrows is not None:
        entries = entries[: args.nrows]

    rng = random.Random(args.seed)
    n_shots = min(args.num_shots, len(shot_pool))
    shots = rng.sample(shot_pool, n_shots) if n_shots > 0 else []

    print(
        f"[llm_pseudo_labeler] task_domain={args.task_domain}  model={args.llm_model_name}"
        f"  sentences={len(entries)}  shots={len(shots)}  categories={len(allowed_categories)}"
    )

    # ---- LLM client ------------------------------------------------------
    client: Optional[OpenRouterClient] = None
    if not args.dry_run:
        try:
            client = OpenRouterClient()
        except OpenRouterError as e:
            parser.error(str(e))

    # ---- output buffers --------------------------------------------------
    txt_lines: List[str] = []
    gold_jsonl_out: List[Dict] = []
    sidecar_out: List[Dict] = []
    stats = {
        "task_domain": args.task_domain,
        "model": args.llm_model_name,
        "prompt_version": PROMPT_VERSION,
        "source_file": source_file,
        "n_sentences": len(entries),
        "n_written": 0,
        "n_dropped_no_valid_quads": 0,
        "n_parse_fail": 0,
        "n_http_fail": 0,
        "n_cache_hits": 0,
        "n_llm_calls": 0,
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "tokens_total": 0,
        "wallclock_s": 0.0,
        "prompt_num_shots": len(shots),
    }

    t0 = time.time()
    for idx, entry in enumerate(entries):
        text = entry["Text"]
        eid = entry["ID"]
        key = _cache_key(text, args.llm_model_name)
        cached = _cache_load(args.cache_dir, key)

        if cached is not None:
            stats["n_cache_hits"] += 1
            content = cached.get("content", "")
        elif args.dry_run:
            if args.verbose:
                print(f"[dry_run] skip uncached: {eid}")
            continue
        else:
            messages = build_pseudo_label_prompt(text, allowed_categories, shots)
            try:
                resp = client.chat(  # type: ignore[union-attr]
                    messages=messages,
                    model=args.llm_model_name,
                    temperature=args.llm_temperature,
                    top_p=args.llm_top_p,
                    max_tokens=args.llm_max_tokens,
                )
            except OpenRouterError as e:
                stats["n_http_fail"] += 1
                if args.verbose:
                    print(f"[http_fail] {eid}: {e}")
                continue
            content = resp.get("content", "") or ""
            usage = resp.get("usage") or {}
            stats["n_llm_calls"] += 1
            stats["tokens_prompt"] += int(usage.get("prompt_tokens") or 0)
            stats["tokens_completion"] += int(usage.get("completion_tokens") or 0)
            stats["tokens_total"] += int(usage.get("total_tokens") or 0)
            _cache_save(args.cache_dir, key, {"content": content, "usage": usage,
                                              "model_served": resp.get("model")})

        parsed = parse_json_content(content)
        if not parsed or "quadruplets" not in parsed:
            stats["n_parse_fail"] += 1
            if args.verbose:
                print(f"[parse_fail] {eid}: {content[:200]!r}")
            continue

        cleaned_quads = []
        for q in parsed["quadruplets"]:
            cq = _clean_quadruplet(q, text, set(allowed_categories))
            if cq is not None:
                cleaned_quads.append(cq)

        if len(cleaned_quads) < args.min_valid_quads:
            stats["n_dropped_no_valid_quads"] += 1
            continue

        # Build the internal TXT format and matching sidecar/gold entries.
        txt_quads_compact = []
        sidecar_quads = []
        for q in cleaned_quads:
            a0, a1 = _find_span(text, q["Aspect"])
            o0, o1 = _find_span(text, q["Opinion"])
            txt_quads_compact.append([q["Category"], f"{a0},{a1}", f"{o0},{o1}", q["VA"]])
            sidecar_quads.append({
                "Aspect": q["Aspect"], "Opinion": q["Opinion"],
                "Category": q["Category"], "VA": q["VA"],
                "aspect_span": [a0, a1], "opinion_span": [o0, o1],
            })

        txt_lines.append(f"{text}####{json.dumps(txt_quads_compact, ensure_ascii=False)}")
        gold_jsonl_out.append({"ID": f"pseudo_{eid}", "Text": text,
                               "Quadruplet": cleaned_quads})
        sidecar_out.append({
            "ID": f"pseudo_{eid}",
            "Text": text,
            "line_index": len(txt_lines) - 1,
            "quads": sidecar_quads,
        })
        stats["n_written"] += 1

        if args.verbose and (idx + 1) % 50 == 0:
            print(f"[progress] {idx + 1}/{len(entries)}  written={stats['n_written']}"
                  f"  llm_calls={stats['n_llm_calls']}  cache_hits={stats['n_cache_hits']}")

    stats["wallclock_s"] = round(time.time() - t0, 2)

    # ---- write outputs ---------------------------------------------------
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_prefix + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + ("\n" if txt_lines else ""))
    with open(args.out_prefix + ".jsonl", "w", encoding="utf-8") as f:
        for g in gold_jsonl_out:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    with open(args.out_prefix + "_sidecar.json", "w", encoding="utf-8") as f:
        json.dump(sidecar_out, f, ensure_ascii=False, indent=2)
    with open(args.out_prefix + "_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(
        f"\nDone. written={stats['n_written']}/{stats['n_sentences']}  "
        f"parse_fail={stats['n_parse_fail']}  dropped={stats['n_dropped_no_valid_quads']}  "
        f"http_fail={stats['n_http_fail']}  cache_hits={stats['n_cache_hits']}  "
        f"llm_calls={stats['n_llm_calls']}  tokens={stats['tokens_total']}  "
        f"wallclock={stats['wallclock_s']}s\n"
        f"  TXT     : {args.out_prefix}.txt\n"
        f"  JSONL   : {args.out_prefix}.jsonl\n"
        f"  sidecar : {args.out_prefix}_sidecar.json\n"
        f"  stats   : {args.out_prefix}_stats.json"
    )


if __name__ == "__main__":
    main()
