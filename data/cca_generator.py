"""Compositional Category Augmentation (CCA) generator for DimASQP.

Three-stage pipeline that exploits the Entity×Attribute Cartesian-product
structure of ABSA categories to generate training data for unseen/rare
category combinations:

    Stage 1 — Entity Grounding:   summarise how an entity appears in reviews
    Stage 2 — Attribute Grounding: summarise how an attribute angle is expressed
    Stage 3 — Compositional Generation: combine both contexts to synthesise
              new sentences for a target E#A category
    (opt)   — Cross-Verification: LLM re-classifies generated sentences;
              discard if the predicted category ≠ target (category drift)

Inputs:
    * category_gap_report_{task_domain}.json  (from tools/category_analysis.py)
    * gold train JSONL

Outputs (same schema as llm_pseudo_labeler.py):
    {out_prefix}.txt          — ####-format for train.py
    {out_prefix}.jsonl        — gold-style JSONL
    {out_prefix}_sidecar.json
    {out_prefix}_stats.json

Usage:
    python data/cca_generator.py \
        --task_domain eng_restaurant \
        --gap_report category_gap_report_eng_restaurant.json \
        --gold_jsonl data/v2/eng/eng_restaurant_train.jsonl \
        --out_prefix data/v2/eng/eng_restaurant_train_cca__llama31-70b
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
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from llm import (  # noqa: E402
    DEFAULTS,
    OpenRouterClient,
    OpenRouterError,
    PROMPT_VERSION,
    build_entity_grounding_prompt,
    build_attribute_grounding_prompt,
    build_cca_generation_prompt,
    build_cross_verify_prompt,
)
from llm.config import CCA_DEFAULTS  # noqa: E402
from llm.openrouter_client import parse_json_content  # noqa: E402


# ---------------------------------------------------------------- data helpers
def _load_jsonl(path: str) -> List[Dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                entries.append(json.loads(ln))
    return entries


def _load_gap_report(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _index_by_entity(entries: List[Dict]) -> Dict[str, List[Dict]]:
    idx: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        seen = set()
        for q in e.get("Quadruplet", []):
            ent = q.get("Category", "").split("#", 1)[0]
            if ent and ent not in seen:
                idx[ent].append(e)
                seen.add(ent)
    return dict(idx)


def _index_by_attribute(entries: List[Dict]) -> Dict[str, List[Dict]]:
    idx: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        seen = set()
        for q in e.get("Quadruplet", []):
            parts = q.get("Category", "").split("#", 1)
            attr = parts[1] if len(parts) == 2 else ""
            if attr and attr not in seen:
                idx[attr].append(e)
                seen.add(attr)
    return dict(idx)


# ----------------------------------------------------------- span validation
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


def _clean_generated_quad(
    q: Dict, text: str, target_category: str
) -> Optional[Dict]:
    aspect = q.get("aspect", "NULL")
    opinion = q.get("opinion", "NULL")
    category = q.get("category", "")
    try:
        v = float(q.get("valence", 5.0))
        a = float(q.get("arousal", 5.0))
    except (TypeError, ValueError):
        return None
    if not (1.0 <= v <= 9.0 and 1.0 <= a <= 9.0):
        v = max(1.0, min(9.0, v))
        a = max(1.0, min(9.0, a))
    if category != target_category:
        return None
    if not isinstance(aspect, str):
        aspect = "NULL"
    if not isinstance(opinion, str):
        opinion = "NULL"
    if aspect != "NULL":
        a0, _ = _find_span(text, aspect)
        if a0 == -1:
            aspect = "NULL"
    if opinion != "NULL":
        o0, _ = _find_span(text, opinion)
        if o0 == -1:
            opinion = "NULL"
    return {
        "Aspect": aspect,
        "Opinion": opinion,
        "Category": target_category,
        "VA": f"{v:.2f}#{a:.2f}",
    }


# ---------------------------------------------------------------- disk cache
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


# ---------------------------------------------------------- LLM call wrapper
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


# -------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="CCA: Compositional Category Augmentation for DimASQP."
    )
    parser.add_argument("--task_domain", required=True)
    parser.add_argument("--gap_report", required=True,
                        help="Path to category_gap_report_{task_domain}.json")
    parser.add_argument("--gold_jsonl", default=None,
                        help="Gold train JSONL. Default: data/v2/{lang}/{task_domain}_train.jsonl")
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--llm_model_name", default=DEFAULTS["model"])
    parser.add_argument("--n_per_category", type=int,
                        default=CCA_DEFAULTS["n_per_category"])
    parser.add_argument("--generation_batch_size", type=int,
                        default=CCA_DEFAULTS["generation_batch_size"])
    parser.add_argument("--max_grounding_examples", type=int,
                        default=CCA_DEFAULTS["max_grounding_examples"])
    parser.add_argument("--max_anchor_examples", type=int,
                        default=CCA_DEFAULTS["max_anchor_examples"])
    parser.add_argument("--no_cross_verify", action="store_true")
    parser.add_argument("--verify_confidence_threshold", type=float,
                        default=CCA_DEFAULTS["verify_confidence_threshold"])
    parser.add_argument("--rare_threshold", type=int,
                        default=CCA_DEFAULTS["rare_threshold"])
    parser.add_argument("--cache_dir", default=CCA_DEFAULTS["cache_dir"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nrows", type=int, default=None,
                        help="Limit to first N gap categories (for smoke tests).")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    lang = args.task_domain.split("_", 1)[0]
    gold_jsonl = args.gold_jsonl or os.path.join(
        _REPO_ROOT, "data", "v2", lang, f"{args.task_domain}_train.jsonl"
    )

    report = _load_gap_report(args.gap_report)
    gold_entries = _load_jsonl(gold_jsonl)
    if not gold_entries:
        parser.error(f"No entries in {gold_jsonl}")

    ent_index = _index_by_entity(gold_entries)
    attr_index = _index_by_attribute(gold_entries)
    all_categories = sorted(set(
        q.get("Category", "")
        for e in gold_entries for q in e.get("Quadruplet", [])
        if q.get("Category")
    ))

    gaps_to_augment = [
        g for g in report["gaps"]
        if g["status"] != "adequate"
    ]
    if args.nrows is not None:
        gaps_to_augment = gaps_to_augment[:args.nrows]

    print(f"[CCA] task={args.task_domain}  model={args.llm_model_name}  "
          f"categories_to_augment={len(gaps_to_augment)}  "
          f"n_per_category={args.n_per_category}  "
          f"cross_verify={'off' if args.no_cross_verify else 'on'}")

    client: Optional[OpenRouterClient] = None
    if not args.dry_run:
        try:
            client = OpenRouterClient()
        except OpenRouterError as e:
            parser.error(str(e))

    rng = random.Random(args.seed)
    stats: Dict[str, Any] = {
        "task_domain": args.task_domain,
        "model": args.llm_model_name,
        "prompt_version": PROMPT_VERSION,
        "pipeline": "CCA",
        "n_target_categories": len(gaps_to_augment),
        "n_per_category": args.n_per_category,
        "cross_verify": not args.no_cross_verify,
        "n_written": 0,
        "n_generated_raw": 0,
        "n_clean_fail": 0,
        "n_verify_reject": 0,
        "n_verify_pass": 0,
        "n_parse_fail": 0,
        "n_http_fail": 0,
        "n_cache_hits": 0,
        "n_llm_calls": 0,
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "tokens_total": 0,
        "wallclock_s": 0.0,
        "per_category": {},
    }

    txt_lines: List[str] = []
    jsonl_out: List[Dict] = []
    sidecar_out: List[Dict] = []

    grounding_cache: Dict[str, str] = {}

    t0 = time.time()

    for gap_idx, gap in enumerate(gaps_to_augment):
        entity = gap["entity"]
        attribute = gap["attribute"]
        target_cat = gap["category"]
        n_target = args.n_per_category

        if args.verbose:
            print(f"\n[{gap_idx+1}/{len(gaps_to_augment)}] {target_cat} "
                  f"(status={gap['status']}, train={gap['train_count']})")

        # ---- Stage 1: Entity Grounding ----
        eg_key = f"entity:{entity}"
        if eg_key not in grounding_cache:
            ent_examples = ent_index.get(entity, [])
            if ent_examples:
                sampled = rng.sample(ent_examples, min(args.max_grounding_examples, len(ent_examples)))
                msgs = build_entity_grounding_prompt(entity, sampled, args.max_grounding_examples)
                content_hash = hashlib.sha256(entity.encode()).hexdigest()
                result = _llm_call(
                    client, msgs, args.llm_model_name,
                    CCA_DEFAULTS["grounding_temperature"],
                    DEFAULTS["max_tokens"], args.cache_dir,
                    "entity_grounding", content_hash,
                    args.dry_run, stats, args.verbose,
                )
                grounding_cache[eg_key] = result or "{}"
            else:
                grounding_cache[eg_key] = json.dumps({
                    "entity": entity,
                    "typical_aspects": [],
                    "sentence_patterns": [],
                    "sentiment_range": "unknown (no training examples)",
                })
        entity_summary = grounding_cache[eg_key]

        # ---- Stage 2: Attribute Grounding ----
        ag_key = f"attribute:{attribute}"
        if ag_key not in grounding_cache:
            attr_examples = attr_index.get(attribute, [])
            if attr_examples:
                sampled = rng.sample(attr_examples, min(args.max_grounding_examples, len(attr_examples)))
                msgs = build_attribute_grounding_prompt(attribute, sampled, args.max_grounding_examples)
                content_hash = hashlib.sha256(attribute.encode()).hexdigest()
                result = _llm_call(
                    client, msgs, args.llm_model_name,
                    CCA_DEFAULTS["grounding_temperature"],
                    DEFAULTS["max_tokens"], args.cache_dir,
                    "attribute_grounding", content_hash,
                    args.dry_run, stats, args.verbose,
                )
                grounding_cache[ag_key] = result or "{}"
            else:
                grounding_cache[ag_key] = json.dumps({
                    "attribute": attribute,
                    "measures": "unknown",
                    "typical_opinions": [],
                    "va_patterns": "unknown (no training examples)",
                })
        attribute_summary = grounding_cache[ag_key]

        # ---- Stage 3: Compositional Generation (in batches) ----
        ent_examples = ent_index.get(entity, [])
        attr_examples = attr_index.get(attribute, [])
        anchor_pool = ent_examples + attr_examples
        if anchor_pool:
            rng.shuffle(anchor_pool)

        cat_generated = 0
        cat_verified = 0
        cat_written = 0
        n_remaining = n_target

        batch_idx = 0
        while n_remaining > 0:
            batch_size = min(args.generation_batch_size, n_remaining)
            anchors = anchor_pool[:args.max_anchor_examples] if anchor_pool else []
            msgs = build_cca_generation_prompt(
                entity, attribute,
                entity_summary, attribute_summary,
                anchors, n_generate=batch_size,
                max_anchors=args.max_anchor_examples,
            )
            content_hash = hashlib.sha256(
                f"{target_cat}:batch{batch_idx}:seed{args.seed}".encode()
            ).hexdigest()
            raw = _llm_call(
                client, msgs, args.llm_model_name,
                CCA_DEFAULTS["generation_temperature"],
                CCA_DEFAULTS["generation_max_tokens"],
                args.cache_dir, "cca_generation", content_hash,
                args.dry_run, stats, args.verbose,
            )
            if raw is None:
                break

            parsed = parse_json_content(raw)
            if parsed is None:
                if isinstance(raw, str) and raw.strip().startswith("["):
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        pass
            if parsed is None:
                stats["n_parse_fail"] += 1
                if args.verbose:
                    print(f"  [parse_fail] batch {batch_idx}")
                break

            items = parsed if isinstance(parsed, list) else parsed.get("sentences", parsed.get("quadruplets", []))
            if not isinstance(items, list):
                stats["n_parse_fail"] += 1
                break

            for item in items:
                if not isinstance(item, dict):
                    continue
                sentence = item.get("sentence", "")
                quads_raw = item.get("quadruplets", [])
                if not sentence or not isinstance(quads_raw, list):
                    continue

                stats["n_generated_raw"] += len(quads_raw)
                cat_generated += len(quads_raw)

                cleaned_quads = []
                for q in quads_raw:
                    cq = _clean_generated_quad(q, sentence, target_cat)
                    if cq is not None:
                        cleaned_quads.append(cq)
                    else:
                        stats["n_clean_fail"] += 1

                if not cleaned_quads:
                    continue

                # ---- Cross-Verification ----
                if not args.no_cross_verify:
                    verified_quads = []
                    for cq in cleaned_quads:
                        v_msgs = build_cross_verify_prompt(
                            sentence, cq["Aspect"], cq["Opinion"],
                            target_cat, all_categories,
                        )
                        v_hash = hashlib.sha256(
                            f"{sentence}:{cq['Aspect']}:{cq['Opinion']}".encode()
                        ).hexdigest()
                        v_raw = _llm_call(
                            client, v_msgs, args.llm_model_name,
                            CCA_DEFAULTS["verify_temperature"],
                            256, args.cache_dir, "cross_verify",
                            v_hash, args.dry_run, stats, args.verbose,
                        )
                        if v_raw is None:
                            verified_quads.append(cq)
                            continue
                        v_parsed = parse_json_content(v_raw)
                        if v_parsed is None:
                            verified_quads.append(cq)
                            continue
                        pred_cat = v_parsed.get("category", "")
                        confidence = float(v_parsed.get("confidence", 0.0))
                        if pred_cat == target_cat:
                            stats["n_verify_pass"] += 1
                            cat_verified += 1
                            verified_quads.append(cq)
                        elif confidence < args.verify_confidence_threshold:
                            stats["n_verify_pass"] += 1
                            cat_verified += 1
                            verified_quads.append(cq)
                        else:
                            stats["n_verify_reject"] += 1
                            if args.verbose:
                                print(f"  [verify_reject] predicted={pred_cat} "
                                      f"conf={confidence:.2f} for target={target_cat}")
                    cleaned_quads = verified_quads

                if not cleaned_quads:
                    continue

                # ---- Write row ----
                line_idx = len(txt_lines)
                txt_quads = []
                sidecar_quads = []
                for cq in cleaned_quads:
                    a0, a1 = _find_span(sentence, cq["Aspect"])
                    o0, o1 = _find_span(sentence, cq["Opinion"])
                    txt_quads.append([cq["Category"], f"{a0},{a1}", f"{o0},{o1}", cq["VA"]])
                    sidecar_quads.append({
                        "Aspect": cq["Aspect"], "Opinion": cq["Opinion"],
                        "Category": cq["Category"], "VA": cq["VA"],
                        "aspect_span": [a0, a1], "opinion_span": [o0, o1],
                    })

                txt_lines.append(f"{sentence}####{json.dumps(txt_quads, ensure_ascii=False)}")
                jsonl_out.append({
                    "ID": f"cca_{target_cat}_{line_idx:04d}",
                    "Text": sentence,
                    "Quadruplet": cleaned_quads,
                })
                sidecar_out.append({
                    "ID": f"cca_{target_cat}_{line_idx:04d}",
                    "Text": sentence,
                    "line_index": line_idx,
                    "quads": sidecar_quads,
                    "source_category": target_cat,
                })
                cat_written += 1
                stats["n_written"] += 1
                n_remaining -= 1

            batch_idx += 1
            if batch_idx > 20:
                if args.verbose:
                    print(f"  [warn] max batches reached for {target_cat}")
                break

        stats["per_category"][target_cat] = {
            "status": gap["status"],
            "train_count": gap["train_count"],
            "generated_raw": cat_generated,
            "verified": cat_verified if not args.no_cross_verify else cat_generated,
            "written": cat_written,
        }
        if args.verbose:
            print(f"  → {target_cat}: written={cat_written}/{n_target}")

    stats["wallclock_s"] = round(time.time() - t0, 2)

    # ---- write outputs ----
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_prefix + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + ("\n" if txt_lines else ""))
    with open(args.out_prefix + ".jsonl", "w", encoding="utf-8") as f:
        for g in jsonl_out:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    with open(args.out_prefix + "_sidecar.json", "w", encoding="utf-8") as f:
        json.dump(sidecar_out, f, ensure_ascii=False, indent=2)
    with open(args.out_prefix + "_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[CCA] Done. written={stats['n_written']}  "
          f"generated_raw={stats['n_generated_raw']}  "
          f"clean_fail={stats['n_clean_fail']}  "
          f"verify_reject={stats['n_verify_reject']}  "
          f"parse_fail={stats['n_parse_fail']}  "
          f"http_fail={stats['n_http_fail']}  "
          f"cache_hits={stats['n_cache_hits']}  "
          f"llm_calls={stats['n_llm_calls']}  "
          f"tokens={stats['tokens_total']}  "
          f"wallclock={stats['wallclock_s']}s")
    print(f"  TXT     : {args.out_prefix}.txt")
    print(f"  JSONL   : {args.out_prefix}.jsonl")
    print(f"  sidecar : {args.out_prefix}_sidecar.json")
    print(f"  stats   : {args.out_prefix}_stats.json")


if __name__ == "__main__":
    main()
