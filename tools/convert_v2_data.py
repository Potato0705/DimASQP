"""
Convert eng_v2 JSONL data to the internal TXT + sidecar format.

Input  (eng_v2/):
  eng_{domain}_train_alltasks.jsonl
  eng_{domain}_dev_task3.jsonl
  eng_{domain}_test_task3.jsonl

Output (eng_v2/):
  eng_{domain}_train.txt           train data for train.py
  eng_{domain}_train_sidecar.json
  eng_{domain}_dev.txt             dev data for train.py / evaluate_local.py
  eng_{domain}_dev_sidecar.json
  eng_{domain}_dev.jsonl           gold JSONL for threshold_sweep / ensemble_eval
  eng_{domain}_test.txt
  eng_{domain}_test_sidecar.json
  eng_{domain}_test.jsonl          gold JSONL for held-out test evaluation

Usage:
    python tools/convert_v2_data.py
"""
import json
import os
import re

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "eng_v2")

SPLITS = [
    ("train", "eng_{domain}_train_alltasks.jsonl"),
    ("dev",   "eng_{domain}_dev_task3.jsonl"),
    ("test",  "eng_{domain}_test_task3.jsonl"),
]

DOMAINS = ["restaurant", "laptop"]


def find_span(text, span_text):
    """Return (start, end) char indices of span_text in text, or (-1,-1)."""
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


def convert(jsonl_path, txt_path, sidecar_path, gold_path=None):
    with open(jsonl_path, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]

    txt_lines, sidecar, gold = [], [], []
    stats = {"total": len(entries), "converted": 0, "skipped": 0, "span_miss": 0}

    for entry in entries:
        text = entry["Text"]
        eid  = entry["ID"]
        quads = entry.get("Quadruplet", [])
        if not quads:
            stats["skipped"] += 1
            continue

        txt_quads, sc_quads = [], []
        for q in quads:
            asp_txt = q["Aspect"]
            opi_txt = q["Opinion"]
            cat     = q["Category"]
            va_str  = q["VA"]

            a0, a1 = find_span(text, asp_txt)
            o0, o1 = find_span(text, opi_txt)
            if asp_txt != "NULL" and a0 == -1:
                stats["span_miss"] += 1
            if opi_txt != "NULL" and o0 == -1:
                stats["span_miss"] += 1

            txt_quads.append([cat, f"{a0},{a1}", f"{o0},{o1}", va_str])
            sc_quads.append({
                "Aspect": asp_txt, "Opinion": opi_txt,
                "Category": cat, "VA": va_str,
                "aspect_span": [a0, a1], "opinion_span": [o0, o1],
            })

        txt_lines.append(f"{text}####{json.dumps(txt_quads, ensure_ascii=False)}")
        sidecar.append({"ID": eid, "Text": text,
                        "line_index": len(txt_lines)-1, "quads": sc_quads})
        gold.append({"ID": eid, "Text": text, "Quadruplet": quads})
        stats["converted"] += 1

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)
    if gold_path:
        with open(gold_path, "w", encoding="utf-8") as f:
            for g in gold:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")

    return stats


def main():
    for domain in DOMAINS:
        print(f"\n=== {domain} ===")
        for split, tmpl in SPLITS:
            src = os.path.join(BASE, tmpl.format(domain=domain))
            if not os.path.exists(src):
                print(f"  [SKIP] {src} not found")
                continue
            prefix = os.path.join(BASE, f"eng_{domain}_{split}")
            gp = prefix + ".jsonl" if split in ("dev", "test") else None
            st = convert(src, prefix + ".txt", prefix + "_sidecar.json", gp)
            print(f"  {split}: {st['converted']}/{st['total']} converted, "
                  f"span_miss={st['span_miss']}, skipped={st['skipped']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
