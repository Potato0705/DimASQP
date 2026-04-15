"""
Convert the DimABSA 2026 (track_a/subtask_3) v2 JSONL data to the internal
TXT + sidecar + gold-JSONL format used by train.py / evaluate_local.py.

This is a generalization of tools/convert_v2_data.py that covers all 6 languages
(eng / jpn / rus / tat / ukr / zho) and all 8 (lang, domain) task combinations.

Input layout  (data/v2/{lang}/):
    {lang}_{domain}_train_alltasks.jsonl
    {lang}_{domain}_dev_task3.jsonl
    {lang}_{domain}_test_task3.jsonl

Output layout (data/v2/{lang}/):
    {lang}_{domain}_train.txt            TXT for train.py
    {lang}_{domain}_train.jsonl          Gold JSONL (kept in parallel for eval / LLM prompt sampling)
    {lang}_{domain}_train_sidecar.json
    {lang}_{domain}_dev.txt
    {lang}_{domain}_dev.jsonl
    {lang}_{domain}_dev_sidecar.json
    {lang}_{domain}_test.txt
    {lang}_{domain}_test.jsonl
    {lang}_{domain}_test_sidecar.json

Usage:
    python tools/convert_v2_all_languages.py
    python tools/convert_v2_all_languages.py --task eng_restaurant
    python tools/convert_v2_all_languages.py --data_dir data/v2
"""
import argparse
import json
import os
import re

TASKS = [
    ("eng", "restaurant"),
    ("eng", "laptop"),
    ("zho", "restaurant"),
    ("zho", "laptop"),
    ("jpn", "hotel"),
    ("rus", "restaurant"),
    ("tat", "restaurant"),
    ("ukr", "restaurant"),
]

SPLITS = [
    ("train", "{lang}_{domain}_train_alltasks.jsonl"),
    ("dev",   "{lang}_{domain}_dev_task3.jsonl"),
    ("test",  "{lang}_{domain}_test_task3.jsonl"),
]


def find_span(text, span_text):
    """Return (start, end) char indices of span_text in text, or (-1, -1) if not found / NULL."""
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


def convert(jsonl_path, txt_path, sidecar_path, gold_path):
    with open(jsonl_path, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]

    txt_lines, sidecar, gold = [], [], []
    stats = {"total": len(entries), "converted": 0, "skipped": 0, "span_miss": 0}

    for entry in entries:
        text = entry["Text"]
        eid = entry["ID"]
        quads = entry.get("Quadruplet", [])
        if not quads:
            stats["skipped"] += 1
            continue

        txt_quads, sc_quads = [], []
        for q in quads:
            asp_txt = q.get("Aspect", "NULL")
            opi_txt = q.get("Opinion", "NULL")
            cat = q["Category"]
            va_str = q["VA"]

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
        sidecar.append({
            "ID": eid,
            "Text": text,
            "line_index": len(txt_lines) - 1,
            "quads": sc_quads,
        })
        gold.append({"ID": eid, "Text": text, "Quadruplet": quads})
        stats["converted"] += 1

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)
    with open(gold_path, "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert DimABSA 2026 v2 JSONL to internal TXT+sidecar+gold format for all languages"
    )
    parser.add_argument("--data_dir", default="data/v2",
                        help="Base dir containing per-language subdirs (default: data/v2)")
    parser.add_argument("--task", default=None,
                        help="Process only one task (e.g., 'eng_restaurant'). Default: all tasks.")
    args = parser.parse_args()

    if args.task:
        if "_" not in args.task:
            parser.error("--task must be '<lang>_<domain>' (e.g., eng_restaurant)")
        lang, domain = args.task.split("_", 1)
        targets = [(lang, domain)]
    else:
        targets = TASKS

    print("=" * 72)
    grand_stats = {"tasks": 0, "files": 0, "total": 0, "converted": 0, "span_miss": 0}
    for lang, domain in targets:
        lang_dir = os.path.join(args.data_dir, lang)
        task_name = f"{lang}_{domain}"
        print(f"\n=== {task_name} ===")
        grand_stats["tasks"] += 1
        for split, tmpl in SPLITS:
            src = os.path.join(lang_dir, tmpl.format(lang=lang, domain=domain))
            if not os.path.exists(src):
                print(f"  [SKIP] {src} not found")
                continue
            prefix = os.path.join(lang_dir, f"{task_name}_{split}")
            st = convert(
                src,
                prefix + ".txt",
                prefix + "_sidecar.json",
                prefix + ".jsonl",
            )
            grand_stats["files"] += 1
            grand_stats["total"] += st["total"]
            grand_stats["converted"] += st["converted"]
            grand_stats["span_miss"] += st["span_miss"]
            print(
                f"  {split}: converted {st['converted']}/{st['total']}  "
                f"skipped={st['skipped']}  span_miss={st['span_miss']}"
            )

    print("\n" + "=" * 72)
    print(
        f"Done. tasks={grand_stats['tasks']}  files={grand_stats['files']}  "
        f"total={grand_stats['total']}  converted={grand_stats['converted']}  "
        f"span_miss={grand_stats['span_miss']}"
    )


if __name__ == "__main__":
    main()
