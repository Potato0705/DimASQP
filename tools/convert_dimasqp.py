"""
Convert DimASQP JSONL data to the internal TXT format used by One-ASQP.

TXT format: text####[["CATEGORY", "aspect_start,aspect_end", "opinion_start,opinion_end", "sentiment_id"]]

Also produces a sidecar JSON file storing original VA values and ID mappings
for later use in Phase 2 (VA regression) and submission generation.

Usage:
    python tools/convert_dimasqp.py --data_dir ./data --lang eng --domain restaurant
    python tools/convert_dimasqp.py --all --data_dir ./data
"""
import json
import os
import re
import argparse

DATASETS = [
    ("eng", "restaurant"),
    ("eng", "laptop"),
    ("zho", "restaurant"),
    ("zho", "laptop"),
    ("jpn", "hotel"),
    ("rus", "restaurant"),
    ("tat", "restaurant"),
    ("ukr", "restaurant"),
]


def va_to_sentiment_id(va_str):
    """Map VA string to discrete sentiment ID (Phase 1 temporary mapping).
    V < 4.0  -> negative (0)
    4.0 <= V <= 6.0 -> neutral (1)
    V > 6.0  -> positive (2)
    """
    v_str, a_str = va_str.split("#")
    v = float(v_str)
    if v < 4.0:
        return 0
    elif v <= 6.0:
        return 1
    else:
        return 2


def find_span_in_text(text, span_text):
    """Find the character start and end index of span_text in text.
    Returns (start, end) where text[start:end] == span_text,
    or (-1, -1) if not found or span is NULL.
    """
    if span_text == "NULL" or span_text is None:
        return -1, -1

    # Direct substring search
    idx = text.find(span_text)
    if idx != -1:
        return idx, idx + len(span_text)

    # Case-insensitive search as fallback
    idx = text.lower().find(span_text.lower())
    if idx != -1:
        return idx, idx + len(span_text)

    # Try matching with normalized whitespace
    # Some texts have extra spaces around punctuation like "ca n ' t"
    # Try removing extra spaces and re-matching
    normalized_text = re.sub(r'\s+', ' ', text)
    normalized_span = re.sub(r'\s+', ' ', span_text)
    idx = normalized_text.find(normalized_span)
    if idx != -1:
        return idx, idx + len(normalized_span)

    return -1, -1


def convert_jsonl_to_txt(jsonl_path, txt_path, sidecar_path):
    """Convert a JSONL file to the internal TXT format + sidecar JSON.

    Returns:
        dict with statistics: total, converted, skipped, span_miss
    """
    stats = {"total": 0, "converted": 0, "skipped": 0, "span_miss": 0}
    sidecar_data = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    txt_lines = []
    for line in lines:
        entry = json.loads(line)
        text = entry["Text"]
        entry_id = entry["ID"]
        quads = entry.get("Quadruplet", [])
        stats["total"] += 1

        if not quads:
            stats["skipped"] += 1
            continue

        converted_quads = []
        sidecar_quads = []
        has_valid_quad = False

        for quad in quads:
            aspect_text = quad["Aspect"]
            opinion_text = quad["Opinion"]
            category = quad["Category"]
            va_str = quad["VA"]

            # Find spans in text
            asp_start, asp_end = find_span_in_text(text, aspect_text)
            opi_start, opi_end = find_span_in_text(text, opinion_text)

            # Track span misses for non-NULL elements
            if aspect_text != "NULL" and asp_start == -1:
                stats["span_miss"] += 1
            if opinion_text != "NULL" and opi_start == -1:
                stats["span_miss"] += 1

            sentiment_id = va_to_sentiment_id(va_str)

            converted_quads.append([
                category,
                f"{asp_start},{asp_end}",
                f"{opi_start},{opi_end}",
                str(sentiment_id)
            ])
            sidecar_quads.append({
                "Aspect": aspect_text,
                "Opinion": opinion_text,
                "Category": category,
                "VA": va_str,
                "aspect_span": [asp_start, asp_end],
                "opinion_span": [opi_start, opi_end],
                "sentiment_id": sentiment_id
            })
            has_valid_quad = True

        if has_valid_quad:
            txt_line = f"{text}####{json.dumps(converted_quads, ensure_ascii=False)}"
            txt_lines.append(txt_line)
            sidecar_data.append({
                "ID": entry_id,
                "Text": text,
                "line_index": len(txt_lines) - 1,
                "quads": sidecar_quads
            })
            stats["converted"] += 1
        else:
            stats["skipped"] += 1

    # Write TXT file
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in txt_lines:
            f.write(line + "\n")

    # Write sidecar JSON
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, ensure_ascii=False, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert DimASQP JSONL to One-ASQP TXT format")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--lang", type=str, help="Language code (e.g., eng, zho)")
    parser.add_argument("--domain", type=str, help="Domain (e.g., restaurant, laptop)")
    parser.add_argument("--all", action="store_true", help="Convert all language-domain combinations")
    args = parser.parse_args()

    if args.all:
        targets = DATASETS
    elif args.lang and args.domain:
        targets = [(args.lang, args.domain)]
    else:
        parser.error("Specify --all or both --lang and --domain")

    print("=" * 60)
    for lang, domain in targets:
        name = f"{lang}_{domain}"
        for split in ["train", "dev"]:
            jsonl_path = os.path.join(args.data_dir, lang, f"{name}_{split}.jsonl")
            if not os.path.exists(jsonl_path):
                print(f"[SKIP] {jsonl_path} not found")
                continue

            txt_path = os.path.join(args.data_dir, lang, f"{name}_{split}.txt")
            sidecar_path = os.path.join(args.data_dir, lang, f"{name}_{split}_sidecar.json")
            stats = convert_jsonl_to_txt(jsonl_path, txt_path, sidecar_path)
            print(f"[{name}_{split}] total={stats['total']}, converted={stats['converted']}, "
                  f"skipped={stats['skipped']}, span_miss={stats['span_miss']}")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
