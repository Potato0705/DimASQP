"""
Convert prediction CSV/JSON from predict.py to official DimASQP submission JSONL format.

Phase 1 uses centroid VA mapping (discrete sentiment -> fixed VA values).
Phase 2+ will use real VA predictions from the VA regression head.

Usage:
    python tools/generate_submission.py \
        --pred_json output/.../eng_restaurant_dev_predictions.json \
        --sidecar data/eng/eng_restaurant_dev_sidecar.json \
        --output submission/eng_restaurant_dev_pred.jsonl
"""
import json
import os
import argparse

# Phase 1: Centroid VA mapping for discrete sentiment
SENTIMENT_VA_MAP = {
    "0": "2.50#5.50",   # negative -> low valence, mid arousal
    "1": "5.00#5.50",   # neutral -> mid valence, mid arousal
    "2": "7.50#5.50",   # positive -> high valence, mid arousal
    "negative": "2.50#5.50",
    "neutral": "5.00#5.50",
    "positive": "7.50#5.50",
}


def load_predictions(pred_json_path):
    """Load predictions from predict.py output JSON."""
    with open(pred_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sidecar(sidecar_path):
    """Load sidecar JSON to get ID mappings."""
    with open(sidecar_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pred_to_submission(predictions, sidecar_data, use_va_pred=False):
    """Convert predictions to submission JSONL format.

    Args:
        predictions: list of dicts from predict.py (each has Text_Id, text, pred_answer)
        sidecar_data: list of dicts with ID/Text/line_index mapping
        use_va_pred: if True, use VA predictions from model; if False, use centroid mapping

    Returns:
        list of dicts in submission format
    """
    # Build line_index -> sidecar entry mapping
    idx_to_sidecar = {entry["line_index"]: entry for entry in sidecar_data}

    submissions = []
    for pred in predictions:
        text_id = pred["Text_Id"]
        text = pred["text"]

        # Find matching sidecar entry
        sidecar_entry = idx_to_sidecar.get(text_id)
        if sidecar_entry is None:
            continue

        original_id = sidecar_entry["ID"]
        original_text = sidecar_entry["Text"]

        # Process predicted quadruplets
        pred_answer = pred.get("pred_answer", [])
        quadruplets = []
        for quad in pred_answer:
            if len(quad) < 4:
                continue

            category, aspect_idx, opinion_idx, sentiment_id = quad[0], quad[1], quad[2], quad[3]

            # Extract text spans from char indices
            asp_start, asp_end = map(int, aspect_idx.split(","))
            opi_start, opi_end = map(int, opinion_idx.split(","))

            if asp_start == -1:
                aspect_text = "NULL"
            else:
                aspect_text = original_text[asp_start:asp_end]

            if opi_start == -1:
                opinion_text = "NULL"
            else:
                opinion_text = original_text[opi_start:opi_end]

            # VA value
            if use_va_pred and "va_pred" in quad:
                va_str = quad["va_pred"]
            else:
                va_str = SENTIMENT_VA_MAP.get(str(sentiment_id), "5.00#5.50")

            quadruplets.append({
                "Aspect": aspect_text,
                "Category": category,
                "Opinion": opinion_text,
                "VA": va_str
            })

        submissions.append({
            "ID": original_id,
            "Quadruplet": quadruplets
        })

    return submissions


def main():
    parser = argparse.ArgumentParser(description="Generate DimASQP submission JSONL")
    parser.add_argument("--pred_json", required=True, type=str,
                        help="Path to predictions JSON from predict.py")
    parser.add_argument("--sidecar", required=True, type=str,
                        help="Path to sidecar JSON from convert_dimasqp.py")
    parser.add_argument("--output", required=True, type=str,
                        help="Output submission JSONL path")
    parser.add_argument("--use_va_pred", action="store_true",
                        help="Use VA predictions from model (Phase 2+)")
    args = parser.parse_args()

    predictions = load_predictions(args.pred_json)
    sidecar_data = load_sidecar(args.sidecar)
    submissions = pred_to_submission(predictions, sidecar_data, args.use_va_pred)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in submissions:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Generated {len(submissions)} submission entries -> {args.output}")
    # Count total quadruplets
    total_quads = sum(len(e["Quadruplet"]) for e in submissions)
    print(f"Total quadruplets: {total_quads}")


if __name__ == "__main__":
    main()
