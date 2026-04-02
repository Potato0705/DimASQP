"""
Threshold sweep on matrix predictions without retraining.

Saves raw matrix logits from model, then evaluates cF1 at different thresholds.

Usage:
    python tools/threshold_sweep.py \
        --model_path output/eng_restaurant_category_... \
        --test_data data/eng/eng_restaurant_dev.txt \
        --sidecar data/eng/eng_restaurant_dev_sidecar.json \
        --gold data/eng/eng_restaurant_dev.jsonl
"""
import json
import os
import sys
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import AcqpDataset, collate_fn
from utils.utils import load_train_args
from predict import create_pred_answer, attach_va_to_pred_answer
from tools.evaluate_local import read_jsonl_file, evaluate_predictions
from tools.generate_submission import load_sidecar, SENTIMENT_VA_MAP

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_for_eval(model_path):
    """Load a saved training model onto the current eval device."""
    best_model_path = os.path.join(model_path, "best_model.pt")
    fallback_model_path = os.path.join(model_path, "model.pt")
    load_path = best_model_path if os.path.exists(best_model_path) else fallback_model_path
    map_location = None if device == "cuda" else torch.device("cpu")
    return torch.load(load_path, weights_only=False, map_location=map_location)


def extract_raw_logits(model_path, test_data_path, batch_size=16):
    """Extract raw matrix logits, VA predictions, and hidden states from model."""
    model = load_model_for_eval(model_path)
    model = model.to(device)
    model.eval()

    training_args = load_train_args(model_path)
    tokenizer = AutoTokenizer.from_pretrained(training_args['model_name_or_path'])
    dataset = AcqpDataset(
        task_domain=training_args['task_domain'],
        tokenizer=tokenizer,
        data_path=test_data_path,
        max_seq_len=training_args['max_seq_len'],
        label_pattern=training_args['label_pattern'],
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate_fn)

    raw_matrices = []
    va_preds = []
    hidden_states_list = []
    va_mode = training_args.get('va_mode', 'position')
    has_span_va = va_mode in ('span_pair', 'opinion_guided')

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Extracting logits"):
            for key in data:
                data[key] = data[key].to(device)
            pred = model(input_ids=data["input_ids"],
                         token_type_ids=data["token_type_ids"],
                         attention_mask=data["attention_mask"])
            for mat in pred["matrix"].cpu().numpy():
                raw_matrices.append(mat)
            for va in pred["va"].cpu().numpy():
                va_preds.append(va)
            # Store hidden states for span-pair / opinion-guided VA
            if has_span_va and "hidden_states" in pred:
                for hs in pred["hidden_states"].cpu():
                    hidden_states_list.append(hs)

    return dataset, raw_matrices, va_preds, hidden_states_list, training_args, model


def decode_at_threshold(dataset, raw_matrices, va_preds, hidden_states_list,
                        threshold, training_args, model=None):
    """Decode predictions at a given threshold and return submission entries."""
    from predict import attach_span_pair_va
    label_pattern = training_args['label_pattern']
    label_types = dataset.label_types
    dimension_types = dataset.dimension_types
    sentiment2id = dataset.sentiment2id
    va_mode = training_args.get('va_mode', 'position')
    # Determine which VA head to use for inference
    if va_mode == 'opinion_guided' and model is not None and hasattr(model, 'opinion_guided_va_head'):
        va_head_obj = model.opinion_guided_va_head
        has_span_va = True
    elif va_mode == 'span_pair' and model is not None and hasattr(model, 'span_pair_va_head'):
        va_head_obj = model.span_pair_va_head
        has_span_va = True
    else:
        va_head_obj = None
        has_span_va = False

    df = dataset.df
    all_preds = []

    for i in range(len(df)):
        mat = raw_matrices[i]
        va = va_preds[i]

        # Apply threshold
        mat_pred = np.argwhere(mat > threshold).tolist()

        # Dummy dim_seq and sen_seq (not used in category mode)
        dim_seq_pred = []
        sen_seq_pred = []

        token_map = df.iloc[i]['Token_Index_Map_Char_Index']
        text = df.iloc[i]['text']

        pred_answer = create_pred_answer(
            mat_pred, dim_seq_pred, sen_seq_pred,
            token_map, label_pattern, label_types,
            dimension_types, sentiment2id
        )

        # Attach VA: use span-pair/opinion-guided VA if available, else per-position
        if has_span_va and hidden_states_list and va_head_obj is not None:
            pred_with_va = attach_span_pair_va(
                pred_answer, hidden_states_list[i], token_map,
                va_head_obj, device)
        else:
            pred_with_va = attach_va_to_pred_answer(pred_answer, va, token_map)
        all_preds.append({
            "text": text,
            "pred_answer_with_va": pred_with_va,
            "line_index": i,
        })

    return all_preds


def preds_to_submission(all_preds, sidecar_data):
    """Convert predictions to submission format."""
    idx_to_sidecar = {entry["line_index"]: entry for entry in sidecar_data}
    submissions = []

    for pred in all_preds:
        idx = pred["line_index"]
        sidecar_entry = idx_to_sidecar.get(idx)
        if sidecar_entry is None:
            continue

        quads = []
        for quad in pred["pred_answer_with_va"]:
            if len(quad) < 4:
                continue
            category, asp_idx, opi_idx, va_str = quad[0], quad[1], quad[2], quad[3]
            asp_s, asp_e = map(int, asp_idx.split(","))
            opi_s, opi_e = map(int, opi_idx.split(","))
            text = sidecar_entry["Text"]

            aspect_text = "NULL" if asp_s == -1 else text[asp_s:asp_e].strip()
            opinion_text = "NULL" if opi_s == -1 else text[opi_s:opi_e].strip()

            if '#' not in str(va_str):
                va_str = SENTIMENT_VA_MAP.get(str(va_str), "5.00#5.50")

            quads.append({
                "Aspect": aspect_text,
                "Category": category,
                "Opinion": opinion_text,
                "VA": va_str,
            })

        submissions.append({"ID": sidecar_entry["ID"], "Quadruplet": quads})
    return submissions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--thresholds", type=str,
                        default="-2.0,-1.5,-1.0,-0.5,-0.3,-0.1,0.0,0.3,0.5,1.0")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]

    print("Extracting raw logits...")
    dataset, raw_matrices, va_preds, hidden_states_list, training_args, model = extract_raw_logits(
        args.model_path, args.test_data)
    sidecar_data = load_sidecar(args.sidecar)
    gold_data = read_jsonl_file(args.gold, task=3, data_type='gold')

    va_mode = training_args.get('va_mode', 'position')
    print(f"  VA mode: {va_mode}")

    print(f"\n{'='*70}")
    print(f"{'Threshold':>10} {'cF1':>8} {'cPrec':>8} {'cRecall':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'#Pred':>7} {'VA%':>6}")
    print(f"{'='*70}")

    results = []
    for threshold in thresholds:
        all_preds = decode_at_threshold(dataset, raw_matrices, va_preds, hidden_states_list,
                                        threshold, training_args, model)
        submissions = preds_to_submission(all_preds, sidecar_data)
        total_preds = sum(len(s["Quadruplet"]) for s in submissions)

        # Write temp file
        tmp_path = f"submission/_tmp_thresh_{threshold:.1f}.jsonl"
        os.makedirs("submission", exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            for entry in submissions:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        pred_data = read_jsonl_file(tmp_path, task=3, data_type='pred')
        metrics = evaluate_predictions(gold_data, pred_data, task=3)

        va_pct = (metrics['cTP'] / metrics['TP'] * 100) if metrics['TP'] > 0 else 0

        print(f"{threshold:>10.1f} {metrics['cF1']:>8.4f} {metrics['cPrecision']:>8.4f} "
              f"{metrics['cRecall']:>8.4f} {metrics['TP']:>6} {metrics['FP']:>6} "
              f"{metrics['FN']:>6} {total_preds:>7} {va_pct:>5.1f}%")

        results.append({"threshold": threshold, **metrics, "total_preds": total_preds})

        os.remove(tmp_path)

    # Find best
    best = max(results, key=lambda x: x['cF1'])
    print(f"\n{'='*70}")
    print(f"Best threshold: {best['threshold']:.1f} -> cF1={best['cF1']:.4f}")
    print(f"  cPrecision={best['cPrecision']:.4f}, cRecall={best['cRecall']:.4f}")


if __name__ == "__main__":
    main()
