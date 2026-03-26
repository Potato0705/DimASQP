"""
Multi-seed ensemble evaluation.

Averages raw logit matrices from multiple models, then uses a reference model's
span-pair VA head for VA prediction. Models are loaded one at a time to save GPU memory.

Usage:
    python tools/ensemble_eval.py \
        --model_paths output/model_seed42 output/model_seed66 output/model_seed123 \
        --va_model_path output/model_seed66 \
        --test_data data/eng/eng_restaurant_dev.txt \
        --sidecar data/eng/eng_restaurant_dev_sidecar.json \
        --gold data/eng/eng_restaurant_dev.jsonl
"""
import json
import os
import sys
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import AcqpDataset, collate_fn
from utils.utils import load_train_model, load_train_args
from predict import create_pred_answer, attach_va_to_pred_answer, attach_span_pair_va
from tools.evaluate_local import read_jsonl_file, evaluate_predictions
from tools.generate_submission import load_sidecar, SENTIMENT_VA_MAP

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_logits_single(model_path, test_data_path, batch_size=16,
                          keep_hidden=False):
    """Extract raw logits from one model. Optionally keep hidden states for VA."""
    print(f"\n  Loading: {os.path.basename(model_path)}")
    model = load_train_model(model_path)
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
    hidden_states_list = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="  Extracting"):
            for key in data:
                data[key] = data[key].to(device)
            pred = model(input_ids=data["input_ids"],
                         token_type_ids=data["token_type_ids"],
                         attention_mask=data["attention_mask"])
            for mat in pred["matrix"].cpu().numpy():
                raw_matrices.append(mat)
            if keep_hidden and "hidden_states" in pred:
                for hs in pred["hidden_states"].cpu():
                    hidden_states_list.append(hs)

    va_head = None
    if keep_hidden:
        # Select VA head based on va_mode
        va_mode = training_args.get('va_mode', 'position')
        if va_mode == 'opinion_guided' and hasattr(model, 'opinion_guided_va_head'):
            va_head = model.opinion_guided_va_head.to(device)
        elif hasattr(model, 'span_pair_va_head'):
            va_head = model.span_pair_va_head.to(device)

    if not keep_hidden:
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  Done: {len(raw_matrices)} samples, GPU freed")
    else:
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  Done: {len(raw_matrices)} samples + {len(hidden_states_list)} hidden states, GPU freed")

    return dataset, raw_matrices, hidden_states_list, va_head, training_args


def ensemble_average(all_matrices_list):
    """Average logit matrices across models."""
    n_models = len(all_matrices_list)
    n_samples = len(all_matrices_list[0])

    avg_matrices = []
    for i in range(n_samples):
        mat_stack = np.stack([all_matrices_list[m][i] for m in range(n_models)])
        avg_matrices.append(mat_stack.mean(axis=0))

    return avg_matrices


def decode_at_threshold_with_span_va(dataset, raw_matrices, hidden_states_list,
                                      va_head, threshold, training_args):
    """Decode with ensemble logits + span-pair VA from reference model."""
    label_types = dataset.label_types
    dimension_types = dataset.dimension_types
    sentiment2id = dataset.sentiment2id
    label_pattern = training_args['label_pattern']
    df = dataset.df
    all_preds = []

    for i in range(len(df)):
        mat = raw_matrices[i]
        mat_pred = np.argwhere(mat > threshold).tolist()
        token_map = df.iloc[i]['Token_Index_Map_Char_Index']
        text = df.iloc[i]['text']

        pred_answer = create_pred_answer(
            mat_pred, [], [], token_map, label_pattern,
            label_types, dimension_types, sentiment2id
        )

        # Use span-pair VA from reference model
        if va_head is not None and hidden_states_list:
            pred_with_va = attach_span_pair_va(
                pred_answer, hidden_states_list[i], token_map,
                va_head, device)
        else:
            # Fallback: no VA (shouldn't happen)
            pred_with_va = [[q[0], q[1], q[2], "5.00#5.00"] for q in pred_answer if len(q) >= 3]

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
    parser.add_argument("--model_paths", nargs="+", required=True,
                        help="Paths to model directories for logit averaging")
    parser.add_argument("--va_model_path", type=str, default=None,
                        help="Model to use for span-pair VA (default: first model)")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--thresholds", type=str,
                        default="-3.0,-2.0,-1.5,-1.0,-0.5,-0.3,-0.1,0.0,0.3,0.5,1.0")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    n_models = len(args.model_paths)
    va_model = args.va_model_path or args.model_paths[0]

    print(f"{'=' * 70}")
    print(f"  ENSEMBLE: {n_models} models (logit avg)")
    print(f"  VA model: {os.path.basename(va_model)}")
    print(f"{'=' * 70}")

    # Step 1: Extract logits from each model
    all_matrices_list = []
    hidden_states_list = []
    va_head = None
    dataset = None
    training_args = None

    for model_path in args.model_paths:
        is_va_model = (os.path.abspath(model_path) == os.path.abspath(va_model))
        ds, matrices, hs_list, vah, t_args = extract_logits_single(
            model_path, args.test_data, keep_hidden=is_va_model)
        all_matrices_list.append(matrices)
        if is_va_model:
            hidden_states_list = hs_list
            va_head = vah
            va_mode = t_args.get('va_mode', 'span_pair')
            print(f"  >> Using this model for VA ({va_mode})")
        if dataset is None:
            dataset = ds
            training_args = t_args

    # If va_model is not in model_paths, load it separately
    if not hidden_states_list:
        print(f"\n  Loading VA model separately: {os.path.basename(va_model)}")
        _, _, hidden_states_list, va_head, _ = extract_logits_single(
            va_model, args.test_data, keep_hidden=True)

    # Step 2: Average logit matrices
    print(f"\nAveraging {n_models} models' logits...")
    avg_matrices = ensemble_average(all_matrices_list)
    print(f"  Ensemble logits ready: {len(avg_matrices)} samples")

    del all_matrices_list
    gc.collect()

    # Step 3: Threshold sweep
    sidecar_data = load_sidecar(args.sidecar)
    gold_data = read_jsonl_file(args.gold, task=3, data_type='gold')

    print(f"\n{'=' * 70}")
    print(f"{'Threshold':>10} {'cF1':>8} {'cPrec':>8} {'cRecall':>8} "
          f"{'TP':>6} {'FP':>6} {'FN':>6} {'#Pred':>7} {'VA%':>6}")
    print(f"{'=' * 70}")

    results = []
    for threshold in thresholds:
        all_preds = decode_at_threshold_with_span_va(
            dataset, avg_matrices, hidden_states_list,
            va_head, threshold, training_args)
        submissions = preds_to_submission(all_preds, sidecar_data)
        total_preds = sum(len(s["Quadruplet"]) for s in submissions)

        tmp_path = "submission/_tmp_ensemble.jsonl"
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

    best = max(results, key=lambda x: x['cF1'])
    print(f"\n{'=' * 70}")
    print(f"  ENSEMBLE BEST: threshold={best['threshold']:.1f} -> cF1={best['cF1']:.4f}")
    print(f"  cPrecision={best['cPrecision']:.4f}, cRecall={best['cRecall']:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
