"""
Multi-seed ensemble evaluation.

Averages raw logit matrices from multiple models for quad extraction.
VA prediction supports two modes:
  --va_ensemble: average VA predictions from ALL models (each uses its own
                 hidden states + VA head). Better VA quality.
  default:       use a single reference model's VA head (--va_model_path).

Models are loaded one at a time to save GPU memory; hidden states and VA heads
are kept on CPU and moved to GPU only during VA inference.

Usage:
    python tools/ensemble_eval.py \
        --model_paths output/seed42 output/seed66 output/seed123 \
        --test_data data/eng/eng_restaurant_dev.txt \
        --sidecar data/eng/eng_restaurant_dev_sidecar.json \
        --gold data/eng/eng_restaurant_dev.jsonl \
        --va_ensemble
"""
import json
import os
import sys
import gc
import tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import torch.nn as _nn
import types as _types
import transformers.models.deberta_v2.modeling_deberta_v2 as _dv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import AcqpDataset, collate_fn
from utils.utils import load_train_model, load_train_args
from predict import create_pred_answer, attach_va_to_pred_answer, attach_span_pair_va
from tools.evaluate_local import read_jsonl_file, evaluate_predictions
from tools.generate_submission import load_sidecar, SENTIMENT_VA_MAP

device = "cuda" if torch.cuda.is_available() else "cpu"

# Patch: transformers.core_model_loading.WeightRenaming missing in some installs
if 'transformers.core_model_loading' not in sys.modules:
    _fake_cml = _types.ModuleType('transformers.core_model_loading')
    class _WeightRenaming: pass
    _fake_cml.WeightRenaming = _WeightRenaming
    sys.modules['transformers.core_model_loading'] = _fake_cml

if not hasattr(_dv2, "StableDropout"):
    class _StableDropout(_nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            import torch.nn.functional as F
            if self.training and self.drop_prob > 0:
                return F.dropout(x, p=self.drop_prob)
            return x

    _dv2.StableDropout = _StableDropout


def _patch_model_configs(model):
    """Repair old DeBERTa checkpoints for newer transformers versions."""
    for m in model.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None:
            if not hasattr(cfg, "_output_attentions"):
                cfg.__dict__["_output_attentions"] = cfg.__dict__.pop(
                    "output_attentions", False
                )
            if not hasattr(cfg, "torchscript"):
                cfg.__dict__["torchscript"] = False
        if type(m).__name__ == "DebertaV2Embeddings":
            for attr in ("token_type_embeddings", "embed_proj", "position_embeddings"):
                if not hasattr(m, attr):
                    object.__setattr__(m, attr, None)
    return model


def load_model_for_eval(model_path):
    """Load a saved training model onto the current eval device."""
    best_model_path = os.path.join(model_path, "best_model.pt")
    fallback_model_path = os.path.join(model_path, "model.pt")
    load_path = best_model_path if os.path.exists(best_model_path) else fallback_model_path
    map_location = None if device == "cuda" else torch.device("cpu")
    model = torch.load(load_path, weights_only=False, map_location=map_location)
    return _patch_model_configs(model)


def extract_logits_single(model_path, test_data_path, batch_size=16,
                          keep_hidden=False):
    """Extract raw logits from one model. Optionally keep hidden states for VA."""
    print(f"\n  Loading: {os.path.basename(model_path)}")
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

    with torch.no_grad():
        for data in tqdm(dataloader, desc="  Extracting"):
            for key in data:
                data[key] = data[key].to(device)
            pred = model(input_ids=data["input_ids"],
                         token_type_ids=data["token_type_ids"],
                         attention_mask=data["attention_mask"])
            for mat in pred["matrix"].cpu().numpy():
                raw_matrices.append(mat)
            for va in pred["va"].cpu().numpy():
                va_preds.append(va)
            if keep_hidden and "hidden_states" in pred:
                for hs in pred["hidden_states"].cpu():
                    hidden_states_list.append(hs)

    va_head = None
    if keep_hidden:
        # Select VA head based on va_mode, keep on CPU to save GPU memory
        va_mode = training_args.get('va_mode', 'position')
        if va_mode == 'opinion_guided' and hasattr(model, 'opinion_guided_va_head'):
            va_head = model.opinion_guided_va_head.cpu()
        elif hasattr(model, 'span_pair_va_head'):
            va_head = model.span_pair_va_head.cpu()

    del model
    torch.cuda.empty_cache()
    gc.collect()
    if keep_hidden:
        print(f"  Done: {len(raw_matrices)} samples + hidden states kept")
    else:
        print(f"  Done: {len(raw_matrices)} samples, GPU freed")

    return dataset, raw_matrices, va_preds, hidden_states_list, va_head, training_args


def stream_average_matrices(model_paths, test_data_path, batch_size=16,
                            va_model_path=None, keep_hidden=False,
                            temp_dir=None):
    """Average matrices with a disk-backed memmap to avoid RAM blow-up."""
    n_models = len(model_paths)
    if n_models == 0:
        raise ValueError("model_paths must not be empty")

    va_model_abs = os.path.abspath(va_model_path or model_paths[0])
    mmap_path = None
    mmap_shape = None
    mmap_array = None
    dataset = None
    training_args = None
    va_mode = None
    va_preds_list = []
    hidden_states_list = []
    va_head = None

    for model_path in model_paths:
        print(f"\n  Loading: {os.path.basename(model_path)}")
        model = load_model_for_eval(model_path)
        model = model.to(device)
        model.eval()

        current_args = load_train_args(model_path)
        tokenizer = AutoTokenizer.from_pretrained(current_args['model_name_or_path'])
        current_dataset = AcqpDataset(
            task_domain=current_args['task_domain'],
            tokenizer=tokenizer,
            data_path=test_data_path,
            max_seq_len=current_args['max_seq_len'],
            label_pattern=current_args['label_pattern'],
        )
        dataloader = DataLoader(
            current_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        current_va_mode = current_args.get('va_mode', 'position')
        if va_mode is None:
            va_mode = current_va_mode
        elif current_va_mode != va_mode:
            raise ValueError(f"Inconsistent va_mode across models: {va_mode} vs {current_va_mode}")

        if dataset is None:
            dataset = current_dataset
            training_args = current_args
        elif len(current_dataset) != len(dataset):
            raise ValueError("Dataset length mismatch across ensemble models")

        is_va_model = os.path.abspath(model_path) == va_model_abs
        offset = 0

        with torch.no_grad():
            for data in tqdm(dataloader, desc="  Extracting"):
                for key in data:
                    data[key] = data[key].to(device)
                pred = model(
                    input_ids=data["input_ids"],
                    token_type_ids=data["token_type_ids"],
                    attention_mask=data["attention_mask"],
                )

                matrices = pred["matrix"].detach().cpu().numpy().astype(np.float32, copy=False)
                if mmap_array is None:
                    sample_shape = matrices.shape[1:]
                    mmap_shape = (len(current_dataset),) + sample_shape
                    fd, mmap_path = tempfile.mkstemp(
                        prefix="ensemble_avg_",
                        suffix=".mmap",
                        dir=temp_dir,
                    )
                    os.close(fd)
                    mmap_array = np.memmap(mmap_path, mode="w+", dtype=np.float32, shape=mmap_shape)
                    mmap_array[:] = 0.0

                batch_size_now = matrices.shape[0]
                mmap_array[offset:offset + batch_size_now] += matrices / n_models

                if is_va_model:
                    if current_va_mode == 'position':
                        va_preds_list.extend(pred["va"].detach().cpu().numpy())
                    elif keep_hidden and "hidden_states" in pred:
                        for hs in pred["hidden_states"].detach().cpu():
                            hidden_states_list.append(hs)

                offset += batch_size_now
                del pred, matrices

        if is_va_model and keep_hidden:
            if current_va_mode == 'opinion_guided' and hasattr(model, 'opinion_guided_va_head'):
                va_head = model.opinion_guided_va_head.cpu()
            elif hasattr(model, 'span_pair_va_head'):
                va_head = model.span_pair_va_head.cpu()

        if mmap_array is not None:
            mmap_array.flush()
        del model
        torch.cuda.empty_cache()
        gc.collect()
        if keep_hidden and is_va_model and current_va_mode != 'position':
            print(f"  Done: {len(current_dataset)} samples + hidden states kept (stream avg)")
        else:
            print(f"  Done: {len(current_dataset)} samples, memmap updated")

    if mmap_array is not None:
        mmap_array.flush()
        del mmap_array

    return dataset, mmap_path, mmap_shape, va_preds_list, hidden_states_list, va_head, training_args, va_mode


def ensemble_average(all_matrices_list):
    """Average logit matrices across models."""
    n_models = len(all_matrices_list)
    n_samples = len(all_matrices_list[0])

    avg_matrices = []
    for i in range(n_samples):
        running_sum = np.zeros_like(all_matrices_list[0][i], dtype=np.float32)
        for m in range(n_models):
            running_sum += all_matrices_list[m][i].astype(np.float32, copy=False)
        avg_matrices.append(running_sum / n_models)

    return avg_matrices


def _run_va_head_single(va_head, hidden_states_tensor, pred_answer, token_map):
    """Run one VA head on one sample, return list of [category, asp, opi, va_str]."""
    result = attach_span_pair_va(pred_answer, hidden_states_tensor, token_map,
                                 va_head, device)
    return result


def decode_with_va_ensemble(dataset, raw_matrices, all_hidden_states, all_va_heads,
                            threshold, training_args):
    """Decode with ensemble logits + VA averaged from ALL models."""
    label_types = dataset.label_types
    dimension_types = dataset.dimension_types
    sentiment2id = dataset.sentiment2id
    label_pattern = training_args['label_pattern']
    df = dataset.df
    all_preds = []
    n_models = len(all_va_heads)

    for i in range(len(df)):
        mat = raw_matrices[i]
        mat_pred = np.argwhere(mat > threshold).tolist()
        token_map = df.iloc[i]['Token_Index_Map_Char_Index']
        text = df.iloc[i]['text']

        pred_answer = create_pred_answer(
            mat_pred, [], [], token_map, label_pattern,
            label_types, dimension_types, sentiment2id
        )

        if not pred_answer:
            all_preds.append({
                "text": text,
                "pred_answer_with_va": [],
                "line_index": i,
            })
            continue

        # Collect VA predictions from each model
        va_per_model = []  # list of list-of-quads, each quad has va_str
        for m in range(n_models):
            va_head_m = all_va_heads[m].to(device)
            hs_m = all_hidden_states[m][i]
            result_m = attach_span_pair_va(pred_answer, hs_m, token_map,
                                           va_head_m, device)
            va_per_model.append(result_m)
            va_head_m.cpu()  # free GPU immediately

        # Average VA across models
        n_quads = len(pred_answer)
        averaged_quads = []
        for q in range(n_quads):
            if len(pred_answer[q]) < 4:
                averaged_quads.append(pred_answer[q])
                continue

            v_sum, a_sum, count = 0.0, 0.0, 0
            for m in range(n_models):
                if q < len(va_per_model[m]) and len(va_per_model[m][q]) >= 4:
                    va_str = va_per_model[m][q][3]
                    if '#' in str(va_str):
                        v, a = va_str.split('#')
                        v_sum += float(v)
                        a_sum += float(a)
                        count += 1

            if count > 0:
                v_avg = np.clip(v_sum / count, 1.0, 9.0)
                a_avg = np.clip(a_sum / count, 1.0, 9.0)
                va_str_avg = f"{v_avg:.2f}#{a_avg:.2f}"
            else:
                va_str_avg = "5.00#5.00"

            averaged_quads.append([pred_answer[q][0], pred_answer[q][1],
                                   pred_answer[q][2], va_str_avg])

        all_preds.append({
            "text": text,
            "pred_answer_with_va": averaged_quads,
            "line_index": i,
        })

    return all_preds


def decode_at_threshold_with_span_va(dataset, raw_matrices, hidden_states_list,
                                      va_head, threshold, training_args):
    """Decode with ensemble logits + VA from single reference model."""
    label_types = dataset.label_types
    dimension_types = dataset.dimension_types
    sentiment2id = dataset.sentiment2id
    label_pattern = training_args['label_pattern']
    df = dataset.df
    all_preds = []

    # Move VA head to GPU for inference
    if va_head is not None:
        va_head = va_head.to(device)

    for i in range(len(df)):
        mat = raw_matrices[i]
        mat_pred = np.argwhere(mat > threshold).tolist()
        token_map = df.iloc[i]['Token_Index_Map_Char_Index']
        text = df.iloc[i]['text']

        pred_answer = create_pred_answer(
            mat_pred, [], [], token_map, label_pattern,
            label_types, dimension_types, sentiment2id
        )

        # Use VA from reference model
        if va_head is not None and hidden_states_list:
            pred_with_va = attach_span_pair_va(
                pred_answer, hidden_states_list[i], token_map,
                va_head, device)
        else:
            pred_with_va = [[q[0], q[1], q[2], "5.00#5.00"] for q in pred_answer if len(q) >= 3]

        all_preds.append({
            "text": text,
            "pred_answer_with_va": pred_with_va,
            "line_index": i,
        })

    # Move VA head back to CPU
    if va_head is not None:
        va_head.cpu()

    return all_preds


def decode_at_threshold_with_position_va(dataset, raw_matrices, va_preds_list,
                                         threshold, training_args):
    """Decode with ensemble logits + per-position VA from one or more models."""
    label_types = dataset.label_types
    dimension_types = dataset.dimension_types
    sentiment2id = dataset.sentiment2id
    label_pattern = training_args['label_pattern']
    df = dataset.df
    all_preds = []

    for i in range(len(df)):
        mat = raw_matrices[i]
        va = va_preds_list[i]
        mat_pred = np.argwhere(mat > threshold).tolist()
        token_map = df.iloc[i]['Token_Index_Map_Char_Index']
        text = df.iloc[i]['text']

        pred_answer = create_pred_answer(
            mat_pred, [], [], token_map, label_pattern,
            label_types, dimension_types, sentiment2id
        )
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
    parser.add_argument("--model_paths", nargs="+", required=True,
                        help="Paths to model directories for logit averaging")
    parser.add_argument("--va_model_path", type=str, default=None,
                        help="Model to use for VA (default: first model). Ignored with --va_ensemble")
    parser.add_argument("--va_ensemble", action="store_true",
                        help="Average VA predictions from ALL models instead of using a single reference")
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
    if args.va_ensemble:
        print(f"  VA mode: ensemble (average VA from ALL {n_models} models)")
    else:
        print(f"  VA mode: single model ({os.path.basename(va_model)})")
    print(f"{'=' * 70}")

    # Step 1: Extract logits from each model
    all_matrices_list = []
    all_va_preds_list = []  # list of position-VA tensors (one per model)
    all_hidden_states = []   # list of lists (one per model) for VA ensemble
    all_va_heads = []        # list of VA heads for VA ensemble
    va_preds_list = []       # single model position VA (non-ensemble mode)
    hidden_states_list = []  # single model hidden states (non-ensemble mode)
    va_head = None
    dataset = None
    training_args = None
    va_mode = None

    # Step 1: Stream-average matrix logits onto disk (memmap) to avoid RAM blow-up.
    # For large datasets (e.g. laptop test with 1000 samples × 134 labels × 128×128)
    # loading all matrices into RAM simultaneously causes OOM / segfault.
    print(f"\nStream-averaging {n_models} models' logits to disk...")
    (dataset, mmap_path, mmap_shape,
     va_preds_list_stream, hidden_states_list_stream,
     va_head_stream, training_args, va_mode) = stream_average_matrices(
        args.model_paths, args.test_data,
        va_model_path=va_model,
        keep_hidden=(not args.va_ensemble),
    )

    # Load mmap into a read-only numpy array for threshold sweep
    avg_mmap = np.memmap(mmap_path, mode="r", dtype=np.float32, shape=mmap_shape)
    avg_matrices = list(avg_mmap)   # list of per-sample ndarrays (view, no copy)

    # Merge VA outputs from stream pass
    if not args.va_ensemble:
        va_preds_list = va_preds_list_stream
        hidden_states_list = hidden_states_list_stream
        va_head = va_head_stream

    # VA ensemble: still need per-model VA — load each model a second time (small tensors)
    avg_va_preds = None
    if args.va_ensemble:
        print(f"\nCollecting per-model VA for ensemble...")
        for model_path in args.model_paths:
            current_va_mode = load_train_args(model_path).get('va_mode', 'position')
            _, _, va_preds, hs_list, vah, _ = extract_logits_single(
                model_path, args.test_data,
                keep_hidden=(current_va_mode != 'position'))
            if va_mode == 'position':
                all_va_preds_list.append(va_preds)
            else:
                all_hidden_states.append(hs_list)
                all_va_heads.append(vah)
        if va_mode == 'position':
            avg_va_preds = ensemble_average(all_va_preds_list)

    print(f"  Ensemble logits ready: {len(avg_matrices)} samples")
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
        if va_mode == 'position':
            if args.va_ensemble:
                all_preds = decode_at_threshold_with_position_va(
                    dataset, avg_matrices, avg_va_preds, threshold, training_args)
            else:
                all_preds = decode_at_threshold_with_position_va(
                    dataset, avg_matrices, va_preds_list, threshold, training_args)
        elif args.va_ensemble:
            all_preds = decode_with_va_ensemble(
                dataset, avg_matrices, all_hidden_states, all_va_heads,
                threshold, training_args)
        else:
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
    va_mode_str = "VA-ensemble" if args.va_ensemble else "single-VA"
    print(f"\n{'=' * 70}")
    print(f"  ENSEMBLE BEST ({va_mode_str}): threshold={best['threshold']:.1f} -> cF1={best['cF1']:.4f}")
    print(f"  cPrecision={best['cPrecision']:.4f}, cRecall={best['cRecall']:.4f}")
    print(f"{'=' * 70}")

    # Clean up disk-backed memmap temp file
    if mmap_path and os.path.exists(mmap_path):
        try:
            del avg_mmap, avg_matrices
            os.remove(mmap_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
