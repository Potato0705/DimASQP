"""Go/No-Go probes for the latent-emotion cross-lingual DimASQP direction.

Probe 1: Per-language VA distribution statistics (are Chinese/Japanese compressed?).
Probe 2: rus/tat/ukr parallel structure — are VA labels identical (label copy)
         or re-annotated (label noise across languages)?
Probe 3: Euclidean vs polar cF1 ranking — we run this only if predictions exist.

All analyses are deterministic and use only numpy + stdlib.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "v2"

LANG_DOMAINS = [
    ("eng", "restaurant"),
    ("eng", "laptop"),
    ("zho", "restaurant"),
    ("zho", "laptop"),
    ("jpn", "hotel"),
    ("rus", "restaurant"),
    ("tat", "restaurant"),
    ("ukr", "restaurant"),
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_va(s: str) -> tuple[float, float]:
    v, a = s.split("#")
    return float(v), float(a)


def extract_va_records(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        for q in r.get("Quadruplet", []):
            v, a = parse_va(q["VA"])
            out.append(
                {
                    "id": r["ID"],
                    "text": r.get("Text", ""),
                    "aspect": q.get("Aspect", "NULL"),
                    "opinion": q.get("Opinion", "NULL"),
                    "category": q.get("Category", ""),
                    "v": v,
                    "a": a,
                }
            )
    return out


def summarize(vs: np.ndarray, as_: np.ndarray) -> dict:
    def _stats(x: np.ndarray) -> dict:
        return {
            "n": int(len(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "q05": float(np.quantile(x, 0.05)),
            "q50": float(np.quantile(x, 0.50)),
            "q95": float(np.quantile(x, 0.95)),
            "range95": float(np.quantile(x, 0.95) - np.quantile(x, 0.05)),
        }

    corr = float(np.corrcoef(vs, as_)[0, 1]) if len(vs) > 1 else float("nan")
    return {"V": _stats(vs), "A": _stats(as_), "corr_VA": corr}


def probe1_va_distributions() -> dict:
    print("\n=== Probe 1: per-language VA distribution statistics (train split) ===")
    result = {}
    for lang, dom in LANG_DOMAINS:
        path = DATA_ROOT / lang / f"{lang}_{dom}_train.jsonl"
        if not path.exists():
            continue
        recs = extract_va_records(load_jsonl(path))
        vs = np.array([r["v"] for r in recs], dtype=np.float64)
        as_ = np.array([r["a"] for r in recs], dtype=np.float64)
        stats = summarize(vs, as_)
        result[f"{lang}_{dom}"] = stats
        print(
            f"  {lang}_{dom:10s}  quads={stats['V']['n']:>5d}  "
            f"V mean={stats['V']['mean']:.2f} std={stats['V']['std']:.2f} range95={stats['V']['range95']:.2f}  "
            f"A mean={stats['A']['mean']:.2f} std={stats['A']['std']:.2f} range95={stats['A']['range95']:.2f}  "
            f"corr(V,A)={stats['corr_VA']:+.3f}"
        )
    return result


def _key_quad(q: dict) -> tuple:
    """Match by category + VA (stable across languages under label-propagation)."""
    return (q["Category"], q["VA"])


def probe2_parallel_structure() -> dict:
    print("\n=== Probe 2: rus/tat/ukr parallel structure (same-ID VA agreement) ===")
    out = {}
    for split in ("train", "dev", "test"):
        paths = {
            lang: DATA_ROOT / lang / f"{lang}_restaurant_{split}.jsonl"
            for lang in ("rus", "tat", "ukr")
        }
        if not all(p.exists() for p in paths.values()):
            continue
        recs = {lang: load_jsonl(paths[lang]) for lang in paths}
        ids = {lang: {r["ID"] for r in recs[lang]} for lang in recs}
        common = ids["rus"] & ids["tat"] & ids["ukr"]
        idx = {
            lang: {r["ID"]: r for r in recs[lang]} for lang in recs
        }

        total_quads = 0
        mismatch_any_va = 0
        mismatch_any_cat = 0
        mismatch_count_quads = 0
        va_diffs_rus_tat = []
        va_diffs_rus_ukr = []
        va_diffs_tat_ukr = []

        for qid in common:
            qr = idx["rus"][qid]["Quadruplet"]
            qt = idx["tat"][qid]["Quadruplet"]
            qu = idx["ukr"][qid]["Quadruplet"]
            if not (len(qr) == len(qt) == len(qu)):
                mismatch_count_quads += 1
                continue
            # Match quads by category+VA key (robust to span surface form differences).
            # If label-copied exactly, for each rus quad there is a tat/ukr quad with
            # identical (Category, VA).
            kr = sorted([_key_quad(q) for q in qr])
            kt = sorted([_key_quad(q) for q in qt])
            ku = sorted([_key_quad(q) for q in qu])
            if kr != kt:
                mismatch_any_va += int(any(q[1] for q in kr) ^ any(q[1] for q in kt))
            # Fine-grained numeric diffs by aligning sorted-by-category order.
            def _sorted_quads(lst):
                return sorted(lst, key=lambda x: (x["Category"], x["VA"]))
            qr_s = _sorted_quads(qr)
            qt_s = _sorted_quads(qt)
            qu_s = _sorted_quads(qu)
            for a, b, c in zip(qr_s, qt_s, qu_s):
                total_quads += 1
                if a["Category"] != b["Category"] or a["Category"] != c["Category"]:
                    mismatch_any_cat += 1
                    continue
                vr, ar_ = parse_va(a["VA"])
                vt, at_ = parse_va(b["VA"])
                vu, au_ = parse_va(c["VA"])
                if a["VA"] != b["VA"] or a["VA"] != c["VA"]:
                    mismatch_any_va += 1
                va_diffs_rus_tat.append(
                    math.hypot(vr - vt, ar_ - at_)
                )
                va_diffs_rus_ukr.append(
                    math.hypot(vr - vu, ar_ - au_)
                )
                va_diffs_tat_ukr.append(
                    math.hypot(vt - vu, at_ - au_)
                )

        arr_rt = np.array(va_diffs_rus_tat) if va_diffs_rus_tat else np.array([0.0])
        arr_ru = np.array(va_diffs_rus_ukr) if va_diffs_rus_ukr else np.array([0.0])
        arr_tu = np.array(va_diffs_tat_ukr) if va_diffs_tat_ukr else np.array([0.0])

        summary = {
            "split": split,
            "common_ids": len(common),
            "total_quads_aligned": total_quads,
            "mismatched_quad_count": mismatch_count_quads,
            "mismatched_category": mismatch_any_cat,
            "mismatched_VA_string": mismatch_any_va,
            "VA_euclid_rus_vs_tat": {
                "mean": float(arr_rt.mean()),
                "max": float(arr_rt.max()),
                "frac_nonzero": float((arr_rt > 1e-9).mean()),
            },
            "VA_euclid_rus_vs_ukr": {
                "mean": float(arr_ru.mean()),
                "max": float(arr_ru.max()),
                "frac_nonzero": float((arr_ru > 1e-9).mean()),
            },
            "VA_euclid_tat_vs_ukr": {
                "mean": float(arr_tu.mean()),
                "max": float(arr_tu.max()),
                "frac_nonzero": float((arr_tu > 1e-9).mean()),
            },
        }
        out[split] = summary
        print(
            f"  [{split}] common_ids={summary['common_ids']}  aligned_quads={total_quads}  "
            f"mismatched_VA_strings={mismatch_any_va}  "
            f"mismatched_category={mismatch_any_cat}  "
            f"VA-euclid(rus,tat)={arr_rt.mean():.4f} max={arr_rt.max():.3f} "
            f"frac_nonzero={arr_rt.mean() > 1e-9 and (arr_rt > 1e-9).mean():.3f}"
        )
    return out


def ks_2samp(a: np.ndarray, b: np.ndarray) -> float:
    """Tiny two-sample Kolmogorov-Smirnov statistic (no p-value)."""
    a = np.sort(a)
    b = np.sort(b)
    all_x = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_x, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_x, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def probe1b_pairwise_ks(result: dict) -> None:
    print("\n=== Probe 1b: pairwise KS distance on V marginal (train) ===")
    # Rebuild raw arrays.
    arrs = {}
    for lang, dom in LANG_DOMAINS:
        key = f"{lang}_{dom}"
        if key not in result:
            continue
        path = DATA_ROOT / lang / f"{lang}_{dom}_train.jsonl"
        recs = extract_va_records(load_jsonl(path))
        arrs[key] = (
            np.array([r["v"] for r in recs]),
            np.array([r["a"] for r in recs]),
        )

    keys = list(arrs.keys())
    print("  V-marginal pairwise KS:")
    header = "    " + "  ".join(f"{k:>16s}" for k in keys)
    print(header)
    for k1 in keys:
        row = [f"{k1:>16s}"]
        for k2 in keys:
            if k1 == k2:
                row.append(f"{0.0:>16.3f}")
            else:
                row.append(f"{ks_2samp(arrs[k1][0], arrs[k2][0]):>16.3f}")
        print("  " + "  ".join(row))

    print("  A-marginal pairwise KS:")
    print(header)
    for k1 in keys:
        row = [f"{k1:>16s}"]
        for k2 in keys:
            if k1 == k2:
                row.append(f"{0.0:>16.3f}")
            else:
                row.append(f"{ks_2samp(arrs[k1][1], arrs[k2][1]):>16.3f}")
        print("  " + "  ".join(row))


def main() -> None:
    stats = probe1_va_distributions()
    probe1b_pairwise_ks(stats)
    parallel = probe2_parallel_structure()

    out_path = Path(__file__).resolve().parents[1] / "docs" / "probe_crosslingual_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"probe1_stats": stats, "probe2_parallel": parallel}, f, indent=2)
    print(f"\nWrote raw results to {out_path}")


if __name__ == "__main__":
    main()
