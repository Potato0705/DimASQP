"""Probe 3: Euclidean cF1 vs polar-aware cF1 on existing eng predictions.

For each seed run under output/laptop_restaurant_all_2026-04-13/{domain}/,
we match predictions to gold by (Aspect, Opinion, Category) exact match
(case-insensitive), then compute both metrics and check whether seed-level
ranking shifts between the two.

Polar cF1 rationale: Russell's circumplex places emotions on a 2D plane
centered at the scale midpoint (5,5); direction (angle) encodes emotion
category, radius encodes intensity. Treating V and A as independent
Euclidean coordinates ignores this structure. Polar distance weights
angular error (category identity) more heavily than radial error
(intensity magnitude) when both exist.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parents[1] / "output" / "laptop_restaurant_all_2026-04-13"
GOLD = {
    "eng_restaurant": Path(__file__).resolve().parents[1] / "data" / "v2" / "eng" / "eng_restaurant_test.jsonl",
    "eng_laptop": Path(__file__).resolve().parents[1] / "data" / "v2" / "eng" / "eng_laptop_test.jsonl",
}

SQRT128 = math.sqrt(128.0)


def load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_va(s):
    v, a = s.split("#")
    return float(v), float(a)


def key(q):
    return (
        (q.get("Aspect") or "NULL").strip().lower(),
        (q.get("Opinion") or "NULL").strip().lower(),
        (q.get("Category") or "").strip().upper(),
    )


def ctp_euclid(vg, ag, vp, ap):
    d = math.hypot(vp - vg, ap - ag)
    return max(0.0, 1.0 - d / SQRT128)


def polar(v, a, cx=5.0, cy=5.0):
    dx = v - cx
    dy = a - cy
    r = math.hypot(dx, dy)
    th = math.atan2(dy, dx)
    return r, th


def ctp_polar(vg, ag, vp, ap, w_theta=0.7, w_r=0.3):
    rg, tg = polar(vg, ag)
    rp, tp = polar(vp, ap)
    # Circular angle distance in [0, pi].
    dtheta = min(abs(tp - tg), 2 * math.pi - abs(tp - tg))
    # Normalized angular error in [0, 1].
    ang_err = dtheta / math.pi
    # Max radius for VA in [1,9] centered at (5,5) is sqrt(32).
    rmax = math.sqrt(32.0)
    rad_err = min(1.0, abs(rp - rg) / rmax)
    # If both points are near origin (neutral), angle is degenerate; down-weight ang.
    if rg < 0.5 and rp < 0.5:
        d = rad_err
    else:
        # Gate angular contribution by gold radius (low-intensity points have uncertain angle).
        gate = min(1.0, rg / 2.0)
        d = w_theta * gate * ang_err + w_r * rad_err + (1 - w_theta * gate - w_r) * 0.0
    # Normalize to [0, 1] so that a "perfect" prediction → 1.0, opposite → 0.0.
    return max(0.0, 1.0 - d)


def compute_cf1(pred_file: Path, gold_file: Path):
    preds = load_jsonl(pred_file)
    gold = load_jsonl(gold_file)
    gold_idx = {r["ID"]: r for r in gold}

    tp_euclid = 0.0
    tp_polar = 0.0
    n_matched = 0
    n_pred = 0
    n_gold = 0

    for p in preds:
        pid = p["ID"]
        if pid not in gold_idx:
            continue
        gquads = gold_idx[pid]["Quadruplet"]
        pquads = p.get("Quadruplet", [])
        gmap = {key(q): q for q in gquads}
        pmap = {key(q): q for q in pquads}
        n_pred += len(pmap)
        n_gold += len(gmap)
        for k, gq in gmap.items():
            if k in pmap:
                pq = pmap[k]
                vg, ag = parse_va(gq["VA"])
                vp, ap_ = parse_va(pq["VA"])
                tp_euclid += ctp_euclid(vg, ag, vp, ap_)
                tp_polar += ctp_polar(vg, ag, vp, ap_)
                n_matched += 1

    def _f1(tp_sum, n_pred, n_gold):
        if n_pred == 0 or n_gold == 0:
            return 0.0, 0.0, 0.0
        cp = tp_sum / n_pred
        cr = tp_sum / n_gold
        cf = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0
        return cp, cr, cf

    cp_e, cr_e, cf_e = _f1(tp_euclid, n_pred, n_gold)
    cp_p, cr_p, cf_p = _f1(tp_polar, n_pred, n_gold)
    return {
        "matched": n_matched,
        "n_pred": n_pred,
        "n_gold": n_gold,
        "cf1_euclid": cf_e,
        "cprec_euclid": cp_e,
        "crec_euclid": cr_e,
        "cf1_polar": cf_p,
        "cprec_polar": cp_p,
        "crec_polar": cr_p,
    }


def spearman(x, y):
    n = len(x)
    rx = [0] * n
    ry = [0] * n
    for r, i in enumerate(sorted(range(n), key=lambda i: x[i])):
        rx[i] = r
    for r, i in enumerate(sorted(range(n), key=lambda i: y[i])):
        ry[i] = r
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def main():
    rows = []
    for domain, gold_path in GOLD.items():
        dom_dir = OUT_ROOT / domain
        if not dom_dir.exists():
            continue
        for run_dir in sorted(dom_dir.iterdir()):
            pred_file = run_dir / "best_thresh_preds.jsonl"
            if not pred_file.exists():
                continue
            res = compute_cf1(pred_file, gold_path)
            res["domain"] = domain
            res["run"] = run_dir.name
            rows.append(res)

    # Report per-domain.
    for domain in GOLD:
        subset = [r for r in rows if r["domain"] == domain]
        if not subset:
            continue
        print(f"\n=== {domain}: {len(subset)} runs ===")
        header = f"{'run':<82s} {'cf1_euclid':>11s} {'cf1_polar':>11s}  diff"
        print(header)
        subset_sorted_e = sorted(subset, key=lambda r: -r["cf1_euclid"])
        subset_sorted_p = sorted(subset, key=lambda r: -r["cf1_polar"])
        rank_e = {r["run"]: i for i, r in enumerate(subset_sorted_e)}
        rank_p = {r["run"]: i for i, r in enumerate(subset_sorted_p)}
        for r in subset_sorted_e:
            diff = r["cf1_polar"] - r["cf1_euclid"]
            print(
                f"{r['run'][:80]:<82s} {r['cf1_euclid']:>11.4f} {r['cf1_polar']:>11.4f}  {diff:+.4f}"
            )
        # Rank-order shift.
        runs = [r["run"] for r in subset]
        re_vals = [rank_e[n] for n in runs]
        rp_vals = [rank_p[n] for n in runs]
        rho = spearman(re_vals, rp_vals)
        # Topk disagreement.
        topk = min(5, len(subset))
        top_e = set(r["run"] for r in subset_sorted_e[:topk])
        top_p = set(r["run"] for r in subset_sorted_p[:topk])
        print(
            f"  Spearman rank(cf1_euclid, cf1_polar) = {rho:+.3f}  "
            f"top-{topk} overlap = {len(top_e & top_p)}/{topk}"
        )
        # How does the Euclidean-best seed rank under polar?
        best_e_run = subset_sorted_e[0]["run"]
        print(
            f"  Euclidean-best run: rank under polar = {rank_p[best_e_run] + 1}/{len(subset)}"
        )
        best_p_run = subset_sorted_p[0]["run"]
        print(
            f"  Polar-best run:     rank under Euclid = {rank_e[best_p_run] + 1}/{len(subset)}"
        )

    out = Path(__file__).resolve().parents[1] / "docs" / "probe_polar_cf1.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {len(rows)} run records to {out}")


if __name__ == "__main__":
    main()
