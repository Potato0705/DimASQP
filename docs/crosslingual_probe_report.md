# Cross-Lingual DimASQP Direction — Go/No-Go Probe Report

**Date**: 2026-04-24
**Scope**: Validate the "latent universal emotion + per-language observation channel"
re-framing before committing to an EMNLP Long submission.
**Scripts**: `tools/probe_crosslingual.py`, `tools/probe_polar_cf1.py`
**Raw outputs**: `docs/probe_crosslingual_report.json`, `docs/probe_polar_cf1.json`

## Summary

| Probe | Hypothesis under test | Outcome |
|---|---|---|
| 1 | Per-language VA distributions differ substantially, motivating a language-specific observation channel $g_\ell$ | **Confirmed, strongly** |
| 2 | rus/tat/ukr are three *independent* annotations of the same semantics, providing identifiability of latent emotion vs measurement bias | **Refuted, decisively** |
| 3 | Euclidean cF1 geometrically misaligned with circumplex structure → polar metric reshuffles leaderboards | **Refuted** |

**Net**: 1 of 3 supporting signals. The distinctive technical weapon (identifiability
from parallel data) is dead at the data level. The "polar metric" side-quest is dead.
The direction collapses to a per-language calibration story, which is not EMNLP-Long
level of novelty on its own.

---

## Probe 1 — Per-language VA distributions (train, quad-level)

| Lang-Domain | N | V mean | V std | V range₉₅ | A mean | A std | A range₉₅ | corr(V,A) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eng_restaurant | 3659 | 6.22 | 1.83 | 5.62 | 6.84 | 1.13 | 3.25 | +0.551 |
| eng_laptop | 5773 | 5.94 | 1.76 | 5.25 | 6.67 | 1.03 | 3.12 | +0.598 |
| zho_restaurant | 8523 | 5.81 | **0.99** | **3.17** | 5.76 | **0.65** | **2.05** | +0.607 |
| zho_laptop | 6502 | 5.72 | **1.14** | **3.47** | 5.62 | **0.67** | **2.13** | +0.402 |
| jpn_hotel | 2846 | 6.32 | 1.20 | 4.00 | 6.39 | **0.56** | **1.88** | +0.483 |
| rus_restaurant | 2487 | 6.58 | 1.80 | 5.93 | 6.87 | 1.02 | 3.30 | +0.298 |
| tat_restaurant | 2487 | 6.58 | 1.80 | 5.93 | 6.87 | 1.02 | 3.30 | +0.298 |
| ukr_restaurant | 2487 | 6.58 | 1.80 | 5.93 | 6.87 | 1.02 | 3.30 | +0.298 |

Pairwise Kolmogorov-Smirnov on V and A marginals (train):

- V-marginal KS(eng_rest, zho_rest) = **0.448**, KS(eng_rest, zho_lap) = 0.431, KS(rus, zho_rest) = **0.590**
- A-marginal KS(eng_rest, zho_lap) = **0.551**, KS(rus, zho_lap) = **0.642**
- V/A marginals of rus/tat/ukr are identical (KS = 0.000) — data-structural artefact, see Probe 2.

**Interpretation.** East-Asian (zho/jpn) VA annotations are compressed by ~40–50% on
both V and A compared with eng/rus. This is entirely consistent with the cross-cultural
affect literature (Russell 1991; Kitayama et al. 2006) and the DimABSA paper's own
qualitative observation. The V–A correlation also varies widely (+0.30 on rus vs
+0.60 on zho_rest), suggesting the joint structure — not just marginals — is
language-specific. A shared-scale assumption across languages is empirically indefensible.

**Verdict**: the descriptive motivation for a language-specific observation channel
is **solid**. This alone gives us an "analysis" contribution but does not by itself
pay for an EMNLP method paper.

---

## Probe 2 — rus/tat/ukr three-view parallel structure (all splits)

| Split | Common IDs | Aligned quads | Mismatched VA strings | Mismatched categories | Mean VA ℓ₂ (rus,tat) | Max VA ℓ₂ (rus,tat) | Frac non-zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 1240 | 2487 | **0** | **0** | **0.0000** | **0.000** | **0.000** |
| dev | 48 | 102 | **0** | **0** | **0.0000** | **0.000** | **0.000** |
| test | 630 | 1310 | **0** | **0** | **0.0000** | **0.000** | **0.000** |

Totals: 3899 aligned quads across the three languages; **zero** show any VA difference
or any category difference. Pairwise distances rus↔tat, rus↔ukr, tat↔ukr are all
exactly 0.

**Interpretation.** tat/ukr were created by translating rus *text* and **copying the
VA labels verbatim** — not by re-annotating. The DimABSA paper's "validated by native
speakers" describes span-level surface verification, not independent VA assessment.

**Consequence for the latent-variable framing.** The identifiability argument I wrote
in the previous message required *three independent views of the latent state*
$z$, each passed through a distinct observation channel $g_\ell$. What we actually
have is *one annotation* paired with *three text views*. That is a shared-label
multi-view problem (well-studied), not a multi-annotation problem. The Slavic/Turkic
triple carries **zero bits of information** about per-language measurement bias.

This is the probe's killer finding. The theoretical pillar of the proposed framing
does not survive contact with the data. Any "identifiability theorem" we would write
would have to assume re-annotation that simply does not exist in DimABSA.

---

## Probe 3 — Euclidean cF1 vs polar-aware cF1 on existing eng predictions

| Domain | Runs | Spearman ρ(euclid, polar) | Top-5 overlap | Best run under polar == best under euclid? |
|---|---:|---:|---:|---|
| eng_restaurant | 25 | **+0.995** | 5/5 | Yes |
| eng_laptop | 15 | **+1.000** | 5/5 | Yes |

Polar cF1 uniformly sits +0.003 to +0.009 above Euclidean cF1 across all 40 runs,
with **no rank reshuffling**.

**Interpretation.** On typical DeBERTa-based extractor outputs, VA predictions cluster
close enough to gold that radial-vs-angular weighting does not materially change the
verdict. To see a difference we would need to compare *structurally distinct* system
families (e.g., an LLM system that over-shoots to extremes vs a conservative regressor).
We do not have those predictions; generating them is non-trivial.

**Verdict.** The "cF1 is geometrically wrong" section is not defensible with a
single-family set of predictions. Even as a side contribution it would be weak.

---

## What this means for the EMNLP pitch

The proposed framing had three pillars:

1. **Descriptive motivation** — languages really do annotate differently. ✅ Supported.
2. **Identifiability theorem via parallel data** — lets the method outperform
   per-language training without needing external anchors. ❌ Not available.
3. **Metric analysis exposing benchmark flaw** — companion evaluation contribution.
   ❌ Not reproducible on extant predictions.

Without (2), the method reduces to "per-language calibration head on top of a shared
encoder", which is an incremental architectural tweak. LogSigma (1st place on 5
datasets in SemEval-2026) already achieves a per-language effect via homoscedastic
uncertainty weighting + per-language encoders. Reviewers will not find our version
of the same idea 耳目一新.

## Options from here

| Option | Cost | Risk | Novelty level |
|---|---|---|---|
| **A. Collect 100–200 cross-lingual re-annotations** to instantiate identifiability honestly | High (annotation $$, 2–4 weeks) | Medium | High — becomes benchmark + method paper |
| **B. Pivot to a non-identifiability story** (e.g., compositional VA, counterfactual robustness, implicit-opinion cross-lingual) | Low–Medium | Medium | Depends on chosen angle |
| **C. Downgrade target to ACL/ARR workshop or ICONIP/PACLIC** and publish the analysis + calibration method as-is | Low | Low | Low–Medium (honest paper, less prestigious venue) |
| **D. Abandon the cross-lingual angle** and stay with the opinion-prior ICONIP paper | Zero | Zero | N/A |

## Recommendation

Given the user's stated target (EMNLP-level venue) and unwillingness to ship a "small
innovation", the evidence from these three probes argues for either (A) — pay real
annotation cost to make the story right — or (B) — pivot the technical angle off the
identifiability line entirely.

I would **not** recommend proceeding with the latent-emotion direction as originally
framed. The data does not cooperate.
