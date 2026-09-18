# Opinion as Prior: VA heads for DimASQP

Code for "Opinion as Prior: Supervised VA Prediction for DimASQP" (ICONIP 2026, Springer CCIS).

**Authors:** Yaoshuo Wu, Haiqin Yang, Zhong Ming

**Status:** The paper has been accepted. The proceedings are not published
yet, so it has no DOI or page numbers. The citation below will be updated when
they are available.

## The task and the contribution

Dimensional Aspect Sentiment Quad Prediction (DimASQP) extracts quadruplets
`(aspect, category, opinion, <valence, arousal>)`, where valence and arousal
(VA) are continuous scores on a `[1, 9]` scale. The official Continuous F1
metric discounts every correctly extracted quadruplet by the Euclidean distance
between its predicted and gold VA, so VA quality is part of the task objective.
Existing token-pair extractors predict VA at token positions, which does not
match quadruplet-level scoring and lets the opinion expression influence VA
only implicitly. The paper proposes **opinion-prior supervision**, an auxiliary
loss that trains a VA prior computed from the opinion span directly against
the gold VA, and realises it as the **Opinion-Guided (OG) VA head**, which adds
a gated, bounded aspect-opinion residual to that prior. With the extraction
architecture held fixed, a controlled ablation attributes the VA gain to the
supervision signal rather than to the prior-residual decomposition.

## What this repository contains

* `dimasqp_va/`: the VA heads, span pooling and VA losses as a standalone
  PyTorch package. It depends only on PyTorch.
* `tests/`: unit tests for the package.
* `examples/attach_to_encoder.py`: a runnable example that attaches every head
  to a small encoder and trains it for a few steps on synthetic data.
* `configs/paper_hyperparameters.md`: the settings reported in the paper, with
  section, table and equation references.

How the code maps to the paper (section and equation numbers refer to the
camera-ready version):

| Paper | Code |
|---|---|
| Span mean pooling (Eq. 2) | `pool_spans`, `pair_features` |
| Position VA, the token-level baseline (Sec. 4) | `PositionVAHead`, `position_targets_from_quads`, `read_position_va` |
| Span-Pair VA (Eq. 3) | `SpanPairVAHead` |
| Opinion prior (Eq. 4) | `OpinionGuidedVAHead`, and `SpanPairVAHead(opinion_prior=True)` for the SP+Prior ablation |
| Bounded residual and gate (Eqs. 5-6), clamp to `[1, 9]` (Sec. 4) | `OpinionGuidedVAHead` |
| VA loss and opinion-prior supervision (Eqs. 7-8) | `quad_va_loss`, `position_va_loss`, `span_va_loss`, `weighted_total_loss` |

The Opinion-Guided head computes

```
y_prior = sigmoid(MLP_prior(h_o)) * 8 + 1                  (Eq. 4)
delta   = 4 * tanh(MLP_res([h_a; h_o; h_a * h_o]))         (Eq. 5)
y_OG    = y_prior + sigmoid(g) * delta                     (Eq. 6)
y_OG    = clamp(y_OG, 1, 9)                                (Sec. 4, text after Eq. 6)
```

The sum in Eq. 6 can leave `[1, 9]`, so the paper clamps it to that range.
Opinion-prior supervision trains the head with
`L_VA(y_OG, y) + lambda_p * L_VA(y_prior, y)` (Eq. 8), where `L_VA` is the
squared Euclidean error averaged over the gold quadruplets.

## What this repository does not contain

* **The span extractor.** The paper uses One-ASQP
  ([Zhou et al., Findings of ACL 2023](https://aclanthology.org/2023.findings-acl.777/))
  as its extraction backbone: a token-pair matrix head (EfficientGlobalPointer
  with rotary position embeddings) and a sentence-level category classifier.
  The paper holds it fixed and changes only the VA head. That extractor is third-party work, and
  its code is not released here. The heads in this package need only the
  encoder's hidden states and the span indices of each quadruplet, so they can
  be attached to any span extractor.
* **The DimABSA data and the official evaluation script.** Both come from the
  SemEval-2026 Task 3 (DimABSA) organisers and are distributed under their
  terms in the official repository,
  [github.com/DimABSA/DimABSA2026](https://github.com/DimABSA/DimABSA2026)
  (data in `task-dataset/`, scorer in `evaluation_script/`). The dataset paper
  is [Lee et al., ACL 2026](https://aclanthology.org/2026.acl-long.1881/).
  This repository contains no DimABSA text, labels or predictions.
* **A full training pipeline and checkpoints.** Without the extractor and the
  data, this package cannot reproduce the paper's tables end to end. The paper
  also notes that the original checkpoints were not retained (Sec. 5.4).

## Install

Requires Python 3.9 or later and PyTorch 1.13 or later. The tests were run
with Python 3.12 and PyTorch 2.10 on CPU.

```bash
git clone https://github.com/Potato0705/DimASQP.git
cd DimASQP
pip install -e .            # the package
pip install -e ".[test]"    # the package and pytest
```

## Attaching the heads to a span extractor

The span heads take three tensors:

| Argument | Shape | Meaning |
|---|---|---|
| `hidden_states` | `[B, L, H]` | output of any token encoder (DeBERTa-v3-base, `H = 768`, in the paper) |
| `quad_spans` | `[B, Q, 4]`, long | `(aspect_start, aspect_end, opinion_start, opinion_end)` for each quadruplet, as inclusive token indices |
| `quad_mask` | `[B, Q]` | 1 for a real quadruplet, 0 for padding (padded rows may hold any indices) |

They return a dict with `"va"`, the predicted `(valence, arousal)` of shape
`[B, Q, 2]` in `[1, 9]`, and, for heads that have an opinion prior,
`"va_prior"` of the same shape. Padded quadruplets are returned as zeros.
As in the paper, pass gold spans during training and the spans decoded by your
extractor at inference.

```python
import torch
from dimasqp_va import OpinionGuidedVAHead, span_va_loss

head = OpinionGuidedVAHead(hidden_size=768)          # DeBERTa-v3-base in the paper

hidden_states = torch.randn(2, 16, 768)              # [B, L, H] from your encoder
quad_spans = torch.tensor([                          # [B, Q, 4] inclusive token indices
    [[2, 3, 5, 5], [1, 1, 8, 9]],                    #   (aspect_start, aspect_end, opinion_start, opinion_end)
    [[4, 4, 6, 7], [-1, -1, -1, -1]],                #   start < 2 means implicit (NULL); rows with quad_mask 0 are padding
])
quad_mask = torch.tensor([[1., 1.], [1., 0.]])      # [B, Q] 1 = real quadruplet
gold_va = torch.tensor([[[7.2, 6.1], [3.0, 5.5]], [[6.0, 4.2], [0., 0.]]])

out = head(hidden_states, quad_spans, quad_mask)     # {"va": [B, Q, 2], "va_prior": [B, Q, 2]}
loss_va = span_va_loss(out, gold_va, quad_mask, prior_weight=0.3)   # Eq. 8
```

The Position baseline works on tokens instead. Continuing the example above:

```python
from dimasqp_va import PositionVAHead, position_targets_from_quads, position_va_loss, read_position_va

pos_head = PositionVAHead(hidden_size=768)
token_va = pos_head(hidden_states)                                            # [B, L, 2]
targets, token_mask = position_targets_from_quads(quad_spans, gold_va, quad_mask, seq_len=16)
loss_pos = position_va_loss(token_va, targets, token_mask)                    # training
quad_va = read_position_va(token_va, quad_spans, quad_mask)                   # VA at each aspect start token
```

In a full model, combine the VA loss with the extractor's own losses as in
Eq. 7: `weighted_total_loss(loss_matrix, loss_category, loss_va)`, which uses
the paper's weights 1.0, 0.5 and 0.5.

**Token layout.** Following One-ASQP, the paper maps an implicit (`NULL`)
aspect or opinion to the token at position 1: index 0 is the encoder's leading
special token, index 1 stands for `NULL`, and the text starts at index 2. A
span whose start index is below 2 is therefore pooled from index 1. For an
implicit opinion the prior has no opinion wording to read (Sec. 4). The span
heads assume this layout; `pool_spans`, `read_position_va` and
`position_targets_from_quads` accept `null_slot` and `first_text_index`
keyword arguments if you call them directly.

**Ablation configurations** (Table 3 of the paper):

| Configuration | Head | `prior_weight` |
|---|---|---|
| Span-Pair (Plain-SP) | `SpanPairVAHead(H)` | ignored |
| Span-Pair + Prior Loss (SP+Prior) | `SpanPairVAHead(H, opinion_prior=True)` | 0.3 |
| OG without prior loss | `OpinionGuidedVAHead(H)` | 0.0 |
| Opinion-Guided (full) | `OpinionGuidedVAHead(H)` | 0.3 |

## Running the tests and the example

From the repository root, after `pip install -e ".[test]"`:

```bash
python -m pytest
python examples/attach_to_encoder.py
```

The tests check output shapes, masking of padded and implicit spans, gradient
flow to every parameter, the output range, residual bound and gate
initialisation of each head, the head parameter counts reported in Sec. 5.4
(623K for the Span-Pair MLP and 820K for the OG head at `H = 768`), the
loss definitions, and that the code snippets in this README run.

The example trains the Position, Span-Pair, SP+Prior and Opinion-Guided heads
on top of a small, randomly initialised Transformer encoder with synthetic
quadruplets. It runs on CPU in a few seconds and downloads nothing. Its
docstring shows where a pre-trained encoder such as DeBERTa-v3-base would plug
in.

## Paper settings

[`configs/paper_hyperparameters.md`](configs/paper_hyperparameters.md) lists
every setting the paper reports, with its source, and the matching name in
`dimasqp_va`. The main ones: DeBERTa-v3-base encoder; `MLP_prior` with one
hidden layer of 256 units, `MLP_sp` and `MLP_res` with hidden layers of 256
and 128 units; residual bound 4; gate initialised to 0.5; loss weights
`lambda_1 = 1.0`, `lambda_2 = lambda_3 = 0.5` and `lambda_p = 0.3`; five seeds
per configuration. Implementation details that the paper does not state are
listed separately at the end of that file.

## Main result

All numbers below are from the camera-ready paper (Secs. 5.1-5.3 and 6,
Tables 2 and 3). Experiments use the DimABSA 2026 English Restaurant and
Laptop test sets, with the extraction architecture held fixed across heads.

**VA error against the token-level baseline.** The paper scores all three heads
on the quadruplets that every one of them extracts correctly (exact match on
aspect, category and opinion), taking the per-seed intersection of their
matched sets (mean size 1,062 on Restaurant and 255 on Laptop). On this common
set, the Opinion-Guided head reduces the mean Euclidean VA error of the
token-level Position baseline by 3.4% on Restaurant and 7.5% on Laptop,
averaged over five seeds.

Mean Euclid on the per-seed common matched set (paper Sec. 5.2):

| Domain | Euclid, Position | Euclid, Span-Pair | Euclid, Opinion-Guided | OG vs. Position | Seed-level p | Bootstrap interval, OG minus Position |
|---|---|---|---|---|---|---|
| Restaurant | 0.872 | 0.865 | 0.842 | -3.4% | 0.070 | -0.030 [-0.050, -0.011] |
| Laptop | 0.804 | 0.746 | 0.744 | -7.5% | < 0.001 | -0.061 [-0.098, -0.027] |

Euclid is the mean distance between predicted and gold VA. These values are
not the ones in Table 2 of the paper, which scores each head on its own
matched set (Restaurant 0.884 / 0.877 / 0.852, Laptop 0.839 / 0.807 / 0.772
for Position / Span-Pair / Opinion-Guided); the paper treats the common-set
comparison above as the primary estimate. The p-values compare OG with
Position using one-tailed paired t-tests over the five seeds, without
correction for multiple comparisons. The intervals come from a bootstrap that
resamples sentences (10^4 replicates).

**What drove the gain.** The ablation on Restaurant (five seeds; each
configuration scored on its own matched set) crosses the prior-residual
decomposition with the prior loss:

| Configuration | Decomposition | Prior loss | Euclid | Change vs. Plain-SP |
|---|---|---|---|---|
| Span-Pair (Plain-SP) | no | no | 0.877 | - |
| Span-Pair + Prior Loss | no | yes | 0.851 | -3.1% |
| OG without prior loss | yes | no | 0.856 | -2.4% |
| Opinion-Guided (full) | yes | yes | 0.852 | -2.9% |

Adding the prior loss to Span-Pair gives nearly the same Euclid as the full OG
head (0.851 vs. 0.852), while the decomposition without the prior loss gives a
smaller reduction. cF1 stays between 0.558 and 0.567 across the four
configurations. The paper therefore attributes the gain mainly to the
supervision signal and treats the Opinion-Guided head as one way to realise
it.

**Scope.** The claim is about VA quality under controlled extraction, not about
the complete DimASQP task:

* cF1 moves by less than one point between heads; closing cF1 gaps would need
  better extraction, which the VA head does not address.
* The Opinion-Guided head's margin over Span-Pair is small and depends on the
  domain. On the Laptop common set the two are nearly equal (0.744 vs. 0.746,
  p = 0.40).
* Extraction thresholds were chosen on the test split for each method and seed,
  so absolute cF1 values, and the matched sets they determine, are optimistic.
  The paper's common-set comparison scores every head on the same quadruplets
  within each seed to keep the between-head comparison fair.
* The results cover two English domains with one extractive backbone.

Full results, including cF1, valence and arousal MAE, and an error analysis by
opinion length and gold valence, are in the paper.

## Citation

The proceedings are not published yet. Volume, pages and DOI will be added
when they are.

```bibtex
@inproceedings{wu2026opinionprior,
  author    = {Wu, Yaoshuo and Yang, Haiqin and Ming, Zhong},
  title     = {Opinion as Prior: Supervised {VA} Prediction for {DimASQP}},
  booktitle = {Neural Information Processing (ICONIP 2026)},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2026},
  note      = {Accepted, to appear}
}
```

Please also cite the work the paper builds on:

* One-ASQP: Zhou et al., *A Unified One-Step Solution for Aspect Sentiment Quad
  Prediction*, Findings of ACL 2023.
  [aclanthology.org/2023.findings-acl.777](https://aclanthology.org/2023.findings-acl.777/)
* DimABSA: Lee et al., *DimABSA: Building Multilingual and Multidomain Datasets
  for Dimensional Aspect-Based Sentiment Analysis*, ACL 2026.
  [aclanthology.org/2026.acl-long.1881](https://aclanthology.org/2026.acl-long.1881/)

## Licence

The code in this repository is released under the [MIT License](LICENSE).
The licence covers only the code in this repository. It does not cover
One-ASQP, the DimABSA data or evaluation script, or the paper itself.
