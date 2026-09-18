# Settings reported in the paper

Source: the camera-ready version of *Opinion as Prior: Supervised VA Prediction
for DimASQP* (ICONIP 2026). Section, table and equation numbers refer to that
version. The last column gives the matching name in `dimasqp_va` where one
exists.

## VA heads (Sec. 4)

| Setting | Paper | Where | `dimasqp_va` |
|---|---|---|---|
| Span representation | mean of the encoder states over the inclusive aspect / opinion span | Eq. 2 | `pool_spans` |
| Implicit (NULL) aspect or opinion | mapped to the `[SEP]` token at position 1, following One-ASQP | Sec. 4 | `NULL_SLOT = 1` |
| Output range | `sigmoid(.) * 8 + 1`, i.e. VA in `[1, 9]` | Sec. 4 | `to_va_range` |
| Position VA | token-level; trained on the tokens of the gold aspect span; a decoded quadruplet reads VA at its aspect start token | Sec. 4 | `PositionVAHead`, `position_targets_from_quads`, `read_position_va` |
| Span-Pair input | `[h_a; h_o; h_a * h_o]` | Eq. 3 | `pair_features` |
| `MLP_prior` | one hidden layer, 256 units | Sec. 4 | `va_hidden=256` |
| `MLP_sp`, `MLP_res` | two hidden layers, 256 and 128 units | Sec. 4 | `va_hidden=256` (second layer `va_hidden // 2`) |
| Residual bound | `4 * tanh(.)`, so the correction lies in `(-4, 4)` | Eq. 5 | `residual_bound=4.0` |
| Gate | learnable `g` in R^2, one value per VA dimension, initialised to 0.5 (`sigmoid(g)` about 0.62) | Sec. 4 | `gate_init=0.5` |
| Final output | `y_prior + sigmoid(g) * delta` (Eq. 6); the sum is then clamped to `[1, 9]`, which the paper states in the text after Eq. 6 | Eq. 6, Sec. 4 | `OpinionGuidedVAHead` |
| Head sizes at H = 768 | Span-Pair MLP 623K parameters; OG adds 197K, 820K in total (about 0.45% of the 184M encoder) | Sec. 5.4 | checked in `tests/test_heads.py` |
| Spans used | gold spans in training, spans predicted by the matrix head at inference | Sec. 4 | caller's choice |

## Loss (Eqs. 7-8)

| Setting | Paper | Where | `dimasqp_va` |
|---|---|---|---|
| `L_VA` | squared error `‖ŷ − y‖²` averaged over the gold quadruplets of a batch; for Position VA, over the tokens of the gold aspect spans | Sec. 4 | `quad_va_loss`, `position_va_loss` |
| Opinion-prior supervision | `L_VA(ŷ_OG, y) + λ_p · L_VA(ŷ_prior, y)` | Eq. 8 | `span_va_loss` |
| λ1 (matrix loss) | 1.0 | Sec. 5.1, Table 1 | `LAMBDA_MATRIX` |
| λ2 (category classifier) | 0.5 | Sec. 5.1, Table 1 | `LAMBDA_CATEGORY` |
| λ3 (VA regression) | 0.5 | Sec. 5.1, Table 1 | `LAMBDA_VA` |
| λp (prior weight) | 0.3 in every reported run; effective prior weight λ3·λp | Sec. 4, 5.1, Table 1 | `PRIOR_WEIGHT` |
| λp sensitivity | single-seed sweep over {0, 0.1, 0.2, 0.3, 0.5, 1.0} on the Restaurant dev set; the selection score stays within 0.772-0.785 and peaks at 1.0 | Sec. 5.3 | - |
| Ablation configurations | Plain-SP, SP+Prior, OG without prior loss, OG full | Table 3 | see `span_va_loss` docstring |

## Training setup (Sec. 5.1, Table 1)

These settings belong to the full extraction model, which is not part of this
package. They are listed so a reimplementation can match the paper.

| Setting | Paper |
|---|---|
| Backbone | One-ASQP token-pair matrix head (EfficientGlobalPointer with RoPE) plus a sentence-level category classifier |
| Encoder | DeBERTa-v3-base |
| Encoder learning rate | 1e-5 |
| Task-head learning rate | 3e-5 |
| Batch size / gradient accumulation | 4 / 8 (effective batch 32) |
| Maximum epochs | 200 |
| Early stopping | 20 epochs |
| Model selection score | micro-F1 of the matrix head on the development set (Sec. 5.3) |
| Optimiser | AdamW, weight decay 0.01, epsilon 1e-8 |
| Gradient clipping | 1.0 |
| LR schedule | reduce on plateau of the development score, factor 0.9, patience 2, floor 1e-5 |
| Adversarial training | FGM on the embedding layer |
| Precision | FP32 |
| Matrix head | multiplicative scoring, head size 256, dropout 0.1 |
| Maximum sequence length | 128 |
| Seeds | five per configuration (the values are not printed in the paper) |
| Extraction threshold | swept on the test split per method and seed, keeping the cF1 maximum; the paper notes this makes absolute scores optimistic (Sec. 5.1, Limitations) |
| Hardware | one NVIDIA RTX 4060 Laptop GPU, about 3-4 hours per run |
| Data | DimABSA 2026 English Restaurant and Laptop: train 2,284 / 4,076, dev 200 / 200, test 1,000 / 1,000 sentences (2,129 / 1,975 test quadruplets) |

## Not stated in the paper

These values come from the authors' implementation and are the defaults in
`dimasqp_va`.

| Setting | Value |
|---|---|
| Dropout inside the VA MLPs | 0.1 |
| Activations | GELU in `MLP_sp`, `MLP_prior`, `MLP_res`; ReLU in `MLP_pos` |
| `MLP_pos` | one hidden layer of 256 units |
| Averaging of `L_VA` | per micro-batch of 4 sentences; gradients are then accumulated over 8 micro-batches |
| Position VA targets | a quadruplet with an implicit aspect and an explicit opinion supervises the NULL slot; where aspect spans overlap, the later quadruplet's VA is used |
| Seeds used in the authors' runs | 7, 42, 66, 123, 2045 |
