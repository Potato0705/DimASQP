"""VA training losses (paper Eqs. 7-8).

The VA loss is the squared Euclidean error ``||y_hat - y||^2`` averaged over the
gold quadruplets of a batch (for Position VA, over the tokens of the gold
aspect spans instead). Opinion-prior supervision adds the same loss on the
opinion-only prior with weight ``lambda_p``:

    L_VA^OG = L_VA(y_OG, y) + lambda_p * L_VA(y_prior, y)            (Eq. 8)

and the full objective is

    L = lambda_1 * L_matrix + lambda_2 * L_cls + lambda_3 * L_VA      (Eq. 7)

with ``lambda_1 = 1.0``, ``lambda_2 = lambda_3 = 0.5`` and ``lambda_p = 0.3`` in
every reported run. ``L_matrix`` and ``L_cls`` belong to the extraction model,
which is not part of this package; :func:`weighted_total_loss` only combines
values you compute yourself.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch

from .pooling import FIRST_TEXT_INDEX, NULL_SLOT

#: Weights used for every run in the paper (Sec. 5.1, Table 1).
LAMBDA_MATRIX = 1.0
LAMBDA_CATEGORY = 0.5
LAMBDA_VA = 0.5
PRIOR_WEIGHT = 0.3


def masked_va_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Squared VA error summed over (V, A) and averaged over the unmasked items.

    Args:
        pred: ``[..., N, 2]`` predicted (V, A).
        target: ``[..., N, 2]`` gold (V, A).
        mask: ``[..., N]`` with 1 for items that count and 0 otherwise.

    Returns:
        Scalar ``sum(mask * ||pred - target||^2) / max(sum(mask), 1)``.
    """
    weights = mask.unsqueeze(-1).to(pred.dtype)
    diff = (pred - target) * weights
    return (diff ** 2).sum() / weights.sum().clamp(min=1.0)


def quad_va_loss(pred_va: torch.Tensor, gold_va: torch.Tensor, quad_mask: torch.Tensor) -> torch.Tensor:
    """``L_VA`` for span heads: mean squared Euclidean error over gold quadruplets."""
    return masked_va_mse(pred_va, gold_va, quad_mask)


def span_va_loss(
    outputs: Mapping[str, torch.Tensor],
    gold_va: torch.Tensor,
    quad_mask: torch.Tensor,
    prior_weight: float = PRIOR_WEIGHT,
) -> torch.Tensor:
    """VA loss for a span head, with opinion-prior supervision when available.

    Returns ``L_VA(outputs["va"])`` plus ``prior_weight * L_VA(outputs["va_prior"])``
    when the head returned a prior and ``prior_weight > 0`` (Eq. 8). This one
    function covers the four configurations of the ablation (Table 3):

    ==============================  ==========================================  ============
    configuration                   head                                        prior_weight
    ==============================  ==========================================  ============
    Span-Pair (Plain-SP)            ``SpanPairVAHead(H)``                       any
    Span-Pair + Prior Loss          ``SpanPairVAHead(H, opinion_prior=True)``   0.3
    OG w/o Prior Loss               ``OpinionGuidedVAHead(H)``                  0.0
    Opinion-Guided (full)           ``OpinionGuidedVAHead(H)``                  0.3
    ==============================  ==========================================  ============
    """
    loss = quad_va_loss(outputs["va"], gold_va, quad_mask)
    va_prior: Optional[torch.Tensor] = outputs.get("va_prior")
    if va_prior is not None and prior_weight > 0:
        loss = loss + prior_weight * quad_va_loss(va_prior, gold_va, quad_mask)
    return loss


def position_va_loss(token_va: torch.Tensor, token_targets: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """``L_VA`` for Position VA: mean squared Euclidean error over supervised tokens."""
    return masked_va_mse(token_va, token_targets, token_mask)


@torch.no_grad()
def position_targets_from_quads(
    quad_spans: torch.Tensor,
    quad_va: torch.Tensor,
    quad_mask: torch.Tensor,
    seq_len: int,
    null_slot: int = NULL_SLOT,
    first_text_index: int = FIRST_TEXT_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level targets for :class:`~dimasqp_va.heads.PositionVAHead`.

    Every token of an explicit gold aspect span receives that quadruplet's
    (V, A). A quadruplet with an implicit aspect and an explicit opinion puts
    its target on the ``NULL`` slot; one with both spans implicit gives no
    token target. Where aspect spans overlap, the later quadruplet wins.

    Args:
        quad_spans: ``[B, Q, 4]`` inclusive span indices.
        quad_va: ``[B, Q, 2]`` gold (V, A).
        quad_mask: ``[B, Q]``.
        seq_len: number of token positions ``L``.

    Returns:
        ``(token_targets [B, L, 2], token_mask [B, L])``.
    """
    batch, num_quads = quad_spans.shape[:2]
    device = quad_spans.device
    positions = torch.arange(seq_len, device=device).view(1, seq_len)
    targets = torch.zeros(batch, seq_len, 2, dtype=quad_va.dtype, device=device)
    mask = torch.zeros(batch, seq_len, dtype=quad_va.dtype, device=device)

    for q in range(num_quads):
        a_start = quad_spans[:, q, 0:1]
        a_end = quad_spans[:, q, 1:2]
        o_start = quad_spans[:, q, 2:3]
        valid = quad_mask[:, q : q + 1].bool()
        explicit_aspect = a_start >= first_text_index
        on_aspect = explicit_aspect & (positions >= a_start) & (positions <= a_end)
        on_null = ~explicit_aspect & (o_start >= first_text_index) & (positions == null_slot)
        selected = (on_aspect | on_null) & valid  # [B, L]
        targets = torch.where(selected.unsqueeze(-1), quad_va[:, q : q + 1, :], targets)
        mask = torch.where(selected, torch.ones_like(mask), mask)
    return targets, mask


def weighted_total_loss(
    loss_matrix: torch.Tensor,
    loss_category: torch.Tensor,
    loss_va: torch.Tensor,
    lambda_matrix: float = LAMBDA_MATRIX,
    lambda_category: float = LAMBDA_CATEGORY,
    lambda_va: float = LAMBDA_VA,
) -> torch.Tensor:
    """Eq. (7): ``lambda_1 * L_matrix + lambda_2 * L_cls + lambda_3 * L_VA``."""
    return lambda_matrix * loss_matrix + lambda_category * loss_category + lambda_va * loss_va
