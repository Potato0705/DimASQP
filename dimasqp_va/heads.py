"""VA prediction heads compared in the paper (Sec. 4, Eqs. 3-6).

* :class:`PositionVAHead` - the token-level baseline ("Position VA").
* :class:`SpanPairVAHead` - VA from the aspect-opinion span pair (Eq. 3).
  With ``opinion_prior=True`` it also returns an opinion-only prior that the
  training loss can supervise; this is the "SP+Prior" configuration of the
  ablation (Table 3).
* :class:`OpinionGuidedVAHead` - opinion prior plus a gated, bounded
  aspect-opinion residual (Eqs. 4-6).

The span heads take encoder hidden states and quadruplet span indices (see
:mod:`dimasqp_va.pooling`) and return a dict with

``"va"``        ``[B, Q, 2]`` predicted (valence, arousal) in ``[1, 9]``
``"va_prior"``  ``[B, Q, 2]`` opinion-only prior in ``[1, 9]`` (only for heads
                that have one)

Padded quadruplets (``quad_mask == 0``) are returned as zeros.

The heads do not depend on any particular encoder or extractor: at training
time the paper feeds gold spans, at inference the spans decoded by the
extraction head.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .pooling import FIRST_TEXT_INDEX, NULL_SLOT, pair_features, pool_spans

VA_MIN = 1.0
VA_MAX = 9.0


def to_va_range(logits: torch.Tensor) -> torch.Tensor:
    """Map real values onto the VA scale: ``sigmoid(x) * 8 + 1``."""
    return torch.sigmoid(logits) * (VA_MAX - VA_MIN) + VA_MIN


def feed_forward(
    in_features: int,
    hidden_sizes: Sequence[int],
    out_features: int,
    dropout: float,
) -> nn.Sequential:
    """``Linear -> GELU -> Dropout`` for each hidden size, then a final ``Linear``."""
    layers: list[nn.Module] = []
    width = in_features
    for hidden in hidden_sizes:
        layers += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)]
        width = hidden
    layers.append(nn.Linear(width, out_features))
    return nn.Sequential(*layers)


def _apply_mask(values: torch.Tensor, quad_mask: torch.Tensor) -> torch.Tensor:
    return values * quad_mask.unsqueeze(-1).to(values.dtype)


# --------------------------------------------------------------------------
# Token-level baseline
# --------------------------------------------------------------------------
class PositionVAHead(nn.Module):
    """Token-level VA baseline ("Position VA", Sec. 4).

    Predicts ``sigmoid(MLP_pos(h_t)) * 8 + 1`` at every token ``t``, with
    ``MLP_pos = Linear(H, 256) -> ReLU -> Dropout -> Linear(256, 2)``. It is
    trained on the tokens of the gold aspect span (see
    :func:`dimasqp_va.losses.position_targets_from_quads`), and a decoded
    quadruplet reads its VA at the aspect start token
    (:func:`read_position_va`), so quadruplets that share an aspect share a VA.
    """

    def __init__(self, hidden_size: int, va_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.hidden = nn.Linear(hidden_size, va_hidden)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(va_hidden, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``[B, L, H] -> [B, L, 2]`` per-token (V, A) in ``[1, 9]``."""
        x = F.relu(self.hidden(hidden_states))
        x = self.dropout(x)
        return to_va_range(self.output(x))


def read_position_va(
    token_va: torch.Tensor,
    quad_spans: torch.Tensor,
    quad_mask: torch.Tensor,
    null_slot: int = NULL_SLOT,
    first_text_index: int = FIRST_TEXT_INDEX,
) -> torch.Tensor:
    """Read quadruplet VA from a token-level prediction.

    Each quadruplet takes the VA predicted at its aspect start token; an
    implicit aspect reads the ``NULL`` slot. The result is clamped to
    ``[1, 9]`` and padded quadruplets are zeroed. As in
    :func:`~dimasqp_va.pooling.pool_spans`, padded quadruplets
    (``quad_mask == 0``) may carry any indices: they are never used to index
    ``token_va``.

    Args:
        token_va: ``[B, L, 2]`` output of :class:`PositionVAHead`.
        quad_spans: ``[B, Q, 4]`` inclusive span indices.
        quad_mask: ``[B, Q]``.

    Returns:
        ``[B, Q, 2]``.
    """
    aspect_start = quad_spans[:, :, 0]
    # Padded rows are sent to the NULL slot before the gather, so out-of-range
    # padding indices cannot fail; _apply_mask zeroes them afterwards.
    explicit = quad_mask.bool() & (aspect_start >= first_text_index)
    index = torch.where(
        explicit,
        aspect_start,
        torch.full_like(aspect_start, null_slot),
    )
    gathered = torch.gather(token_va, 1, index.unsqueeze(-1).expand(-1, -1, token_va.shape[-1]))
    return _apply_mask(gathered.clamp(VA_MIN, VA_MAX), quad_mask)


# --------------------------------------------------------------------------
# Span-level heads
# --------------------------------------------------------------------------
class SpanPairVAHead(nn.Module):
    """Span-Pair VA (Eq. 3): ``sigmoid(MLP_sp([h_a; h_o; h_a * h_o])) * 8 + 1``.

    ``MLP_sp`` has hidden layers of 256 and 128 units with GELU and dropout.

    Args:
        hidden_size: encoder hidden size ``H``.
        va_hidden: width of the first hidden layer (the second is half of it).
        dropout: dropout rate inside the MLPs.
        opinion_prior: also compute the opinion-only prior
            ``sigmoid(MLP_prior(h_o)) * 8 + 1`` (Eq. 4) and return it as
            ``"va_prior"``. The prior does not enter ``"va"``; it only gives the
            loss something to supervise (the SP+Prior ablation, Table 3).
    """

    def __init__(
        self,
        hidden_size: int,
        va_hidden: int = 256,
        dropout: float = 0.1,
        opinion_prior: bool = False,
    ):
        super().__init__()
        self.mlp = feed_forward(3 * hidden_size, (va_hidden, va_hidden // 2), 2, dropout)
        self.prior_mlp = (
            feed_forward(hidden_size, (va_hidden,), 2, dropout) if opinion_prior else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        quad_spans: torch.Tensor,
        quad_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h_aspect, h_opinion = pool_spans(hidden_states, quad_spans, quad_mask)
        out = {"va": _apply_mask(to_va_range(self.mlp(pair_features(h_aspect, h_opinion))), quad_mask)}
        if self.prior_mlp is not None:
            out["va_prior"] = _apply_mask(to_va_range(self.prior_mlp(h_opinion)), quad_mask)
        return out


class OpinionGuidedVAHead(nn.Module):
    """Opinion-Guided VA (Eqs. 4-6).

    ::

        y_prior = sigmoid(MLP_prior(h_o)) * 8 + 1                    (Eq. 4)
        delta   = bound * tanh(MLP_res([h_a; h_o; h_a * h_o]))       (Eq. 5)
        y_OG    = y_prior + sigmoid(g) * delta                       (Eq. 6)
        y_OG    = clamp(y_OG, 1, 9)             (Sec. 4, text after Eq. 6)

    Unlike the sigmoid-bounded heads, the sum in Eq. 6 can leave ``[1, 9]``,
    so the paper clamps it to that range; the clamp is stated in the text,
    not in Eq. 6 itself.

    ``MLP_prior`` has one hidden layer of 256 units, ``MLP_res`` two of 256 and
    128. The gate ``g`` holds one learnable value per VA dimension, initialised
    to 0.5 (so ``sigmoid(g)`` starts near 0.62), and ``bound`` is 4, half of
    the 8-point VA range. The prior always enters with coefficient 1.

    Both ``"va"`` (``y_OG``) and ``"va_prior"`` are returned; opinion-prior
    supervision (Eq. 8) adds a loss on the prior, see
    :func:`dimasqp_va.losses.span_va_loss`.
    """

    def __init__(
        self,
        hidden_size: int,
        va_hidden: int = 256,
        dropout: float = 0.1,
        gate_init: float = 0.5,
        residual_bound: float = 4.0,
    ):
        super().__init__()
        self.prior_mlp = feed_forward(hidden_size, (va_hidden,), 2, dropout)
        self.residual_mlp = feed_forward(3 * hidden_size, (va_hidden, va_hidden // 2), 2, dropout)
        self.gate = nn.Parameter(torch.full((2,), float(gate_init)))
        self.residual_bound = residual_bound

    def gate_values(self) -> torch.Tensor:
        """``sigmoid(g)`` for (valence, arousal)."""
        return torch.sigmoid(self.gate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        quad_spans: torch.Tensor,
        quad_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h_aspect, h_opinion = pool_spans(hidden_states, quad_spans, quad_mask)
        va_prior = to_va_range(self.prior_mlp(h_opinion))
        delta = torch.tanh(self.residual_mlp(pair_features(h_aspect, h_opinion))) * self.residual_bound
        va = torch.clamp(va_prior + self.gate_values() * delta, VA_MIN, VA_MAX)
        return {"va": _apply_mask(va, quad_mask), "va_prior": _apply_mask(va_prior, quad_mask)}
