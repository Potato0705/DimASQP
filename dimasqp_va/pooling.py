"""Span pooling for quadruplet-level VA prediction (paper Eq. 2).

Every VA head in this package reads a batch of quadruplets through the same
interface:

``hidden_states``  float tensor ``[B, L, H]`` from any token encoder.

``quad_spans``     long tensor ``[B, Q, 4]`` holding, for each quadruplet,
                   ``(aspect_start, aspect_end, opinion_start, opinion_end)``
                   as *inclusive* token indices into ``hidden_states``.

``quad_mask``      tensor ``[B, Q]`` with 1 for a real quadruplet and 0 for
                   padding.

Token layout. The paper follows One-ASQP in reserving a slot for implicit
(``NULL``) aspects and opinions: index 0 holds the encoder's leading special
token, index 1 holds a reserved token that stands for ``NULL``, and the text
starts at index 2. A span whose start index is below 2 is therefore treated as
implicit and pooled from index 1 alone. Padding quadruplets may carry any
indices (for example ``-1``); they are masked out and pool to zeros.
"""

from __future__ import annotations

import torch

#: Index of the reserved token that represents an implicit (NULL) span.
NULL_SLOT = 1

#: First token index that belongs to the input text.
FIRST_TEXT_INDEX = 2


def _span_mask(
    starts: torch.Tensor,
    ends: torch.Tensor,
    positions: torch.Tensor,
    valid_quads: torch.Tensor,
    null_slot: int,
    first_text_index: int,
) -> torch.Tensor:
    """Boolean ``[B, Q, L]`` mask of the tokens that make up each span."""
    starts = starts.unsqueeze(-1)  # [B, Q, 1]
    ends = ends.unsqueeze(-1)
    explicit = starts >= first_text_index
    inside = (positions >= starts) & (positions <= ends) & explicit
    implicit = ~explicit & (positions == null_slot)
    return (inside | implicit) & valid_quads


def pool_spans(
    hidden_states: torch.Tensor,
    quad_spans: torch.Tensor,
    quad_mask: torch.Tensor,
    null_slot: int = NULL_SLOT,
    first_text_index: int = FIRST_TEXT_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool the aspect and opinion span of every quadruplet.

    Implements Eq. (2) of the paper: ``h_a`` and ``h_o`` are the averages of the
    encoder states over the aspect and the opinion span. Implicit spans pool the
    ``NULL`` slot; padded quadruplets return zero vectors.

    Args:
        hidden_states: ``[B, L, H]`` encoder output.
        quad_spans: ``[B, Q, 4]`` inclusive ``(a_s, a_e, o_s, o_e)`` indices.
        quad_mask: ``[B, Q]``, 1 for real quadruplets and 0 for padding.
        null_slot: index pooled for an implicit span.
        first_text_index: spans starting below this index are implicit.

    Returns:
        ``(h_aspect, h_opinion)``, each ``[B, Q, H]``.
    """
    seq_len = hidden_states.shape[1]
    positions = torch.arange(seq_len, device=hidden_states.device).view(1, 1, seq_len)
    valid_quads = quad_mask.bool().unsqueeze(-1)  # [B, Q, 1]

    pooled = []
    for start_col, end_col in ((0, 1), (2, 3)):
        mask = _span_mask(
            quad_spans[:, :, start_col],
            quad_spans[:, :, end_col],
            positions,
            valid_quads,
            null_slot,
            first_text_index,
        ).to(hidden_states.dtype)  # [B, Q, L]
        counts = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        pooled.append(torch.bmm(mask, hidden_states) / counts)
    return pooled[0], pooled[1]


def pair_features(h_aspect: torch.Tensor, h_opinion: torch.Tensor) -> torch.Tensor:
    """Aspect-opinion pair representation ``[h_a; h_o; h_a * h_o]`` (``[..., 3H]``)."""
    return torch.cat([h_aspect, h_opinion, h_aspect * h_opinion], dim=-1)
