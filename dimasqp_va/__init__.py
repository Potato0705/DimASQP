"""Opinion-prior supervision and VA heads for DimASQP.

Reference implementation of the VA components of
"Opinion as Prior: Supervised VA Prediction for DimASQP" (ICONIP 2026).
"""

from .heads import (
    OpinionGuidedVAHead,
    PositionVAHead,
    SpanPairVAHead,
    feed_forward,
    read_position_va,
    to_va_range,
)
from .losses import (
    LAMBDA_CATEGORY,
    LAMBDA_MATRIX,
    LAMBDA_VA,
    PRIOR_WEIGHT,
    masked_va_mse,
    position_targets_from_quads,
    position_va_loss,
    quad_va_loss,
    span_va_loss,
    weighted_total_loss,
)
from .pooling import FIRST_TEXT_INDEX, NULL_SLOT, pair_features, pool_spans

__version__ = "1.0.0"

__all__ = [
    "FIRST_TEXT_INDEX",
    "LAMBDA_CATEGORY",
    "LAMBDA_MATRIX",
    "LAMBDA_VA",
    "NULL_SLOT",
    "OpinionGuidedVAHead",
    "PRIOR_WEIGHT",
    "PositionVAHead",
    "SpanPairVAHead",
    "feed_forward",
    "masked_va_mse",
    "pair_features",
    "pool_spans",
    "position_targets_from_quads",
    "position_va_loss",
    "quad_va_loss",
    "read_position_va",
    "span_va_loss",
    "to_va_range",
    "weighted_total_loss",
]
