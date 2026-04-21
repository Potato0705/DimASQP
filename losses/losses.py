import torch
import torch.nn.functional as F

# Safe masking constant: 1e12 overflows in FP16 (max ~65504). Use 1e4 which is
# large enough to dominate any real logit but safe under mixed-precision.
_INF = 1e4
# Per-element loss clamp to prevent single outliers from exploding the sum.
_LOSS_CLAMP = 1e4


def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * _INF
    y_pred_pos = y_pred - (1 - y_true) * _INF
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    per_element = neg_loss + pos_loss
    # Clamp to prevent explosion from numerical edge cases
    per_element = per_element.clamp(max=_LOSS_CLAMP)
    return per_element


def global_pointer_crossentropy(y_true, y_pred, mask_rate=None, tril_mask=False):
    """给GlobalPointer设计的交叉熵
    """
    if mask_rate is not None:
        if y_pred.is_cuda:
            # Avoid CUDA graph RNG mismatch by using the active RNG on y_pred's device.
            with torch.cuda.device(y_pred.device):
                masked = torch.rand(y_true.shape, device=y_pred.device) < (1 - mask_rate)
        else:
            masked = torch.rand(y_true.shape, device=y_pred.device) < (1 - mask_rate)
        masked = masked + y_true > 0

        if y_true.ndim == 4:
            mask_row_col = torch.any(y_true>0, dim=-3, keepdim=True)
            masked = masked | mask_row_col

        if tril_mask:
            mask_tril = torch.triu(torch.ones_like(masked), diagonal=0)
            masked = masked & mask_tril

        # y_true == -1的，mask掉
        masked = torch.where(y_true < 0, torch.zeros_like(masked), masked)

        masked = masked.to(y_pred.dtype)
        y_pred = y_pred - (1 - masked) * _INF

    bh = y_pred.shape[0] * y_pred.shape[1]
    y_pred = torch.reshape(y_pred, (bh, -1))
    y_true = torch.reshape(y_true, (bh, -1))

    raw_loss = multilabel_categorical_crossentropy(y_true, y_pred)
    # Normalise by positive label count, with a floor to avoid extreme amplification
    n_pos = torch.sum(y_true).clamp(min=1.0)
    loss = torch.sum(raw_loss) / n_pos

    # Final safety clamp — if loss is still unreasonably large, cap it
    loss = loss.clamp(max=_LOSS_CLAMP)
    return loss
