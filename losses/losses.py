# -*- coding: utf-8 -*-
import torch

INF = 1e12

def multilabel_categorical_crossentropy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Safe for fp16 AMP:
    - compute in float32 to avoid overflow
    """
    y_true = y_true.float()
    y_pred = y_pred.float()

    # (1 - 2*y_true) * y_pred
    y_pred = (1.0 - 2.0 * y_true) * y_pred

    # Use dtype-aware large number (float32 here, safe)
    inf = torch.finfo(y_pred.dtype).max

    y_pred_neg = y_pred - y_true * inf
    y_pred_pos = y_pred - (1.0 - y_true) * inf

    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss



def global_pointer_crossentropy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    attention_mask: torch.Tensor = None,
    tril_mask: bool = False
) -> torch.Tensor:
    """
    Safe for fp16 AMP:
    - force y_pred/y_true to float32 inside loss
    - masked_fill uses dtype-aware finfo.min (no overflow)
    """
    # keep a "valid" mask before clamping
    valid = (y_true >= 0)

    y_true = torch.clamp(y_true, 0, 1).float()
    y_pred = y_pred.float()

    B, H, L, _ = y_pred.shape
    device = y_pred.device

    if attention_mask is not None:
        am = attention_mask.bool()
        m = am[:, None, :, None] & am[:, None, None, :]
    else:
        m = torch.ones((B, 1, L, L), dtype=torch.bool, device=device)

    if tril_mask:
        tri = torch.triu(torch.ones((L, L), dtype=torch.bool, device=device), diagonal=0)
        m = m & tri[None, None, :, :]

    m = m & valid.bool()

    neg_inf = torch.finfo(y_pred.dtype).min
    y_pred = y_pred.masked_fill(~m, neg_inf)
    y_true = y_true.masked_fill(~m, 0.0)

    y_pred = y_pred.reshape(B * H, -1)
    y_true = y_true.reshape(B * H, -1)

    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss.mean()



def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """
    logits/targets: [B,C,L] 或 [B,L]
    token_mask: [B,L]，True 表示有效 token（建议屏蔽 PAD/SEP，保留 CLS 看你策略）
    """
    if logits.dim() == 2:
        # [B,L]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        loss = loss.masked_fill(~token_mask, 0.0)
        denom = token_mask.sum().clamp_min(1).float()
        return loss.sum() / denom

    if logits.dim() == 3:
        # [B,C,L]
        m = token_mask[:, None, :].expand_as(logits)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        loss = loss.masked_fill(~m, 0.0)
        denom = m.sum().clamp_min(1).float()
        return loss.sum() / denom

    raise ValueError(f"Unsupported logits dim: {logits.dim()}")
