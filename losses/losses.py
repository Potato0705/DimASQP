# -*- coding: utf-8 -*-
from typing import Optional

import torch
import torch.nn.functional as F


def _make_pair_mask_from_attention(attn: torch.Tensor) -> torch.Tensor:
    """
    attn: [B,L] (0/1)
    return: [B,L,L] bool
    """
    m = attn.bool()
    return (m.unsqueeze(2) & m.unsqueeze(1))


def global_pointer_crossentropy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pair_mask: Optional[torch.Tensor] = None,
    tril_mask: bool = False,
    neg_topk: int = 0,
    neg_ratio: float = 0.0,
) -> torch.Tensor:
    """
    更干净的负样本/候选约束：
      - 只在 (valid_token_i, valid_token_j) 上算 loss（attention_mask / pair_mask）
      - 可选 hard negative mining：只取 hardest negatives（neg_topk 或 neg_ratio）
    y_true/y_pred:
      - [B,L,L] 或 [B,H,L,L]
    """
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)  # [B,1,L,L]
        y_pred = y_pred.unsqueeze(1)

    B, H, L, _ = y_true.shape

    mask = None
    if attention_mask is not None:
        mask = _make_pair_mask_from_attention(attention_mask)  # [B,L,L]
    if pair_mask is not None:
        mask = pair_mask.bool() if mask is None else (mask & pair_mask.bool())
    if mask is None:
        mask = torch.ones((B, L, L), device=y_true.device, dtype=torch.bool)

    if tril_mask:
        tri = torch.tril(torch.ones((L, L), device=y_true.device, dtype=torch.bool), diagonal=-1)
        mask = mask & (~tri)

    mask = mask.unsqueeze(1).expand(B, H, L, L)  # [B,H,L,L]

    # per-position BCE
    loss_map = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")  # [B,H,L,L]
    loss_map = loss_map.masked_fill(~mask, 0.0)

    # positives always kept
    pos = (y_true > 0.5) & mask
    neg = (~pos) & mask

    pos_loss = loss_map[pos]
    neg_loss = loss_map[neg]

    # 如果没有 hard mining，直接平均
    if (neg_topk is None or neg_topk <= 0) and (neg_ratio is None or neg_ratio <= 0):
        denom = pos.sum().clamp(min=1) + neg.sum().clamp(min=1)
        return loss_map.sum() / denom.float()

    # hard negative mining：按样本+head 做，避免被大样本吞掉
    total_loss = 0.0
    total_count = 0

    for b in range(B):
        for h in range(H):
            pm = pos[b, h]
            nm = neg[b, h]
            pl = loss_map[b, h][pm]
            nl = loss_map[b, h][nm]

            # negatives selection
            if nl.numel() > 0:
                if neg_topk is not None and neg_topk > 0:
                    k = min(int(neg_topk), int(nl.numel()))
                else:
                    # 按正样本比例采样
                    pnum = int(pl.numel())
                    k = int(max(1, neg_ratio * max(1, pnum)))
                    k = min(k, int(nl.numel()))
                if k < nl.numel():
                    nl, _ = torch.topk(nl, k=k, largest=True)

            s = 0.0
            c = 0
            if pl.numel() > 0:
                s = s + pl.sum()
                c += int(pl.numel())
            if nl.numel() > 0:
                s = s + nl.sum()
                c += int(nl.numel())

            if c > 0:
                total_loss += s
                total_count += c

    if total_count == 0:
        return loss_map.mean()
    return total_loss / float(total_count)


def boundary_bce_loss(
    logits_1d: torch.Tensor,
    targets_1d: torch.Tensor,
    valid_mask_1d: torch.Tensor,
) -> torch.Tensor:
    """
    logits_1d/targets_1d: [B,L]
    valid_mask_1d: [B,L] bool
    """
    loss = F.binary_cross_entropy_with_logits(logits_1d, targets_1d, reduction="none")
    loss = loss.masked_fill(~valid_mask_1d.bool(), 0.0)
    denom = valid_mask_1d.sum().clamp(min=1).float()
    return loss.sum() / denom
