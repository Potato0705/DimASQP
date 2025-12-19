"""
@File ：layers.py
@DESCRIPTION:
GlobalPointer / EfficientGlobalPointer + RoPE + lightweight metrics.

Fixes:
- Use unbind(dim=...) for PyTorch compatibility (axis may break)
- Use boolean masking + masked_fill(finfo.min) for fp16 stability (avoid INF multiplication)
- RoPE caches in float32 but casts to input dtype at runtime (AMP-friendly)
- MetricsCalculator provides micro/macro + per-label precision/recall/f1 (still memory-safe)
"""

import torch
import torch.nn as nn

INF = 1e12


# -------------------------
# Metrics (memory-safe)
# -------------------------
class MetricsCalculator(object):
    """
    Memory-safe metrics for multi-label GlobalPointer outputs.

    Expected y_pred/y_true shape:
      - [B, H, L, L] (typical)
    It aggregates TP / PredPos / TruePos per head (label).
    """
    def __init__(self, labels, ignore_index: int = 0, eps: float = 1e-9):
        self.labels = list(labels)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)
        self.reset()

    def reset(self):
        n = len(self.labels)
        self.tp = torch.zeros(n, dtype=torch.float64)
        self.pred = torch.zeros(n, dtype=torch.float64)
        self.true = torch.zeros(n, dtype=torch.float64)

    @torch.no_grad()
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # positives are logits > 0
        pred_pos = (y_pred > 0)
        true_pos = (y_true > 0)

        # sum over batch + all dims except head dim=1
        # y_pred: [B, H, ...] -> keep H
        dims = (0,) + tuple(range(2, y_pred.ndim))
        self.tp += (pred_pos & true_pos).sum(dims).cpu().to(torch.float64)
        self.pred += pred_pos.sum(dims).cpu().to(torch.float64)
        self.true += true_pos.sum(dims).cpu().to(torch.float64)

    def _prf(self, tp, p, t):
        prec = tp / (p + self.eps)
        rec = tp / (t + self.eps)
        f1 = 2 * prec * rec / (prec + rec + self.eps)
        return prec, rec, f1

    def result(self):
        """
        Return a dict similar in spirit to sklearn classification_report, but safe.
        Applies ignore_index by dropping the first K labels.
        """
        s = slice(self.ignore_index, len(self.labels))

        tp = self.tp[s]
        p = self.pred[s]
        t = self.true[s]

        per_p, per_r, per_f1 = self._prf(tp, p, t)

        # micro
        micro_tp = tp.sum()
        micro_p = p.sum()
        micro_t = t.sum()
        micro_prec, micro_rec, micro_f1 = self._prf(micro_tp, micro_p, micro_t)

        # macro (average over labels)
        macro_prec = per_p.mean() if len(per_p) > 0 else torch.tensor(0.0)
        macro_rec = per_r.mean() if len(per_r) > 0 else torch.tensor(0.0)
        macro_f1 = per_f1.mean() if len(per_f1) > 0 else torch.tensor(0.0)

        report = {
            "micro avg": {
                "precision": float(micro_prec),
                "recall": float(micro_rec),
                "f1-score": float(micro_f1),
                "support": float(micro_t),
            },
            "macro avg": {
                "precision": float(macro_prec),
                "recall": float(macro_rec),
                "f1-score": float(macro_f1),
                "support": float(micro_t),
            },
        }

        # per-label
        for i, name in enumerate(self.labels[self.ignore_index:]):
            report[name] = {
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1-score": float(per_f1[i]),
                "support": float(t[i]),
            }

        return report


# -------------------------
# RoPE
# -------------------------
class RotaryPositionEmbedding(nn.Module):
    """
    RoPE implementation that supports inputs of shape:
      - [B, L, D]
      - [B, L, H, D]
    where D is even.

    Buffers sin/cos are stored as float32 but cast to input dtype at runtime.
    """
    def __init__(self, dim: int, max_length: int = 512, head_axis=None):
        super().__init__()
        dim = int(dim)
        max_length = int(max_length)
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got dim={dim}")

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_length, dtype=torch.float32)
        freqs = torch.einsum("n,d->nd", t, inv_freq)  # [max_len, dim/2]

        # Keep head_axis for API compatibility, but we don't rely on it.
        self.register_buffer("sin", freqs.sin(), persistent=False)  # [max_len, dim/2]
        self.register_buffer("cos", freqs.cos(), persistent=False)  # [max_len, dim/2]

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: [B, L, D] or [B, L, H, D]
        seqlen = x.size(1)
        sin = self.sin[offset: offset + seqlen].to(dtype=x.dtype, device=x.device)  # [L, D/2]
        cos = self.cos[offset: offset + seqlen].to(dtype=x.dtype, device=x.device)  # [L, D/2]

        # reshape for broadcasting to x[..., 0::2]
        # target: [1, L, 1, ..., D/2]
        view = [1] * x.ndim
        view[1] = seqlen
        view[-1] = sin.size(-1)
        sin = sin.view(*view)
        cos = cos.view(*view)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).flatten(-2, -1)


# -------------------------
# GlobalPointer
# -------------------------
class GlobalPointer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int = 12,
        head_size: int = 64,
        RoPE: bool = True,
        use_bias: bool = True,
        tril_mask: bool = True,
        max_length: int = 512,
    ):
        super().__init__()
        self.heads = int(heads)
        self.head_size = int(head_size)
        self.RoPE = bool(RoPE)
        self.tril_mask = bool(tril_mask)

        self.dense = nn.Linear(hidden_size, self.heads * 2 * self.head_size, bias=use_bias)
        if self.RoPE:
            self.rotary = RotaryPositionEmbedding(self.head_size, max_length=max_length, head_axis=-2)

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        inputs: [B, L, hidden]
        attention_mask: [B, L] (0/1)
        return logits: [B, H, L, L]
        """
        x = self.dense(inputs)  # [B, L, H*2*D]
        bs, seqlen = x.shape[:2]

        x = x.reshape(bs, seqlen, self.heads, 2, self.head_size)
        qw, kw = x.unbind(dim=-2)  # IMPORTANT: dim not axis

        if self.RoPE:
            qw = self.rotary(qw)
            kw = self.rotary(kw)

        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)  # [B,H,L,L]
        logits = logits / (self.head_size ** 0.5)

        neg_inf = torch.finfo(logits.dtype).min

        # padding mask
        if attention_mask is not None:
            am = attention_mask.to(dtype=torch.bool)
            valid = am[:, None, None, :] & am[:, None, :, None]  # [B,1,L,L]
            logits = logits.masked_fill(~valid, neg_inf)

        # lower-triangular mask (exclude i>j)
        if self.tril_mask:
            tri = torch.tril(torch.ones((seqlen, seqlen), device=logits.device, dtype=torch.bool), diagonal=-1)
            logits = logits.masked_fill(tri[None, None, :, :], neg_inf)

        return logits


# -------------------------
# EfficientGlobalPointer
# -------------------------
class EfficientGlobalPointer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int = 12,
        head_size: int = 64,
        RoPE: bool = True,
        use_bias: bool = True,
        tril_mask: bool = True,
        max_length: int = 512,
    ):
        super().__init__()
        self.heads = int(heads)
        self.head_size = int(head_size)
        self.RoPE = bool(RoPE)
        self.tril_mask = bool(tril_mask)

        self.dense1 = nn.Linear(hidden_size, self.head_size * 2, bias=use_bias)
        self.dense2 = nn.Linear(self.head_size * 2, self.heads * 2, bias=use_bias)

        if self.RoPE:
            self.rotary = RotaryPositionEmbedding(self.head_size, max_length=max_length)

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        inputs: [B, L, hidden]
        attention_mask: [B, L] (0/1)
        return logits: [B, H, L, L]
        """
        x = self.dense1(inputs)  # [B, L, 2D]
        qw = x[..., 0::2]        # [B, L, D]
        kw = x[..., 1::2]        # [B, L, D]

        if self.RoPE:
            qw = self.rotary(qw)
            kw = self.rotary(kw)

        base = torch.einsum("bmd,bnd->bmn", qw, kw) / (self.head_size ** 0.5)  # [B, L, L]

        bias = self.dense2(x).transpose(1, 2) / 2.0  # [B, 2H, L]
        logits = base[:, None, :, :] + bias[:, ::2, None, :] + bias[:, 1::2, :, None]  # [B,H,L,L]

        neg_inf = torch.finfo(logits.dtype).min

        if attention_mask is not None:
            am = attention_mask.to(dtype=torch.bool)
            valid = am[:, None, None, :] & am[:, None, :, None]  # [B,1,L,L]
            logits = logits.masked_fill(~valid, neg_inf)

        if self.tril_mask:
            seqlen = logits.size(-1)
            tri = torch.tril(torch.ones((seqlen, seqlen), device=logits.device, dtype=torch.bool), diagonal=-1)
            logits = logits.masked_fill(tri[None, None, :, :], neg_inf)

        return logits
