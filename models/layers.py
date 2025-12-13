"""
@Time : 2022/12/17 20:35
@Auth : zhoujx
@File ：layers.py
@DESCRIPTION:
"""
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import classification_report

INF = 1e4


class MetricsCalculator(object):
    def __init__(self, labels, ignore_index=0):
        super().__init__()
        self.labels = labels
        self.ignore_index = ignore_index
        self.preds = []
        self.trues = []

    def reset(self):
        self.preds = []
        self.trues = []

    def update(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0)
        y_true = torch.gt(y_true, 0)
        tp = torch.argwhere((y_pred == True) & (y_true == True)).cpu().numpy()
        fp = torch.argwhere((y_pred == True) & (y_true == False)).cpu().numpy()
        fn = torch.argwhere((y_pred == False) & (y_true == True)).cpu().numpy()

        for _, dim, *_ in tp:
            self.preds.append(dim)
            self.trues.append(dim)

        for _, dim, *_ in fp:
            self.preds.append(dim)
            self.trues.append(len(self.labels))

        for _, dim, *_ in fn:
            self.preds.append(len(self.labels))
            self.trues.append(dim)

    def result(self):
        print(
            classification_report(
                y_true=self.trues,
                y_pred=self.preds,
                labels=list(range(len(self.labels)))[self.ignore_index:],
                target_names=self.labels[self.ignore_index:],
                digits=4,
                zero_division=0,
            )
        )

        dic_cls_report = classification_report(
            y_true=self.trues,
            y_pred=self.preds,
            labels=list(range(len(self.labels)))[self.ignore_index:],
            target_names=self.labels[self.ignore_index:],
            output_dict=True,
            digits=4,
            zero_division=0,
        )
        return dic_cls_report


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_length=512, head_axis=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_length, dtype=inv_freq.dtype)
        freqs = torch.einsum("n , d -> n d", t, inv_freq)
        if head_axis is not None:
            freqs = freqs.unsqueeze(head_axis)

        self.register_buffer("sin", freqs.sin(), persistent=False)
        self.register_buffer("cos", freqs.cos(), persistent=False)

    def forward(self, t, offset=0):
        seqlen = t.shape[1]
        sin = self.sin[offset : offset + seqlen, :]
        cos = self.cos[offset : offset + seqlen, :]

        t1, t2 = t[..., 0::2], t[..., 1::2]
        return torch.stack([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1).flatten(-2, -1)


class GlobalPointer(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads=12,
        head_size=64,
        RoPE=True,
        use_bias=True,
        tril_mask=True,
        max_length=512,
    ):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense = nn.Linear(hidden_size, heads * 2 * head_size, bias=use_bias)
        if RoPE:
            self.rotary = RotaryPositionEmbedding(head_size, max_length, head_axis=-2)

    def forward(self, inputs, attention_mask=None):
        inputs = self.dense(inputs)
        bs, seqlen = inputs.shape[:2]

        inputs = inputs.reshape(bs, seqlen, self.heads, 2, self.head_size)

        # ✅ PyTorch 参数名是 dim，不是 axis
        qw, kw = inputs.unbind(dim=-2)

        if self.RoPE:
            qw, kw = self.rotary(qw), self.rotary(kw)

        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)

        if attention_mask is not None:
            att = attention_mask.to(dtype=logits.dtype)
            am = attention_mask.float()
            attn_mask = (1 - am[:, None, None, :] * am[:, None, :, None])

            logits = logits - attn_mask * INF

        if self.tril_mask:
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)
            logits = logits - mask * INF

        return logits / (self.head_size ** 0.5)


class EfficientGlobalPointer(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads=12,
        head_size=64,
        RoPE=True,
        use_bias=True,
        tril_mask=True,
        max_length=512,
    ):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense1 = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.dense2 = nn.Linear(head_size * 2, heads * 2, bias=use_bias)
        if RoPE:
            self.rotary = RotaryPositionEmbedding(head_size, max_length)

    def forward(self, inputs, attention_mask=None):
        inputs = self.dense1(inputs)  # [B, L, head_size * 2]
        qw, kw = inputs[..., ::2], inputs[..., 1::2]

        if self.RoPE:
            qw, kw = self.rotary(qw), self.rotary(kw)

        logits = torch.einsum("bmd,bnd->bmn", qw, kw) / (self.head_size ** 0.5)  # [B, L, L]
        bias = self.dense2(inputs).transpose(1, 2) / 2  # [B, head*2, L]
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]  # [B, N, L, L]

        if attention_mask is not None:
            att = attention_mask.to(dtype=logits.dtype)
            am = attention_mask.float()
            attn_mask = (1 - am[:, None, None, :] * am[:, None, :, None])

            logits = logits - attn_mask * INF

        if self.tril_mask:
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)
            logits = logits - mask * INF

        return logits
