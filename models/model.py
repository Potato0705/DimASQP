# -*- coding: utf-8 -*-
import math
from typing import Dict, Any, Optional

import torch
from torch import nn
from transformers import AutoModel, AutoConfig


class QuadrupleModel(nn.Module):
    """
    输出：
      matrix: [B, C, L, L] logits
      dimension: [B, D] logits
    同时提供：
      encode_embeddings(...) -> [B, H] L2-normalized embedding（用于检索）
    """

    def __init__(
        self,
        num_label_types: int,
        num_dimension_types: int,
        max_seq_len: int,
        pretrain_model_path: str,
        gp_inner_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_label_types = int(num_label_types)
        self.num_dimension_types = int(num_dimension_types)
        self.max_seq_len = int(max_seq_len)
        self.pretrain_model_path = pretrain_model_path
        self.gp_inner_dim = int(gp_inner_dim)

        cfg = AutoConfig.from_pretrained(pretrain_model_path)
        self.encoder = AutoModel.from_pretrained(pretrain_model_path, config=cfg)

        hidden = cfg.hidden_size
        self.dropout = nn.Dropout(dropout)

        # GlobalPointer: 线性映射到 (Q,K) for each head
        # [B,L,H] -> [B,L,C,2*D]
        self.qk_proj = nn.Linear(hidden, self.num_label_types * self.gp_inner_dim * 2)

        # sentence-level dimension classifier
        self.dim_proj = nn.Linear(hidden, self.num_dimension_types)

    def forward(self, input_ids, token_type_ids, attention_mask) -> Dict[str, Any]:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )
        h = out.last_hidden_state  # [B,L,H]
        h = self.dropout(h)

        B, L, H = h.shape

        qk = self.qk_proj(h)  # [B,L,C*2D]
        qk = qk.view(B, L, self.num_label_types, 2 * self.gp_inner_dim)
        qw, kw = torch.split(qk, self.gp_inner_dim, dim=-1)  # [B,L,C,D]

        # logits: [B,C,L,L]
        # einsum: (B,L,C,D) x (B,L,C,D) -> (B,C,L,L)
        logits = torch.einsum("blcd,bmcd->bclm", qw, kw) / math.sqrt(self.gp_inner_dim)

        # dimension logits using CLS
        cls = h[:, 0]  # [B,H]
        dim_logits = self.dim_proj(cls)

        return {"matrix": logits, "dimension": dim_logits}

    @torch.no_grad()
    def encode_embeddings(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        pooling: str = "mean",
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        同源 encoder embedding：用于检索建库/查询。
        pooling:
          - "cls": 使用 CLS
          - "mean": attention_mask 平均池化
        """
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )
        h = out.last_hidden_state  # [B,L,H]

        if pooling == "cls":
            emb = h[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).float()  # [B,L,1]
            denom = torch.clamp(mask.sum(dim=1), min=1.0)
            emb = (h * mask).sum(dim=1) / denom

        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        return emb
