# -*- coding: utf-8 -*-
"""
QuadrupleModel for One-ASQP, adapted for SemEval-2026 Task3 (DimABSA)
+ Pair-level Category classifier (num_categories already includes INVALID)
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from transformers import AutoModel

from models.layers import EfficientGlobalPointer, GlobalPointer


class QuadrupleModel(Module):
    """
    - GlobalPointer: entity + relation
    - CLS dimension
    - token-level dim/sent
    - pair-level category (includes INVALID as last class)
    """

    def __init__(
        self,
        num_label_types,
        num_dimension_types,
        max_seq_len,
        pretrain_model_path,
        with_adversarial_training=False,
        use_efficient_global_pointer=True,
        mode="mul",
        head_size=64,
        matrix_hidden_size=400,
        dimension_hidden_size=400,
        dimension_sequence_hidden_size=400,
        sentiment_sequence_hidden_size=400,
        dropout_rate=0.1,
        RoPE=True,
        num_categories: int = 0,   # already includes INVALID
    ):
        super().__init__()

        self.num_label_types = num_label_types
        self.num_dimension_types = num_dimension_types
        self.num_sentiment_types = 3
        self.head_size = head_size
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.RoPE = RoPE

        # encoder
        self.encoder = AutoModel.from_pretrained(pretrain_model_path)
        self.encoder_hidden_size = self.encoder.config.hidden_size
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

        # GlobalPointer
        self.matrix_linear = nn.Linear(self.encoder_hidden_size, matrix_hidden_size)

        if mode == "mul" and use_efficient_global_pointer:
            self.global_pointer_layer = EfficientGlobalPointer(
                hidden_size=matrix_hidden_size,
                heads=num_label_types,
                head_size=head_size,
                RoPE=RoPE,
                use_bias=True,
                tril_mask=False,
                max_length=max_seq_len,
            )
        elif mode == "mul":
            self.global_pointer_layer = GlobalPointer(
                hidden_size=matrix_hidden_size,
                heads=num_label_types,
                head_size=head_size,
                RoPE=RoPE,
                use_bias=True,
                tril_mask=False,
                max_length=max_seq_len,
            )
        else:
            raise ValueError(f"Unsupported GP mode={mode}")

        # CLS dimension
        self.dimension_linear = nn.Linear(self.encoder_hidden_size, dimension_hidden_size)
        self.dimension_output = nn.Linear(dimension_hidden_size, num_dimension_types)

        # dimension sequence
        self.dimension_sequence_linear = nn.Linear(self.encoder_hidden_size, dimension_sequence_hidden_size)
        self.dimension_sequence_output = nn.Linear(dimension_sequence_hidden_size, num_dimension_types)

        # sentiment sequence
        self.sentiment_sequence_linear = nn.Linear(self.encoder_hidden_size, sentiment_sequence_hidden_size)
        self.sentiment_sequence_output = nn.Linear(sentiment_sequence_hidden_size, self.num_sentiment_types)

        # Pair-level Category Classifier
        self.num_categories = int(num_categories)  # includes INVALID
        if self.num_categories > 1:
            self.category_mlp = nn.Sequential(
                nn.Linear(self.encoder_hidden_size * 4, 512),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(512, self.num_categories),
            )
        else:
            self.category_mlp = None

    def classify_pairs(self, sequence_output: torch.Tensor, pair_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence_output: [B, L, H]
            pair_indices: [N,3] = (b, a_start, o_end) or [N,2] if B==1
        Returns:
            logits: [N, C]
        """
        if self.category_mlp is None:
            raise RuntimeError("Category head not initialized (num_categories=0).")

        if pair_indices.dim() != 2 or pair_indices.size(-1) not in (2, 3):
            raise ValueError(f"pair_indices must be [N,2] or [N,3], got {pair_indices.shape}")

        if pair_indices.size(-1) == 2:
            b_idx = torch.zeros((pair_indices.size(0),), dtype=torch.long, device=pair_indices.device)
            a_idx = pair_indices[:, 0].long()
            o_idx = pair_indices[:, 1].long()
        else:
            b_idx = pair_indices[:, 0].long()
            a_idx = pair_indices[:, 1].long()
            o_idx = pair_indices[:, 2].long()

        h_a = sequence_output[b_idx, a_idx, :]  # [N,H]
        h_o = sequence_output[b_idx, o_idx, :]  # [N,H]

        pair_repr = torch.cat([h_a, h_o, torch.abs(h_a - h_o), h_a * h_o], dim=-1)  # [N,4H]
        return self.category_mlp(pair_repr)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state  # [B,L,H]
        cls_output = sequence_output[:, 0, :]

        # CLS dimension
        dim_cls = self.dimension_linear(cls_output)
        dim_cls = F.relu(dim_cls)
        dim_cls = self.dropout_layer(dim_cls)
        dim_cls = self.dimension_output(dim_cls)

        # relation matrix
        mat = self.matrix_linear(sequence_output)
        mat = F.relu(mat)
        mat = self.dropout_layer(mat)
        mat = self.global_pointer_layer(mat, attention_mask=attention_mask)

        # dimension sequence
        dim_seq = self.dimension_sequence_linear(sequence_output)
        dim_seq = F.relu(dim_seq)
        dim_seq = self.dropout_layer(dim_seq)
        dim_seq = self.dimension_sequence_output(dim_seq)
        dim_seq = dim_seq.transpose(1, 2)

        # sentiment sequence
        sen_seq = self.sentiment_sequence_linear(sequence_output)
        sen_seq = F.relu(sen_seq)
        sen_seq = self.dropout_layer(sen_seq)
        sen_seq = self.sentiment_sequence_output(sen_seq)
        sen_seq = sen_seq.transpose(1, 2)

        return {
            "matrix": mat,
            "dimension": dim_cls,
            "dimension_sequence": dim_seq,
            "sentiment_sequence": sen_seq,
            "sequence_output": sequence_output,
        }
