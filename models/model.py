# -*- coding: utf-8 -*-
"""
@Time : 2022/12/17 (updated 2025/11 for SemEval-2026 Task3)
@Auth : zhoujx + adapter
@File : model.py
@Description :
    QuadrupleModel for One-ASQP, adapted for SemEval-2026 Task3 (DimABSA)
    - Fully compatible with new dataset.py outputs
    - No structural change to One-ASQP model core
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from dataset.dataset import AcqpDataset, collate_fn
from models.layers import EfficientGlobalPointer, GlobalPointer


class QuadrupleModel(Module):
    """四元组抽取模型（兼容 One-ASQP + SemEval）"""

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
    ):
        super(QuadrupleModel, self).__init__()
        self.num_label_types = num_label_types
        self.num_dimension_types = num_dimension_types
        self.num_sentiment_types = 3
        self.head_size = head_size
        self.mode = mode
        self.matrix_hidden_size = matrix_hidden_size
        self.dimension_hidden_size = dimension_hidden_size
        self.dimension_sequence_hidden_size = dimension_sequence_hidden_size
        self.sentiment_sequence_hidden_size = sentiment_sequence_hidden_size
        self.dropout_rate = dropout_rate
        self.RoPE = RoPE
        self.max_seq_len = max_seq_len
        self.pretrain_model_path = pretrain_model_path

        # backbone encoder
        self.encoder = AutoModel.from_pretrained(self.pretrain_model_path)
        self.encoder_hidden_size = self.encoder.config.hidden_size
        self.with_adversarial_training = with_adversarial_training
        self.use_efficient_global_pointer = use_efficient_global_pointer
        self.dropout_layer = torch.nn.Dropout(p=self.dropout_rate)

        # ===== Matrix (Global Pointer) =====
        self.matrix_linear = torch.nn.Linear(
            in_features=self.encoder_hidden_size, out_features=self.matrix_hidden_size
        )
        if self.mode == "mul" and self.use_efficient_global_pointer:
            self.global_pointer_layer = EfficientGlobalPointer(
                hidden_size=self.matrix_hidden_size,
                heads=self.num_label_types,
                head_size=self.head_size,
                RoPE=self.RoPE,
                use_bias=True,
                tril_mask=False,
                max_length=self.max_seq_len,
            )
        elif self.mode == "mul" and not self.use_efficient_global_pointer:
            self.global_pointer_layer = GlobalPointer(
                hidden_size=self.matrix_hidden_size,
                heads=self.num_label_types,
                head_size=self.head_size,
                RoPE=self.RoPE,
                use_bias=True,
                tril_mask=False,
                max_length=self.max_seq_len,
            )
        elif self.mode == "add":
            self.matrix_linear = torch.nn.Linear(
                in_features=self.encoder_hidden_size * 2, out_features=self.head_size
            )
            self.matrix_output = torch.nn.Linear(
                in_features=self.head_size, out_features=self.num_label_types
            )
        elif self.mode == "biaffine":
            self.linear_layer_start = torch.nn.Linear(
                in_features=self.encoder_hidden_size, out_features=self.head_size
            )
            self.linear_layer_end = torch.nn.Linear(
                in_features=self.encoder_hidden_size, out_features=self.head_size
            )
            self.U = torch.nn.Parameter(
                torch.nn.init.xavier_uniform_(
                    torch.randn(
                        [self.head_size + 1, self.num_label_types, self.head_size + 1]
                    )
                ),
                requires_grad=True,
            )

        # ===== CLS-based dimension prediction =====
        self.dimension_linear = torch.nn.Linear(
            in_features=self.encoder_hidden_size, out_features=self.dimension_hidden_size
        )
        self.dimension_output = torch.nn.Linear(
            in_features=self.dimension_hidden_size, out_features=self.num_dimension_types
        )

        # ===== Dimension sequence =====
        self.dimension_sequence_linear = torch.nn.Linear(
            in_features=self.encoder_hidden_size,
            out_features=self.dimension_sequence_hidden_size,
        )
        self.dimension_sequence_output = torch.nn.Linear(
            in_features=self.dimension_sequence_hidden_size,
            out_features=self.num_dimension_types,
        )

        # ===== Sentiment sequence =====
        self.sentiment_sequence_linear = torch.nn.Linear(
            in_features=self.encoder_hidden_size,
            out_features=self.sentiment_sequence_hidden_size,
        )
        self.sentiment_sequence_output = torch.nn.Linear(
            in_features=self.sentiment_sequence_hidden_size,
            out_features=self.num_sentiment_types,
        )

    # ===========================================================
    # forward
    # ===========================================================
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state  # [B, L, H]
        cls_output = sequence_output[:, 0, :]  # [CLS] embedding

        # ----- sentence-level dimension -----
        cls_dim_output = self.dimension_linear(cls_output)
        cls_dim_output = F.relu(cls_dim_output)
        cls_dim_output = self.dropout_layer(cls_dim_output)
        cls_dim_output = self.dimension_output(cls_dim_output)  # [B, num_dim]

        # ----- relation matrix -----
        if self.mode == "mul":
            matrix_output = self.matrix_linear(sequence_output)
            matrix_output = F.relu(matrix_output)
            matrix_output = self.dropout_layer(matrix_output)
            matrix_output = self.global_pointer_layer(
                matrix_output, attention_mask=attention_mask
            )  # [B, num_label, L, L]

        elif self.mode == "add":
            start_extend = torch.unsqueeze(sequence_output, 2)
            start_extend = start_extend.expand(-1, -1, self.max_seq_len, -1)
            end_extend = torch.unsqueeze(sequence_output, 1)
            end_extend = end_extend.expand(-1, self.max_seq_len, -1, -1)
            span_matrix = torch.cat([start_extend, end_extend], 3)
            matrix_output = self.matrix_linear(span_matrix)
            matrix_output = F.relu(matrix_output)
            matrix_output = self.dropout_layer(matrix_output)
            matrix_output = self.matrix_output(matrix_output)
            matrix_output = torch.transpose(matrix_output, 1, 3)
            matrix_output = torch.transpose(matrix_output, 2, 3)

        elif self.mode == "biaffine":
            start = self.linear_layer_start(sequence_output)
            end = self.linear_layer_end(sequence_output)
            start = torch.cat(
                (start, torch.ones_like(start[..., :1])), dim=-1
            )  # [B,L,H+1]
            end = torch.cat((end, torch.ones_like(end[..., :1])), dim=-1)
            matrix_output = torch.einsum("bxi,ioj,byj->bxyo", start, self.U, end)
            matrix_output = torch.transpose(matrix_output, 1, 3)
            matrix_output = torch.transpose(matrix_output, 2, 3)

        # ----- dimension sequence -----
        dim_seq_output = self.dimension_sequence_linear(sequence_output)
        dim_seq_output = F.relu(dim_seq_output)
        dim_seq_output = self.dropout_layer(dim_seq_output)
        dim_seq_output = self.dimension_sequence_output(dim_seq_output)
        dim_seq_output = dim_seq_output.transpose(1, 2)  # [B, num_dim, L]

        # ----- sentiment sequence -----
        sen_seq_output = self.sentiment_sequence_linear(sequence_output)
        sen_seq_output = F.relu(sen_seq_output)
        sen_seq_output = self.dropout_layer(sen_seq_output)
        sen_seq_output = self.sentiment_sequence_output(sen_seq_output)
        sen_seq_output = sen_seq_output.transpose(1, 2)  # [B, 3, L]

        return {
            "matrix": matrix_output,
            "dimension": cls_dim_output,
            "dimension_sequence": dim_seq_output,
            "sentiment_sequence": sen_seq_output,
        }


# ===========================================================
# Self-test (with SemEval dataset)
# ===========================================================
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    dataset = AcqpDataset(
        task_domain="SemEval_Eng_Laptop",
        tokenizer=tokenizer,
        data_path="../data/eng_laptop_train_alltasks.jsonl",
        max_seq_len=128,
        label_pattern="sentiment_dim",
    )

    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, drop_last=False, collate_fn=collate_fn
    )

    model = QuadrupleModel(
        num_label_types=len(dataset.label_types),
        num_dimension_types=len(dataset.dimension2id),
        max_seq_len=128,
        pretrain_model_path="microsoft/deberta-v3-base",
        with_adversarial_training=False,
        use_efficient_global_pointer=True,
    )

    batch = next(iter(dataloader))
    output = model(
        input_ids=batch["input_ids"],
        token_type_ids=batch["token_type_ids"],
        attention_mask=batch["attention_mask"],
    )

    print("==== Model Output Shapes ====")
    for k, v in output.items():
        print(f"{k:20s} {tuple(v.shape)}")
