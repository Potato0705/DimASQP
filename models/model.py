"""
@Time : 2022/12/1720:27
@Auth : zhoujx
@File ：models.py
@DESCRIPTION:

"""
import torch.nn
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from dataset.dataset import AcqpDataset, collate_fn
from models.layers import EfficientGlobalPointer, GlobalPointer


def _pool_span(hidden_states, b, start, end):
    """Mean-pool a span from hidden states. Falls back to [SEP] (pos 1) for NULL spans."""
    if start >= 2 and end >= start:
        return hidden_states[b, start:end + 1].mean(dim=0)
    else:
        return hidden_states[b, 1]  # [SEP] for implicit/NULL span


class SpanPairVAHead(Module):
    """Predict VA conditioned on (aspect, opinion) span pair representations.

    Input:  encoder hidden states + span pair token indices
    Output: per-pair (V, A) prediction in [1, 9]
    """

    def __init__(self, hidden_size, va_hidden=256, dropout=0.1):
        super().__init__()
        # Input: [h_asp; h_opi; h_asp * h_opi] = 3 * hidden_size
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 3, va_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(va_hidden, va_hidden // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(va_hidden // 2, 2),
        )

    def forward(self, hidden_states, quad_spans, quad_mask):
        """
        hidden_states: [B, L, H]
        quad_spans:    [B, Q, 4]  (asp_s, asp_e, opi_s, opi_e) inclusive token indices
        quad_mask:     [B, Q]     1=valid, 0=padding
        Returns:       [B, Q, 2]  (V, A) in [1, 9]
        """
        B, Q = quad_spans.shape[:2]
        device = hidden_states.device
        va_preds = torch.zeros(B, Q, 2, device=device)

        for b in range(B):
            for q in range(Q):
                if quad_mask[b, q] < 0.5:
                    continue
                asp_s, asp_e, opi_s, opi_e = quad_spans[b, q].tolist()
                h_asp = _pool_span(hidden_states, b, asp_s, asp_e)
                h_opi = _pool_span(hidden_states, b, opi_s, opi_e)
                h_pair = torch.cat([h_asp, h_opi, h_asp * h_opi], dim=-1)
                va_preds[b, q] = torch.sigmoid(self.mlp(h_pair)) * 8.0 + 1.0

        return va_preds


class OpinionGuidedVAHead(Module):
    """Opinion-Guided VA Calibration: VA = prior(opinion) + gate * residual(asp, opi).

    Stage 1 - Opinion Prior:  h_opi -> MLP -> VA_prior in [1, 9]
    Stage 2 - Span-Pair Residual: [h_asp; h_opi; h_asp*h_opi] -> MLP -> delta_VA in [-4, 4]
    Stage 3 - Calibration: VA_final = clamp(VA_prior + gate * delta_VA, 1, 9)

    The opinion prior provides a stable anchor based on opinion semantics alone,
    while the residual adjusts for aspect-specific context.
    """

    def __init__(self, hidden_size, va_hidden=256, dropout=0.1):
        super().__init__()

        # Opinion Prior: opinion span -> VA anchor
        self.prior_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, va_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(va_hidden, 2),
        )

        # Span-Pair Residual: (asp, opi) pair -> VA adjustment
        self.residual_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 3, va_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(va_hidden, va_hidden // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(va_hidden // 2, 2),
        )

        # Learnable gate per VA dimension: controls residual influence
        self.gate = torch.nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, hidden_states, quad_spans, quad_mask):
        """
        Returns: dict with:
          'va_final':  [B, Q, 2]  calibrated VA in [1, 9]
          'va_prior':  [B, Q, 2]  opinion-only VA prior in [1, 9]
        """
        B, Q = quad_spans.shape[:2]
        device = hidden_states.device
        va_final = torch.zeros(B, Q, 2, device=device)
        va_prior = torch.zeros(B, Q, 2, device=device)

        gate = torch.sigmoid(self.gate)  # bounded [0, 1]

        for b in range(B):
            for q in range(Q):
                if quad_mask[b, q] < 0.5:
                    continue
                asp_s, asp_e, opi_s, opi_e = quad_spans[b, q].tolist()
                h_asp = _pool_span(hidden_states, b, asp_s, asp_e)
                h_opi = _pool_span(hidden_states, b, opi_s, opi_e)

                # Stage 1: Opinion prior
                prior = torch.sigmoid(self.prior_mlp(h_opi)) * 8.0 + 1.0  # [1, 9]
                va_prior[b, q] = prior

                # Stage 2: Span-pair residual
                h_pair = torch.cat([h_asp, h_opi, h_asp * h_opi], dim=-1)
                delta = torch.tanh(self.residual_mlp(h_pair)) * 4.0  # [-4, 4]

                # Stage 3: Calibrated output
                va_final[b, q] = torch.clamp(prior + gate * delta, 1.0, 9.0)

        return {'va_final': va_final, 'va_prior': va_prior}


class QuadrupleModel(Module):
    """
    新版三元组模型(四元组)
    """

    def __init__(self,
                 num_label_types,
                 num_dimension_types,
                 max_seq_len,
                 pretrain_model_path,
                 with_adversarial_training,
                 use_efficient_global_pointer,
                 mode='multiply',
                 head_size=64,
                 matrix_hidden_size=400,
                 dimension_hidden_size=400,
                 dimension_sequence_hidden_size=400,
                 sentiment_sequence_hidden_size=400,
                 dropout_rate=0.1,
                 RoPE=True):
        super(QuadrupleModel, self).__init__()
        self.num_label_types = num_label_types
        self.num_dimension_types = num_dimension_types
        self.num_sentiment_types = 3
        self.head_size = head_size
        self.mode = mode
        self.matrix_hidden_size = matrix_hidden_size,
        self.dimension_hidden_size = dimension_hidden_size,
        self.dimension_sequence_hidden_size = dimension_sequence_hidden_size,
        self.sentiment_sequence_hidden_size = sentiment_sequence_hidden_size,
        self.dropout_rate = dropout_rate
        self.RoPE = RoPE
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.pretrain_model_path = pretrain_model_path
        self.encoder = AutoModel.from_pretrained(self.pretrain_model_path)
        self.encoder_hidden_size = self.encoder.config.hidden_size
        self.with_adversarial_training = with_adversarial_training
        self.use_efficient_global_pointer = use_efficient_global_pointer  # True
        self.dropout_layer = torch.nn.Dropout(p=self.dropout_rate, )

        #
        self.matrix_linear = torch.nn.Linear(in_features=self.encoder_hidden_size, out_features=matrix_hidden_size)
        if self.mode == 'mul' and self.use_efficient_global_pointer:
            self.global_pointer_layer = EfficientGlobalPointer(hidden_size=matrix_hidden_size,
                                                               heads=self.num_label_types,
                                                               head_size=self.head_size,
                                                               RoPE=self.RoPE,
                                                               use_bias=True,
                                                               tril_mask=False,
                                                               max_length=self.max_seq_len)
        elif self.mode == 'mul' and not self.use_efficient_global_pointer:
            self.global_pointer_layer = GlobalPointer(hidden_size=matrix_hidden_size,
                                                      heads=self.num_label_types,
                                                      head_size=self.head_size,
                                                      RoPE=self.RoPE,
                                                      use_bias=True,
                                                      tril_mask=False,
                                                      max_length=self.max_seq_len)
        elif self.mode == 'add':
            self.matrix_linear = torch.nn.Linear(in_features=self.encoder_hidden_size * 2,
                                                 out_features=self.head_size)
            self.matrix_output = torch.nn.Linear(in_features=self.head_size,
                                                 out_features=self.num_label_types)
        elif self.mode == 'biaffine':
            self.linear_layer_start = torch.nn.Linear(in_features=self.encoder_hidden_size,
                                                      out_features=self.head_size)
            self.linear_layer_end = torch.nn.Linear(in_features=self.encoder_hidden_size,
                                                    out_features=self.head_size)
            self.U = torch.nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.randn([self.head_size + 1, self.num_label_types, self.head_size + 1])),
                requires_grad=True)

        #
        self.dimension_linear = torch.nn.Linear(in_features=self.encoder_hidden_size,
                                                out_features=dimension_hidden_size)
        self.dimension_output = torch.nn.Linear(in_features=dimension_hidden_size,
                                                out_features=self.num_dimension_types)

        # 维度序列
        self.dimension_sequence_linear = torch.nn.Linear(in_features=self.encoder_hidden_size,
                                                         out_features=dimension_sequence_hidden_size)
        self.dimension_sequence_output = torch.nn.Linear(in_features=dimension_sequence_hidden_size,
                                                         out_features=self.num_dimension_types)

        # 情感序列
        self.sentiment_sequence_linear = torch.nn.Linear(in_features=self.encoder_hidden_size,
                                                         out_features=sentiment_sequence_hidden_size)
        self.sentiment_sequence_output = torch.nn.Linear(in_features=sentiment_sequence_hidden_size,
                                                         out_features=self.num_sentiment_types)

        # VA回归头 (per-position): 每个token位置预测 [V, A], 通过 sigmoid*8+1 映射到 [1, 9]
        va_hidden_size = 256
        self.va_linear = torch.nn.Linear(in_features=self.encoder_hidden_size, out_features=va_hidden_size)
        self.va_output = torch.nn.Linear(in_features=va_hidden_size, out_features=2)

        # Span-Pair Conditioned VA head: VA由(aspect, opinion)对共同决定
        self.span_pair_va_head = SpanPairVAHead(
            hidden_size=self.encoder_hidden_size,
            va_hidden=va_hidden_size,
            dropout=dropout_rate,
        )

        # Opinion-Guided VA Calibration head: prior(opinion) + residual(asp, opi)
        self.opinion_guided_va_head = OpinionGuidedVAHead(
            hidden_size=self.encoder_hidden_size,
            va_hidden=va_hidden_size,
            dropout=dropout_rate,
        )

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,
                quad_spans=None, quad_mask=None, va_mode='span_pair'):

        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state

        cls_output = sequence_output[:, 0, :]
        # cls_output = F.relu(cls_output)
        # cls_output = self.dropout_layer(cls_output)
        cls_dim_output = self.dimension_linear(cls_output)
        cls_dim_output = F.relu(cls_dim_output)
        cls_dim_output = self.dropout_layer(cls_dim_output)
        cls_dim_output = self.dimension_output(cls_dim_output)

        # multiply
        if self.mode == 'mul':
            matrix_output = self.matrix_linear(sequence_output)
            matrix_output = F.relu(matrix_output)
            matrix_output = self.dropout_layer(matrix_output)
            matrix_output = self.global_pointer_layer(matrix_output, attention_mask=attention_mask)  # [B, 3, L, L]
        # add
        elif self.mode == 'add':
            start_extend = torch.unsqueeze(sequence_output, 2)  # [B, L, 1, H]
            start_extend = start_extend.expand(-1, -1, self.max_seq_len, -1)
            end_extend = torch.unsqueeze(sequence_output, 1)  # [B, 1, L, H]
            end_extend = end_extend.expand(-1, self.max_seq_len, -1, -1)
            span_matrix = torch.cat([start_extend, end_extend], 3)  # [B, L, L, 2*H]
            matrix_output = self.matrix_linear(span_matrix)
            matrix_output = F.relu(matrix_output)
            matrix_output = self.dropout_layer(matrix_output)
            matrix_output = self.matrix_output(matrix_output)
            matrix_output = torch.transpose(matrix_output, 1, 3)
            matrix_output = torch.transpose(matrix_output, 2, 3)
        # biaffine
        elif self.mode == 'biaffine':
            start = self.linear_layer_start(sequence_output)
            end = self.linear_layer_end(sequence_output)
            start = torch.cat((start, torch.ones_like(start[..., :1])), dim=-1)  # [B, L, Hide_size+1]
            end = torch.cat((end, torch.ones_like(end[..., :1])), dim=-1)  # [B, L, Hide_size+1]
            matrix_output = torch.einsum('bxi,ioj,byj->bxyo', start, self.U, end)
            matrix_output = torch.transpose(matrix_output, 1, 3)
            matrix_output = torch.transpose(matrix_output, 2, 3)

        # 维度序列
        dim_seq_output = self.dimension_sequence_linear(sequence_output)
        dim_seq_output = F.relu(dim_seq_output)
        dim_seq_output = self.dropout_layer(dim_seq_output)
        dim_seq_output = self.dimension_sequence_output(dim_seq_output)
        dim_seq_output = dim_seq_output.transpose(1, 2)

        # 情感序列
        sen_seq_output = self.sentiment_sequence_linear(sequence_output)
        sen_seq_output = F.relu(sen_seq_output)
        sen_seq_output = self.dropout_layer(sen_seq_output)
        sen_seq_output = self.sentiment_sequence_output(sen_seq_output)
        sen_seq_output = sen_seq_output.transpose(1, 2)

        # VA回归: [B, L, 2] -> sigmoid*8+1 -> [1, 9]
        va_output = self.va_linear(sequence_output)
        va_output = F.relu(va_output)
        va_output = self.dropout_layer(va_output)
        va_output = self.va_output(va_output)
        va_output = torch.sigmoid(va_output) * 8.0 + 1.0  # map to [1, 9]

        result = {"matrix": matrix_output,
                  "dimension": cls_dim_output,
                  "dimension_sequence": dim_seq_output,
                  "sentiment_sequence": sen_seq_output,
                  "va": va_output,
                  "hidden_states": sequence_output}

        # Span-Pair VA: only compute when span info is provided
        if quad_spans is not None and quad_mask is not None:
            if va_mode == 'opinion_guided':
                og_out = self.opinion_guided_va_head(sequence_output, quad_spans, quad_mask)
                result["span_va"] = og_out['va_final']       # calibrated VA
                result["va_prior"] = og_out['va_prior']       # opinion-only prior (for aux loss)
            else:
                result["span_va"] = self.span_pair_va_head(sequence_output, quad_spans, quad_mask)

        return result


if __name__ == '__main__':
    model = QuadrupleModel(num_label_types=40,
                           num_dimension_types=40,
                           max_seq_len=128,
                           pretrain_model_path="microsoft/deberta-v3-base",
                           with_adversarial_training=True,
                           use_efficient_global_pointer=True, )
    tokenizer = AutoTokenizer.from_pretrained("/data/transformers_pretrain_models/transformers_tf_en_deberta-v3-base/")
    dataset = AcqpDataset(task_domain="天美游戏",
                          tokenizer=tokenizer,
                          data_path="../data/游戏三元组标注数据_20221206.csv",
                          max_seq_len=128,
                          nrows=2000)
    dataloader = DataLoader(dataset,
                            batch_size=32,
                            shuffle=True,
                            num_workers=16,
                            drop_last=False,
                            collate_fn=collate_fn)
    data = next(iter(dataloader))
    input_ids = data["all_input_ids"]
    token_type_ids = data["all_token_type_ids"]
    attention_mask = data["all_attention_mask"]
    model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
