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


def _batch_pool_spans(hidden_states, quad_spans, quad_mask):
    """Vectorized span pooling — replaces nested Python for-loops.

    For each valid quad, mean-pool aspect and opinion spans from hidden_states.
    NULL/implicit spans (start < 2) fall back to [SEP] token (position 1).

    Args:
        hidden_states: [B, L, H]
        quad_spans:    [B, Q, 4]  (asp_s, asp_e, opi_s, opi_e) inclusive indices
        quad_mask:     [B, Q]     1=valid, 0=padding

    Returns:
        h_asp: [B, Q, H]  aspect span representations (zero for masked quads)
        h_opi: [B, Q, H]  opinion span representations (zero for masked quads)
    """
    B, L, H = hidden_states.shape
    Q = quad_spans.shape[1]
    device = hidden_states.device

    # Position indices for mask construction
    pos = torch.arange(L, device=device).view(1, 1, L)  # [1, 1, L]

    asp_s = quad_spans[:, :, 0].unsqueeze(-1)  # [B, Q, 1]
    asp_e = quad_spans[:, :, 1].unsqueeze(-1)
    opi_s = quad_spans[:, :, 2].unsqueeze(-1)
    opi_e = quad_spans[:, :, 3].unsqueeze(-1)

    # Build span masks via pure broadcasting (no in-place boolean indexing)
    asp_valid = (asp_s >= 2)  # [B, Q, 1]  True if real span
    opi_valid = (opi_s >= 2)

    # Normal spans: token positions within [start, end]
    asp_span_mask = (pos >= asp_s) & (pos <= asp_e) & asp_valid     # [B, Q, L]
    opi_span_mask = (pos >= opi_s) & (pos <= opi_e) & opi_valid

    # NULL/implicit spans: fall back to [SEP] at position 1
    sep_mask = (pos == 1)  # [1, 1, L] broadcast-ready
    asp_span_mask = asp_span_mask | (~asp_valid & sep_mask)
    opi_span_mask = opi_span_mask | (~opi_valid & sep_mask)

    # Zero out padding quads
    qm = quad_mask.bool().unsqueeze(-1)  # [B, Q, 1]
    asp_span_mask = asp_span_mask & qm
    opi_span_mask = opi_span_mask & qm

    # Float masks for mean pooling
    asp_f = asp_span_mask.float()  # [B, Q, L]
    opi_f = opi_span_mask.float()

    # Batched mean-pool: [B, Q, L] @ [B, L, H] -> [B, Q, H]
    h_asp = torch.bmm(asp_f, hidden_states) / asp_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
    h_opi = torch.bmm(opi_f, hidden_states) / opi_f.sum(dim=-1, keepdim=True).clamp(min=1.0)

    return h_asp, h_opi


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
        h_asp, h_opi = _batch_pool_spans(hidden_states, quad_spans, quad_mask)
        h_pair = torch.cat([h_asp, h_opi, h_asp * h_opi], dim=-1)  # [B, Q, 3H]
        va_preds = torch.sigmoid(self.mlp(h_pair)) * 8.0 + 1.0     # [B, Q, 2]
        # Zero out padding quads
        va_preds = va_preds * quad_mask.unsqueeze(-1)
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

    def forward_prior(self, hidden_states, quad_spans, quad_mask):
        """Opinion-only VA prior in [1, 9], exposed for auxiliary loss."""
        _, h_opi = _batch_pool_spans(hidden_states, quad_spans, quad_mask)
        mask = quad_mask.unsqueeze(-1)
        va_prior = torch.sigmoid(self.prior_mlp(h_opi)) * 8.0 + 1.0
        return va_prior * mask

    def forward(self, hidden_states, quad_spans, quad_mask):
        """
        Returns: dict with:
          'va_final':  [B, Q, 2]  calibrated VA in [1, 9]
          'va_prior':  [B, Q, 2]  opinion-only VA prior in [1, 9]
        """
        h_asp, h_opi = _batch_pool_spans(hidden_states, quad_spans, quad_mask)
        mask = quad_mask.unsqueeze(-1)  # [B, Q, 1]
        gate = torch.sigmoid(self.gate)  # [2]

        # Stage 1: Opinion prior
        va_prior = torch.sigmoid(self.prior_mlp(h_opi)) * 8.0 + 1.0  # [B, Q, 2]

        # Stage 2: Span-pair residual
        h_pair = torch.cat([h_asp, h_opi, h_asp * h_opi], dim=-1)  # [B, Q, 3H]
        delta = torch.tanh(self.residual_mlp(h_pair)) * 4.0  # [B, Q, 2]

        # Stage 3: Calibrated output
        va_final = torch.clamp(va_prior + gate * delta, 1.0, 9.0)

        # Zero out padding quads
        va_final = va_final * mask
        va_prior = va_prior * mask

        return {'va_final': va_final, 'va_prior': va_prior}


class VAContrastiveProjection(Module):
    """Project span-pair representations to a low-dim space for contrastive learning.

    Used only during training to enforce VA-aware structure in the representation
    space. Not needed at inference time.

    Input:  h_pair = [h_asp; h_opi; h_asp*h_opi]  (hidden_size * 3)
    Output: L2-normalized embedding in R^proj_dim
    """

    def __init__(self, hidden_size, proj_dim=64, dropout=0.1):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 3, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, proj_dim),
        )

    def forward(self, hidden_states, quad_spans, quad_mask):
        """Return L2-normalized embeddings [B, Q, proj_dim]."""
        h_asp, h_opi = _batch_pool_spans(hidden_states, quad_spans, quad_mask)
        h_pair = torch.cat([h_asp, h_opi, h_asp * h_opi], dim=-1)  # [B, Q, 3H]
        embeds = self.proj(h_pair)  # [B, Q, proj_dim]

        # Zero out padding quads before normalization
        embeds = embeds * quad_mask.unsqueeze(-1)

        # L2 normalize (avoid division by zero for masked positions)
        norms = embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        embeds = embeds / norms
        return embeds


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
        # Force float32: some servers default to float16 loading.
        # float16 max ≈ 65504 — INF masking constants (originally 1e12, now 1e4)
        # still risk overflow if the model itself computes in float16.
        # Explicitly loading in float32 guarantees stable numerics on all servers.
        self.encoder = AutoModel.from_pretrained(self.pretrain_model_path,
                                                 torch_dtype=torch.float32)
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

        # VA-Aware Contrastive Projection (training-only, not used at inference)
        self.va_cl_projection = VAContrastiveProjection(
            hidden_size=self.encoder_hidden_size,
            proj_dim=64,
            dropout=dropout_rate,
        )

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,
                quad_spans=None, quad_mask=None, va_mode='span_pair'):

        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        head_dtype = self.dimension_linear.weight.dtype
        if sequence_output.dtype != head_dtype:
            sequence_output = sequence_output.to(head_dtype)

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
                if va_mode == 'span_pair':
                    result["va_prior"] = self.opinion_guided_va_head.forward_prior(
                        sequence_output, quad_spans, quad_mask
                    )

            # VA-Aware Contrastive embeddings (training only)
            if self.training:
                result["va_cl_embeds"] = self.va_cl_projection(
                    sequence_output, quad_spans, quad_mask)  # [B, Q, 64]

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
