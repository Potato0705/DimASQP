"""Attach the VA heads to a token encoder and train them for a few steps.

Runs on CPU in a few seconds with a tiny, randomly initialised Transformer
encoder and synthetic quadruplets, so nothing is downloaded. In the paper the
encoder is DeBERTa-v3-base; with Hugging Face ``transformers`` you would use

    encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
    hidden_states = encoder(input_ids=..., attention_mask=...).last_hidden_state

and pass ``hidden_states`` to the heads exactly as below. The extraction model
that produces the spans at inference time is not part of this package.

Usage:
    python examples/attach_to_encoder.py
"""

import torch
from torch import nn

from dimasqp_va import (
    NULL_SLOT,
    OpinionGuidedVAHead,
    PositionVAHead,
    SpanPairVAHead,
    position_targets_from_quads,
    position_va_loss,
    read_position_va,
    span_va_loss,
)

VOCAB, HIDDEN, SEQ_LEN, BATCH, MAX_QUADS = 100, 64, 16, 8, 3


class TinyEncoder(nn.Module):
    """Stand-in for a pre-trained encoder: returns ``[B, L, H]`` hidden states."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        layer = nn.TransformerEncoderLayer(HIDDEN, nhead=4, dim_feedforward=128, batch_first=True)
        self.layers = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, input_ids):
        return self.layers(self.embed(input_ids))


def synthetic_batch():
    """Token 0: leading special token, token 1: NULL slot, text from token 2."""
    input_ids = torch.randint(3, VOCAB, (BATCH, SEQ_LEN), generator=torch.Generator().manual_seed(0))
    input_ids[:, 0], input_ids[:, 1] = 0, 1
    quad_spans = torch.full((BATCH, MAX_QUADS, 4), -1, dtype=torch.long)
    quad_mask = torch.zeros(BATCH, MAX_QUADS)
    for b in range(BATCH):
        quad_spans[b, 0] = torch.tensor([2, 3, 5, 5])  # explicit aspect and opinion
        quad_spans[b, 1] = torch.tensor([2, 3, 8, 9])  # same aspect, another opinion
        quad_spans[b, 2] = torch.tensor([NULL_SLOT, NULL_SLOT, 11, 11])  # implicit aspect
        quad_mask[b, : 2 + b % 2] = 1.0  # odd rows have three quadruplets, even rows two
    # Let the gold VA depend on the first opinion token, as opinion-prior supervision assumes.
    opinion_tokens = torch.gather(input_ids, 1, quad_spans[:, :, 2].clamp(min=0))
    gold_va = 1.0 + 8.0 * torch.stack([(opinion_tokens % 7) / 6, (opinion_tokens % 5) / 4], dim=-1)
    return input_ids, quad_spans, quad_mask, gold_va * quad_mask.unsqueeze(-1)


def fmt(values):
    return [round(v, 2) for v in values.flatten().tolist()]


def train(head_name, batch, steps=60):
    torch.manual_seed(0)
    encoder = TinyEncoder()
    if head_name == "Position":
        head = PositionVAHead(HIDDEN)
    elif head_name == "Span-Pair":
        head = SpanPairVAHead(HIDDEN)
    elif head_name == "SP+Prior":
        head = SpanPairVAHead(HIDDEN, opinion_prior=True)
    else:
        head = OpinionGuidedVAHead(HIDDEN)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    input_ids, quad_spans, quad_mask, gold_va = batch

    for step in range(steps + 1):
        hidden_states = encoder(input_ids)
        if head_name == "Position":
            targets, token_mask = position_targets_from_quads(quad_spans, gold_va, quad_mask, SEQ_LEN)
            loss = position_va_loss(head(hidden_states), targets, token_mask)
        else:
            loss = span_va_loss(head(hidden_states, quad_spans, quad_mask), gold_va, quad_mask, prior_weight=0.3)
        if step % 20 == 0:
            print(f"  {head_name:<14} step {step:3d}  L_VA = {loss.item():.3f}")
        if step < steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Inference: predicted quadruplet VA (spans would come from the extractor).
    encoder.eval(), head.eval()
    with torch.no_grad():
        hidden_states = encoder(input_ids)
        if head_name == "Position":
            va = read_position_va(head(hidden_states), quad_spans, quad_mask)
        else:
            va = head(hidden_states, quad_spans, quad_mask)["va"]
    print(f"  {head_name:<14} sentence 0, predicted VA: {fmt(va[0, :2])}   gold: {fmt(gold_va[0, :2])}")
    if isinstance(head, OpinionGuidedVAHead):
        print(f"  {head_name:<14} gate sigmoid(g) = {fmt(head.gate_values())}")


if __name__ == "__main__":
    batch = synthetic_batch()
    for name in ("Position", "Span-Pair", "SP+Prior", "Opinion-Guided"):
        train(name, batch)
