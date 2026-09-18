import math

import pytest
import torch

from dimasqp_va import (
    NULL_SLOT,
    OpinionGuidedVAHead,
    PositionVAHead,
    SpanPairVAHead,
    position_targets_from_quads,
    read_position_va,
    span_va_loss,
    to_va_range,
)

H = 16


def n_params(module):
    return sum(p.numel() for p in module.parameters())


@pytest.fixture(params=["span_pair", "span_pair_prior", "opinion_guided"])
def span_head(request):
    torch.manual_seed(1)
    if request.param == "span_pair":
        return SpanPairVAHead(H)
    if request.param == "span_pair_prior":
        return SpanPairVAHead(H, opinion_prior=True)
    return OpinionGuidedVAHead(H)


def test_to_va_range_bounds():
    x = torch.tensor([-1e4, -3.0, 0.0, 3.0, 1e4])
    y = to_va_range(x)
    assert y.min() >= 1.0 and y.max() <= 9.0
    assert y[2] == 5.0


def test_span_head_shapes_range_and_masking(span_head, batch):
    hidden, spans, mask, _ = batch
    span_head.eval()
    out = span_head(hidden, spans, mask)
    keys = {"va", "va_prior"} if not (isinstance(span_head, SpanPairVAHead) and span_head.prior_mlp is None) else {"va"}
    assert set(out) == keys
    for value in out.values():
        assert value.shape == (2, 4, 2)
        valid = mask.bool()
        assert (value[valid] >= 1.0).all() and (value[valid] <= 9.0).all()
        assert torch.count_nonzero(value[~valid]) == 0


def test_span_head_gradient_reaches_every_parameter(span_head, batch):
    hidden, spans, mask, gold = batch
    hidden = hidden.clone().requires_grad_(True)
    span_head.train()
    loss = span_va_loss(span_head(hidden, spans, mask), gold, mask, prior_weight=0.3)
    loss.backward()
    for name, p in span_head.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert hidden.grad.abs().sum() > 0


def test_opinion_guided_gate_and_bound(batch):
    hidden, spans, mask, _ = batch
    head = OpinionGuidedVAHead(H).eval()
    torch.testing.assert_close(head.gate_values(), torch.full((2,), 1 / (1 + math.exp(-0.5))))
    assert abs(head.gate_values()[0].item() - 0.6225) < 1e-4

    # A saturated residual can move the prior by at most sigmoid(g) * 4 before clamping.
    with torch.no_grad():
        head.residual_mlp[-1].weight.mul_(1e3)
    out = head(hidden, spans, mask)
    valid = mask.bool()
    shift = (out["va"] - out["va_prior"])[valid]
    assert (shift.abs() <= head.gate_values() * 4.0 + 1e-5).all()
    assert (out["va"][valid] >= 1.0).all() and (out["va"][valid] <= 9.0).all()


def test_opinion_guided_zero_residual_equals_prior(batch):
    hidden, spans, mask, _ = batch
    head = OpinionGuidedVAHead(H).eval()
    with torch.no_grad():
        head.residual_mlp[-1].weight.zero_()
        head.residual_mlp[-1].bias.zero_()
    out = head(hidden, spans, mask)
    torch.testing.assert_close(out["va"], out["va_prior"])


def test_opinion_guided_clamps_to_va_range(batch):
    hidden, spans, mask, _ = batch
    head = OpinionGuidedVAHead(H, gate_init=10.0).eval()
    with torch.no_grad():
        head.prior_mlp[-1].bias.fill_(20.0)  # prior ~ 9
        head.residual_mlp[-1].bias.fill_(20.0)  # residual ~ +4
    out = head(hidden, spans, mask)
    assert (out["va"][mask.bool()] == 9.0).all()


def test_prior_reads_only_the_opinion_span(batch):
    hidden, spans, mask, _ = batch
    head = OpinionGuidedVAHead(H).eval()
    moved = spans.clone()
    moved[0, 0, :2] = torch.tensor([9, 10])  # move the aspect, keep the opinion
    a = head(hidden, spans, mask)
    b = head(hidden, moved, mask)
    torch.testing.assert_close(a["va_prior"], b["va_prior"])
    assert not torch.allclose(a["va"][0, 0], b["va"][0, 0])


def test_parameter_counts_match_paper():
    # Sec. 5.4: Span-Pair MLP 623K, OG adds 197K, 820K in total (DeBERTa-v3-base, H = 768).
    sp = SpanPairVAHead(768)
    og = OpinionGuidedVAHead(768)
    assert n_params(sp) == 623_234
    assert n_params(og) == 820_614
    assert n_params(og) - n_params(sp) == 197_380
    assert n_params(SpanPairVAHead(768, opinion_prior=True)) == 623_234 + 197_378


def test_state_dict_layout():
    og = OpinionGuidedVAHead(8)
    assert set(og.state_dict()) == {
        "gate",
        *(f"prior_mlp.{i}.{p}" for i in (0, 3) for p in ("weight", "bias")),
        *(f"residual_mlp.{i}.{p}" for i in (0, 3, 6) for p in ("weight", "bias")),
    }
    assert "prior_mlp.0.weight" not in SpanPairVAHead(8).state_dict()


def test_position_head_shape_and_range():
    head = PositionVAHead(H).eval()
    out = head(torch.randn(3, 10, H))
    assert out.shape == (3, 10, 2)
    assert (out > 1.0).all() and (out < 9.0).all()


def test_position_head_gradient():
    head = PositionVAHead(H).train()
    hidden = torch.randn(2, 6, H, requires_grad=True)
    head(hidden).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    assert hidden.grad.abs().sum() > 0


def test_read_position_va(batch):
    _, spans, mask, _ = batch
    token_va = torch.arange(2 * 12 * 2, dtype=torch.float32).view(2, 12, 2) % 8 + 1.5
    out = read_position_va(token_va, spans, mask)
    torch.testing.assert_close(out[0, 0], token_va[0, 2])  # aspect start
    torch.testing.assert_close(out[0, 1], token_va[0, NULL_SLOT])  # NULL aspect
    torch.testing.assert_close(out[0, 2], token_va[0, 4])
    torch.testing.assert_close(out[1, 0], token_va[1, 6])
    assert torch.count_nonzero(out[~mask.bool()]) == 0


def test_padded_rows_may_hold_out_of_range_indices(span_head):
    """quad_mask, not the index values, decides padding (pooling.py docstring)."""
    L = 10
    hidden = torch.randn(1, L, H)
    spans = torch.tensor([[[3, 4, 6, 6], [99, 99, 99, 99]]])
    mask = torch.tensor([[1.0, 0.0]])
    gold = torch.tensor([[[6.0, 5.0], [0.0, 0.0]]])
    span_head.eval()
    out = span_head(hidden, spans, mask)
    assert torch.count_nonzero(out["va"][0, 1]) == 0
    token_va = to_va_range(torch.randn(1, L, 2))
    quad_va = read_position_va(token_va, spans, mask)
    torch.testing.assert_close(quad_va[0, 0], token_va[0, 3])
    assert torch.count_nonzero(quad_va[0, 1]) == 0
    # the same rows with in-range padding give the same result
    in_range = spans.clone()
    in_range[0, 1] = torch.tensor([5, 5, 7, 7])
    torch.testing.assert_close(read_position_va(token_va, in_range, mask), quad_va)
    torch.testing.assert_close(span_head(hidden, in_range, mask)["va"], out["va"])
    _, token_mask = position_targets_from_quads(spans, gold, mask, seq_len=L)
    assert token_mask.sum() == 2  # only the real quadruplet's aspect tokens 3-4


def test_read_position_va_shares_va_across_quads_with_same_aspect():
    token_va = torch.rand(1, 10, 2) * 8 + 1
    spans = torch.tensor([[[3, 4, 6, 6], [3, 4, 8, 9]]])  # same aspect, different opinions
    out = read_position_va(token_va, spans, torch.ones(1, 2))
    torch.testing.assert_close(out[0, 0], out[0, 1])
