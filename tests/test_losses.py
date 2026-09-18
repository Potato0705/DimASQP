import pytest
import torch

from dimasqp_va import (
    LAMBDA_CATEGORY,
    LAMBDA_MATRIX,
    LAMBDA_VA,
    NULL_SLOT,
    PRIOR_WEIGHT,
    OpinionGuidedVAHead,
    PositionVAHead,
    SpanPairVAHead,
    masked_va_mse,
    position_targets_from_quads,
    position_va_loss,
    quad_va_loss,
    read_position_va,
    span_va_loss,
    weighted_total_loss,
)


def test_paper_weights():
    assert (LAMBDA_MATRIX, LAMBDA_CATEGORY, LAMBDA_VA, PRIOR_WEIGHT) == (1.0, 0.5, 0.5, 0.3)


def test_quad_va_loss_is_mean_squared_euclidean_distance(batch):
    _, _, mask, gold = batch
    pred = gold + torch.randn_like(gold)
    valid = mask.bool()
    expected = ((pred[valid] - gold[valid]) ** 2).sum(-1).mean()
    torch.testing.assert_close(quad_va_loss(pred, gold, mask), expected)


def test_padding_does_not_change_the_loss(batch):
    _, _, mask, gold = batch
    pred = gold + 0.5
    base = quad_va_loss(pred, gold, mask)
    noisy = pred.clone()
    noisy[~mask.bool()] = 1e3  # garbage in padded slots
    torch.testing.assert_close(quad_va_loss(noisy, gold, mask), base)


def test_all_masked_gives_zero():
    pred = torch.rand(2, 3, 2) * 8 + 1
    assert masked_va_mse(pred, torch.zeros_like(pred), torch.zeros(2, 3)).item() == 0.0


def test_prior_term_added_with_weight(batch):
    _, _, mask, gold = batch
    va = gold + 0.3
    prior = gold - 0.8
    outputs = {"va": va, "va_prior": prior}
    for w in (0.0, 0.3, 1.0):
        expected = quad_va_loss(va, gold, mask) + w * quad_va_loss(prior, gold, mask)
        torch.testing.assert_close(span_va_loss(outputs, gold, mask, prior_weight=w), expected)
    # a head without a prior ignores the weight
    torch.testing.assert_close(span_va_loss({"va": va}, gold, mask, 0.3), quad_va_loss(va, gold, mask))


def test_prior_loss_vanishes_for_a_perfect_prior(batch):
    _, _, mask, gold = batch
    outputs = {"va": gold + 1.0, "va_prior": gold.clone()}
    torch.testing.assert_close(span_va_loss(outputs, gold, mask, 0.3), quad_va_loss(outputs["va"], gold, mask))


def test_prior_gradient_is_scaled_by_prior_weight(batch):
    _, _, mask, gold = batch
    prior = (gold + 0.7).requires_grad_(True)
    span_va_loss({"va": gold.clone(), "va_prior": prior}, gold, mask, prior_weight=0.3).backward()
    n_valid = mask.sum()
    expected = 0.3 * 2 * (prior.detach() - gold) * mask.unsqueeze(-1) / n_valid
    torch.testing.assert_close(prior.grad, expected)


def test_sp_prior_supervision_trains_prior_only_when_weighted(batch):
    hidden, spans, mask, gold = batch
    for weight, expect_grad in ((0.0, False), (0.3, True)):
        head = SpanPairVAHead(hidden.shape[-1], opinion_prior=True)
        span_va_loss(head(hidden, spans, mask), gold, mask, prior_weight=weight).backward()
        prior_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.prior_mlp.parameters())
        assert prior_has_grad == expect_grad


def test_og_prior_is_trained_through_the_final_prediction_even_without_prior_loss(batch):
    hidden, spans, mask, gold = batch
    head = OpinionGuidedVAHead(hidden.shape[-1])
    span_va_loss(head(hidden, spans, mask), gold, mask, prior_weight=0.0).backward()
    assert all(p.grad.abs().sum() > 0 for p in head.prior_mlp.parameters())


def test_opinion_prior_supervision_fits_opinion_driven_va():
    """On synthetic data where VA is a function of the opinion token only, supervising
    the prior makes the prior itself a good VA predictor; without the prior loss the
    prior is not identifiable and stays far off."""
    torch.manual_seed(0)
    B, L, H, Q, vocab = 32, 10, 16, 2, 6
    embed = torch.randn(vocab, H)
    word_va = torch.rand(vocab, 2) * 6 + 2  # each "opinion word" has a VA value
    ids = torch.randint(0, vocab, (B, L))
    hidden = embed[ids]
    spans = torch.stack(
        [torch.tensor([2, 3, 5, 5]).expand(B, 4), torch.tensor([4, 4, 8, 8]).expand(B, 4)], dim=1
    )
    mask = torch.ones(B, Q)
    gold = torch.stack([word_va[ids[:, 5]], word_va[ids[:, 8]]], dim=1)

    def prior_error(weight):
        torch.manual_seed(1)
        head = OpinionGuidedVAHead(H, va_hidden=32, dropout=0.0)
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        for _ in range(300):
            opt.zero_grad()
            span_va_loss(head(hidden, spans, mask), gold, mask, prior_weight=weight).backward()
            opt.step()
        with torch.no_grad():
            return quad_va_loss(head(hidden, spans, mask)["va_prior"], gold, mask).item()

    supervised, unsupervised = prior_error(0.3), prior_error(0.0)
    assert supervised < 0.05
    assert supervised < 0.2 * unsupervised


def test_position_targets_cover_gold_aspect_tokens(batch):
    _, spans, mask, gold = batch
    targets, tmask = position_targets_from_quads(spans, gold, mask, seq_len=12)
    # batch 0: aspect 2..3 -> quad 0; NULL aspect + explicit opinion -> NULL slot; aspect 4 -> quad 2
    assert tmask[0].nonzero().flatten().tolist() == [NULL_SLOT, 2, 3, 4]
    torch.testing.assert_close(targets[0, 2], gold[0, 0])
    torch.testing.assert_close(targets[0, 3], gold[0, 0])
    torch.testing.assert_close(targets[0, NULL_SLOT], gold[0, 1])
    torch.testing.assert_close(targets[0, 4], gold[0, 2])
    # batch 1: aspect 6 -> quad 0; both-NULL quad gives no target
    assert tmask[1].nonzero().flatten().tolist() == [6]


def test_position_targets_later_quad_wins_on_overlap():
    spans = torch.tensor([[[3, 5, 7, 7], [4, 4, 8, 8]]])
    va = torch.tensor([[[2.0, 3.0], [8.0, 7.0]]])
    targets, tmask = position_targets_from_quads(spans, va, torch.ones(1, 2), seq_len=10)
    assert tmask[0].nonzero().flatten().tolist() == [3, 4, 5]
    torch.testing.assert_close(targets[0, 4], va[0, 1])
    torch.testing.assert_close(targets[0, 3], va[0, 0])


def test_position_pipeline_end_to_end(batch):
    hidden, spans, mask, gold = batch
    head = PositionVAHead(hidden.shape[-1])
    targets, tmask = position_targets_from_quads(spans, gold, mask, seq_len=hidden.shape[1])
    token_va = head(hidden)
    loss = position_va_loss(token_va, targets, tmask)
    loss.backward()
    assert loss.item() > 0
    assert read_position_va(token_va.detach(), spans, mask).shape == (2, 4, 2)


@pytest.mark.parametrize("weights", [(1.0, 0.5, 0.5), (2.0, 0.1, 0.0)])
def test_weighted_total_loss(weights):
    lm, lc, lv = torch.tensor(1.5), torch.tensor(0.7), torch.tensor(0.9)
    expected = weights[0] * 1.5 + weights[1] * 0.7 + weights[2] * 0.9
    got = weighted_total_loss(lm, lc, lv, *weights)
    assert abs(got.item() - expected) < 1e-6
    assert abs(weighted_total_loss(lm, lc, lv).item() - (1.5 + 0.35 + 0.45)) < 1e-6
