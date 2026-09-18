import torch

from dimasqp_va import NULL_SLOT, pair_features, pool_spans


def test_shapes(batch):
    hidden, spans, mask, _ = batch
    h_a, h_o = pool_spans(hidden, spans, mask)
    assert h_a.shape == h_o.shape == (2, 4, hidden.shape[-1])


def test_explicit_span_is_mean_of_inclusive_range(batch):
    hidden, spans, mask, _ = batch
    h_a, h_o = pool_spans(hidden, spans, mask)
    torch.testing.assert_close(h_a[0, 0], hidden[0, 2:4].mean(0))  # aspect 2..3 inclusive
    torch.testing.assert_close(h_o[0, 0], hidden[0, 5])  # single-token opinion
    torch.testing.assert_close(h_o[1, 0], hidden[1, 2:5].mean(0))  # opinion 2..4 inclusive


def test_implicit_span_reads_null_slot(batch):
    hidden, spans, mask, _ = batch
    h_a, h_o = pool_spans(hidden, spans, mask)
    torch.testing.assert_close(h_a[0, 1], hidden[0, NULL_SLOT])  # NULL aspect
    torch.testing.assert_close(h_o[0, 2], hidden[0, NULL_SLOT])  # NULL opinion
    torch.testing.assert_close(h_a[1, 1], hidden[1, NULL_SLOT])  # both NULL
    torch.testing.assert_close(h_o[1, 1], hidden[1, NULL_SLOT])


def test_padding_pools_to_zero(batch):
    hidden, spans, mask, _ = batch
    h_a, h_o = pool_spans(hidden, spans, mask)
    for b, q in [(0, 3), (1, 2), (1, 3)]:
        assert torch.count_nonzero(h_a[b, q]) == 0
        assert torch.count_nonzero(h_o[b, q]) == 0


def test_mask_overrides_real_indices(batch):
    hidden, spans, mask, _ = batch
    mask = mask.clone()
    mask[0, 0] = 0.0  # a real-looking span that is marked as padding
    h_a, _ = pool_spans(hidden, spans, mask)
    assert torch.count_nonzero(h_a[0, 0]) == 0


def test_padding_positions_get_no_gradient(batch):
    hidden, spans, mask, _ = batch
    hidden = hidden.clone().requires_grad_(True)
    h_a, h_o = pool_spans(hidden, spans, mask)
    (h_a.sum() + h_o.sum()).backward()
    touched = {(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 7), (0, 8), (1, 1), (1, 2), (1, 3), (1, 4), (1, 6)}
    for b in range(2):
        for t in range(hidden.shape[1]):
            has_grad = bool(hidden.grad[b, t].abs().sum() > 0)
            assert has_grad == ((b, t) in touched), (b, t)


def test_pair_features():
    a = torch.randn(2, 3, 5)
    o = torch.randn(2, 3, 5)
    feats = pair_features(a, o)
    assert feats.shape == (2, 3, 15)
    torch.testing.assert_close(feats[..., 10:], a * o)
