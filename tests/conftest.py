import pytest
import torch


@pytest.fixture
def batch():
    """A small synthetic batch covering explicit, implicit (NULL) and padded quadruplets.

    Token layout: index 0 leading special token, index 1 NULL slot, text from index 2.
    """
    torch.manual_seed(0)
    B, L, H, Q = 2, 12, 16, 4
    hidden = torch.randn(B, L, H)
    spans = torch.tensor(
        [
            [[2, 3, 5, 5], [1, 1, 7, 8], [4, 4, 1, 1], [-1, -1, -1, -1]],  # explicit, NULL aspect, NULL opinion, pad
            [[6, 6, 2, 4], [1, 1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1]],  # explicit, both NULL, pad, pad
        ]
    )
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    gold = torch.tensor(
        [
            [[7.0, 6.0], [3.0, 5.5], [6.5, 4.0], [0.0, 0.0]],
            [[2.5, 7.0], [5.0, 5.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    return hidden, spans, mask, gold
