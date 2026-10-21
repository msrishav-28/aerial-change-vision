import torch

from aerial_change.baselines import threshold_scores


def test_threshold_scores_identical_pair_is_zero():
    x = torch.rand(2, 3, 8, 8)
    pair = torch.stack([x, x], dim=1)
    assert torch.allclose(threshold_scores(pair), torch.zeros(2))


def test_threshold_scores_known_difference():
    t0 = torch.zeros(1, 3, 4, 4)
    t1 = torch.full((1, 3, 4, 4), 0.5)
    pair = torch.stack([t0, t1], dim=1)
    assert torch.allclose(threshold_scores(pair), torch.tensor([0.5]))


def test_threshold_scores_shape():
    pair = torch.rand(5, 2, 3, 8, 8)
    assert threshold_scores(pair).shape == (5,)
