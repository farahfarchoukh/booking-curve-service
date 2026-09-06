"""
The empirical-Bayes shrinkage weight is the mechanism DESIGN.md §6.1/§6.2
leans on for both "one model or many" and cold start. If its formula is
wrong, both arguments are wrong, so it gets a direct unit test rather than
only being exercised indirectly through a trained model.
"""

from src.model import _shrink_weight


def test_zero_observations_means_zero_trust_in_own_data():
    # A brand-new hotel: the backbone prediction should be used as-is.
    assert _shrink_weight(0, 15.0) == 0.0


def test_weight_increases_monotonically_with_n():
    weights = [_shrink_weight(n, 15.0) for n in (0, 5, 15, 50, 500)]
    assert weights == sorted(weights)
    assert all(0.0 <= w <= 1.0 for w in weights)


def test_weight_approaches_one_for_large_n():
    assert _shrink_weight(1_000_000, 15.0) > 0.999


def test_n_equal_k_gives_half_weight():
    assert abs(_shrink_weight(15.0, 15.0) - 0.5) < 1e-9


def test_larger_k_means_slower_trust_for_the_same_n():
    # This is the actual design choice behind shape trusting the pool more
    # than level does (SHAPE_SHRINK_K=30 > LEVEL_SHRINK_K=15 in train.py).
    n = 20.0
    assert _shrink_weight(n, 30.0) < _shrink_weight(n, 15.0)
