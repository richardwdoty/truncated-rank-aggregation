import numpy as np

import tra


def test_threshold_shape():
    a = tra.thresholds(alpha=0.05, n=100, k=5)
    assert a.shape == (5,)


def test_threshold_bounds():
    a = tra.thresholds(alpha=0.05, n=50, k=6)
    assert np.all(a >= 0.0)
    assert np.all(a <= 1.0)


def test_threshold_monotone():
    a = tra.thresholds(alpha=0.05, n=60, k=8)
    assert np.all(np.diff(a) >= 0.0)


def test_threshold_consistency_with_isf():
    n, k, alpha = 40, 5, 0.05

    c = tra.isf(alpha, n, k)
    a = tra.thresholds(alpha, n, k)

    from scipy.stats import beta

    i = np.arange(1, k + 1)
    a_expected = beta.ppf(c, i, n - i + 1)

    assert np.allclose(a, a_expected)


def test_threshold_alpha_monotonicity():
    n, k = 100, 6

    a_lo = tra.thresholds(0.01, n, k)
    a_hi = tra.thresholds(0.10, n, k)

    # larger alpha → less stringent thresholds
    assert np.all(a_lo >= a_hi)