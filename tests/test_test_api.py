import numpy as np

import tra


def test_test_matches_statistic_and_pvalue():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=40)
    k = 5

    res = tra.test(p, k)
    t = tra.statistic(p, k)
    pv = tra.pvalue(p, k)

    assert res.statistic == t
    assert res.pvalue == pv
    assert res.n == 40
    assert res.k == 5


def test_test_result_fields_in_range():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, size=25)

    res = tra.test(p, k=4)

    assert 0.0 <= res.statistic <= 1.0
    assert 0.0 <= res.pvalue <= 1.0


def test_pvalue_consistency_with_cdf():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, size=30)
    k = 6
    n = 30

    t = tra.statistic(p, k)
    pv = tra.pvalue(p, k)

    # p-value should be exactly the lower-tail cdf evaluated at the statistic
    assert np.isclose(pv, tra.cdf(t, n=n, k=k))


def test_pvalue_ordering_for_extreme_inputs():
    # Smaller p-values are more extreme for TRA.
    # Therefore, a test with smaller p-values should yield a smaller statistic
    # and a smaller (more significant) TRA p-value.
    p_mild = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    p_extreme = np.array([0.01, 0.02, 0.03, 0.4, 0.5])
    k = 3

    pv_mild = tra.pvalue(p_mild, k)
    pv_extreme = tra.pvalue(p_extreme, k)

    assert pv_extreme < pv_mild


def test_functional_vs_distribution_api():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, size=40)
    n, k = 40, 5

    dist = tra.null_dist(n=n, k=k)

    t = tra.statistic(p, k)

    # Check consistency of distribution methods with functional API
    assert np.isclose(dist.cdf(t), tra.cdf(t, n=n, k=k))
    assert np.isclose(dist.sf(t), tra.sf(t, n=n, k=k))
    assert np.isclose(dist.pvalue(p), tra.pvalue(p, k))

    alpha = 0.05
    assert np.isclose(dist.ppf(alpha), tra.ppf(alpha, n=n, k=k))
    assert np.isclose(dist.isf(alpha), tra.isf(alpha, n=n, k=k))
    assert np.allclose(dist.thresholds(alpha), tra.thresholds(alpha, n=n, k=k))
