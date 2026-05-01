import numpy as np

import tra


def test_null_dist_stores_parameters():
    dist = tra.null_dist(n=50, k=6)
    assert dist.n == 50
    assert dist.k == 6
    assert dist.method == "exact"


def test_distribution_sf_matches_functional_api():
    dist = tra.null_dist(n=40, k=5)
    c = 0.1
    assert dist.sf(c) == tra.sf(c, n=40, k=5)


def test_distribution_cdf_matches_functional_api():
    dist = tra.null_dist(n=40, k=5)
    c = 0.1
    assert dist.cdf(c) == tra.cdf(c, n=40, k=5)


def test_distribution_sf_grid_matches_functional_api():
    dist = tra.null_dist(n=30, k=4)
    cs = np.linspace(0.01, 0.99, 25)
    vals_obj = dist.sf_grid(cs)
    vals_fun = tra.sf_grid(cs, n=30, k=4)
    assert np.allclose(vals_obj, vals_fun, rtol=1e-12, atol=1e-14)


def test_distribution_isf_matches_functional_api():
    dist = tra.null_dist(n=60, k=7)
    alpha = 0.05
    assert dist.isf(alpha) == tra.isf(alpha, n=60, k=7)


def test_distribution_ppf_matches_functional_api():
    dist = tra.null_dist(n=60, k=7)
    alpha = 0.05
    assert dist.ppf(alpha) == tra.ppf(alpha, n=60, k=7)


def test_distribution_thresholds_matches_functional_api():
    dist = tra.null_dist(n=80, k=8)
    alpha = 0.05
    a_obj = dist.thresholds(alpha)
    a_fun = tra.thresholds(alpha, n=80, k=8)
    assert np.allclose(a_obj, a_fun, rtol=1e-12, atol=1e-14)


def test_distribution_test_matches_functional_api():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=50)

    dist = tra.null_dist(n=50, k=5)
    res_obj = dist.test(p)
    res_fun = tra.test(p, k=5)

    assert res_obj.statistic == res_fun.statistic
    assert res_obj.pvalue == res_fun.pvalue
    assert res_obj.n == res_fun.n
    assert res_obj.k == res_fun.k


def test_distribution_pvalue_matches_functional_api():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, size=40)

    dist = tra.null_dist(n=40, k=4)
    pv_obj = dist.pvalue(p)
    pv_fun = tra.pvalue(p, k=4)

    assert pv_obj == pv_fun