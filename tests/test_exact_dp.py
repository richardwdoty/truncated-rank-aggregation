import numpy as np

import tra


def test_sf_endpoints():
    n, k = 25, 5
    assert tra.sf(0.0, n, k) == 1.0
    assert tra.sf(1.0, n, k) == 0.0


def test_sf_monotone():
    n, k = 30, 6
    cs = np.linspace(0.0, 1.0, 21)
    vals = np.array([tra.sf(float(c), n, k) for c in cs])
    # Survival should be nonincreasing
    assert np.all(np.diff(vals) <= 1e-12)


def test_k1_matches_uniform_survival():
    n, k = 40, 1
    cs = np.linspace(0.0, 1.0, 11)
    for c in cs:
        assert abs(tra.sf(float(c), n, k) - (1.0 - float(c))) < 1e-10


def test_isf_inverts_sf():
    n, k = 50, 4
    alphas = [0.9, 0.5, 0.1, 0.01]
    for a in alphas:
        c = tra.isf(a, n, k)
        s = tra.sf(c, n, k)
        # We want S(c) <= alpha, and just below should be >= alpha (up to numeric tolerance)
        assert s <= a + 5e-10
        c_lo = max(0.0, c - 1e-6)
        s_lo = tra.sf(c_lo, n, k)
        assert s_lo >= a - 5e-6


def test_pvalue_pipeline_runs():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=60)
    pv = tra.pvalue(p, k=5)
    assert 0.0 <= pv <= 1.0