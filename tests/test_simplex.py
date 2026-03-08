import numpy as np

import tra


def test_simplex_endpoints():
    n, k = 12, 4
    assert tra.sf(0.0, n, k, method="simplex") == 1.0
    assert tra.sf(1.0, n, k, method="simplex") == 0.0


def test_simplex_matches_exact_small_cases():
    test_cases = [
        (10, 4, 0.01),
        (10, 4, 0.10),
        (10, 4, 0.50),
        (20, 5, 0.05),
        (20, 5, 0.20),
    ]

    for n, k, c in test_cases:
        s_exact = tra.sf(c, n, k, method="exact")
        s_simplex = tra.sf(c, n, k, method="simplex")
        assert abs(s_exact - s_simplex) < 1e-13


def test_simplex_monotone():
    n, k = 15, 5
    cs = np.linspace(0.0, 1.0, 21)
    vals = np.array([tra.sf(float(c), n, k, method="simplex") for c in cs])
    assert np.all(np.diff(vals) <= 1e-12)