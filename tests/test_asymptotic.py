import numpy as np

import tra


def test_asymptotic_endpoints():
    k = 5
    assert tra.sf(0.0, k=k, method="asymptotic") == 1.0
    assert tra.sf(1.0, k=k, method="asymptotic") == 0.0


def test_asymptotic_grid_matches_scalar():
    k = 6
    cs = np.linspace(0.01, 0.99, 31)

    vals_grid = tra.sf_grid(cs, k=k, method="asymptotic")
    vals_scalar = np.array([tra.sf(float(c), k=k, method="asymptotic") for c in cs])

    assert np.allclose(vals_grid, vals_scalar, rtol=1e-12, atol=1e-14)


def test_asymptotic_monotone():
    k = 7
    cs = np.linspace(0.0, 1.0, 41)
    vals = tra.sf_grid(cs, k=k, method="asymptotic")
    assert np.all(np.diff(vals) <= 1e-12)


def test_asymptotic_approximates_exact_for_large_n():
    n, k = 2000, 5
    cs = np.array([0.05, 0.2, 0.5, 0.8])

    exact = tra.sf_grid(cs, n=n, k=k, method="exact")
    asymp = tra.sf_grid(cs, k=k, method="asymptotic")

    # Not machine-precision agreement; just sanity that the approximation is close.
    assert np.max(np.abs(exact - asymp)) < 5e-3


def test_asymptotic_distribution_object():
    dist = tra.null_dist(k=5, method="asymptotic")
    cs = np.linspace(0.1, 0.9, 9)

    vals_obj = dist.sf_grid(cs)
    vals_fun = tra.sf_grid(cs, k=5, method="asymptotic")

    assert np.allclose(vals_obj, vals_fun, rtol=1e-12, atol=1e-14)