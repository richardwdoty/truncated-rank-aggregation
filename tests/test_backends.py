import numpy as np
import pytest

import tra

METHODS = ["exact", "simplex", "asymptotic"]


@pytest.mark.parametrize("method", METHODS)
def test_functional_api_isf_ppf_inversion(method):
    k = 5
    n = 20 if method != "asymptotic" else None

    alpha = 0.05
    c_alpha = tra.isf(alpha, n=n, k=k, method=method)

    # Check bounds
    assert 0.0 <= c_alpha <= 1.0

    # Check inversion sf(isf(alpha)) == alpha
    alpha_rec = tra.sf(c_alpha, n=n, k=k, method=method)
    assert np.isclose(alpha_rec, alpha, atol=1e-7)

    # Check ppf: ppf(alpha) = isf(1-alpha) => cdf(ppf(alpha)) = alpha
    q_alpha = tra.ppf(alpha, n=n, k=k, method=method)
    alpha_cdf = tra.cdf(q_alpha, n=n, k=k, method=method)
    assert np.isclose(alpha_cdf, alpha, atol=1e-7)


@pytest.mark.parametrize("method", METHODS)
def test_functional_api_boundaries(method):
    k = 5
    n = 20 if method != "asymptotic" else None

    assert tra.isf(1.0, n=n, k=k, method=method) == 0.0
    assert tra.isf(0.0, n=n, k=k, method=method) == 1.0
    assert tra.ppf(0.0, n=n, k=k, method=method) == 0.0
    assert tra.ppf(1.0, n=n, k=k, method=method) == 1.0


@pytest.mark.parametrize("method", METHODS)
def test_thresholds_bounds_monotonicity(method):
    k = 8
    n = 30 if method != "asymptotic" else None

    a = tra.thresholds(0.05, n=n, k=k, method=method)
    assert len(a) == k

    # Lower bounds
    assert np.all(a >= 0.0)

    # Finite-n thresholds should be <= 1.0
    if method != "asymptotic":
        assert np.all(a <= 1.0)

    # Monotonicity
    assert np.all(np.diff(a) >= 0.0)


@pytest.mark.parametrize("method", METHODS)
def test_distribution_object(method):
    k = 6
    n = 40 if method != "asymptotic" else None

    dist = tra.null_dist(n=n, k=k, method=method)
    alpha = 0.05

    # Check isf
    c_alpha = dist.isf(alpha)
    assert np.isclose(dist.sf(c_alpha), alpha, atol=1e-7)

    # Check critical_value and ppf
    crit = dist.critical_value(alpha)
    q_alpha = dist.ppf(alpha)
    assert crit == q_alpha
    assert np.isclose(dist.cdf(crit), alpha, atol=1e-7)

    # Lower-tail orientation: smaller c means smaller cdf.
    # So if we test at alpha/2, the critical value should be smaller than at alpha.
    crit_smaller = dist.critical_value(alpha / 2)
    assert crit_smaller < crit

    # Check thresholds
    th = dist.thresholds(alpha)
    assert len(th) == k
    assert np.all(np.diff(th) >= 0.0)


def test_backend_comparison_exact_simplex():
    # Exact and simplex should show close finite-n agreement
    n, k = 15, 4
    alpha = 0.05

    c_exact = tra.isf(alpha, n=n, k=k, method="exact")
    c_simplex = tra.isf(alpha, n=n, k=k, method="simplex")
    assert np.isclose(c_exact, c_simplex, atol=1e-6)

    th_exact = tra.thresholds(alpha, n=n, k=k, method="exact")
    th_simplex = tra.thresholds(alpha, n=n, k=k, method="simplex")
    assert np.allclose(th_exact, th_simplex, atol=1e-6)


def test_backend_comparison_asymptotic_convergence():
    # Asymptotic backend should agree more closely with exact as n increases
    k = 3
    alpha = 0.05

    c_asymp = tra.isf(alpha, k=k, method="asymptotic")

    c_exact_small = tra.isf(alpha, n=20, k=k, method="exact")
    c_exact_large = tra.isf(alpha, n=1000, k=k, method="exact")

    diff_small = abs(c_exact_small - c_asymp)
    diff_large = abs(c_exact_large - c_asymp)

    assert diff_large < diff_small
