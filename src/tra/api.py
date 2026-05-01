from __future__ import annotations

from typing import Iterable

import numpy as np

from .backends.asymptotic import sf_asymptotic, sf_asymptotic_grid
from .backends.exact_dp import _validate_nk, isf_exact, sf_exact, sf_exact_grid
from .result import TRATestResult
from .statistic import _as_1d_float_array, statistic


def sf(
    c: float, n: int | None = None, k: int | None = None, method: str = "exact"
) -> float:
    """
    Survival function under the global null.

    Methods
    -------
    exact
        Finite-n exact DP backend. Requires n and k.
    simplex
        Finite-n exact ordered-simplex backend. Requires n and k.
    asymptotic
        Fixed-k limiting survival L_k(c). Requires k only.
    """
    if method == "exact":
        if n is None or k is None:
            raise ValueError("method='exact' requires both n and k.")
        return sf_exact(c, n, k)

    if method == "simplex":
        if n is None or k is None:
            raise ValueError("method='simplex' requires both n and k.")
        from .backends.simplex import sf_simplex

        return sf_simplex(c, n, k)

    if method == "asymptotic":
        if k is None:
            raise ValueError("method='asymptotic' requires k.")
        return sf_asymptotic(c, k)

    raise ValueError(f"Unknown method={method!r}.")


def sf_grid(
    c: Iterable[float] | np.ndarray,
    n: int | None = None,
    k: int | None = None,
    method: str = "exact",
) -> np.ndarray:
    """
    Survival function evaluated over a grid of c values.
    """
    c = np.asarray(c, dtype=float).reshape(-1)

    if method == "exact":
        if n is None or k is None:
            raise ValueError("method='exact' requires both n and k.")
        return sf_exact_grid(c, n, k)

    if method == "simplex":
        if n is None or k is None:
            raise ValueError("method='simplex' requires both n and k.")
        from .backends.simplex import sf_simplex_grid

        return sf_simplex_grid(c, n, k)

    if method == "asymptotic":
        if k is None:
            raise ValueError("method='asymptotic' requires k.")
        return sf_asymptotic_grid(c, k)

    raise ValueError(f"Unknown method={method!r}.")


def cdf(
    c: float, n: int | None = None, k: int | None = None, method: str = "exact"
) -> float:
    """CDF under the global null."""
    return 1.0 - sf(c, n=n, k=k, method=method)


def isf(
    alpha: float, n: int | None = None, k: int | None = None, method: str = "exact"
) -> float:
    """Inverse survival."""
    if method == "exact":
        if n is None or k is None:
            raise ValueError("method='exact' requires both n and k.")
        return isf_exact(alpha, n, k)

    if method == "simplex":
        if n is None or k is None:
            raise ValueError("method='simplex' requires both n and k.")
        from .backends.simplex import isf_simplex

        return isf_simplex(alpha, n, k)

    if method == "asymptotic":
        if k is None:
            raise ValueError("method='asymptotic' requires k.")
        from .backends.asymptotic import isf_asymptotic

        return isf_asymptotic(alpha, k)

    raise ValueError(f"Unknown method={method!r}.")


def ppf(
    alpha: float, n: int | None = None, k: int | None = None, method: str = "exact"
) -> float:
    """Percent point function (inverse of CDF)."""
    return isf(1.0 - alpha, n=n, k=k, method=method)


def pvalue(pvals: Iterable[float] | np.ndarray, k: int, method: str = "exact") -> float:
    """
    Compute a p-value for observed p-values by:

      1) computing t = T_{n:k}(pvals)
      2) returning the null CDF at t
    """
    x = _as_1d_float_array(pvals)
    n = int(x.size)
    _validate_nk(n, k)
    t = statistic(x, k)

    return cdf(t, n=n, k=k, method=method)


def test(
    pvals: Iterable[float] | np.ndarray,
    k: int,
    method: str = "exact",
) -> TRATestResult:
    """
    Run the Truncated Rank Aggregation test.
    """
    x = _as_1d_float_array(pvals)
    n = int(x.size)
    _validate_nk(n, k)

    t = statistic(x, k)
    pv = cdf(t, n=n, k=k, method=method)

    return TRATestResult(statistic=t, pvalue=pv, n=n, k=k)
