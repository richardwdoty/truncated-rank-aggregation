from __future__ import annotations

from typing import Iterable

import numpy as np

from .statistic import _as_1d_float_array, statistic
from .backends.exact_dp import isf_exact, sf_exact


def sf(c: float, n: int, k: int, method: str = "exact") -> float:
    """
    Survival function S_{n:k}(c) = P(T_{n:k} > c) under the global null.
    """
    if method == "exact":
        return sf_exact(c, n, k)
    raise ValueError(f"Unknown method={method!r}.")


def cdf(c: float, n: int, k: int, method: str = "exact") -> float:
    """CDF of T_{n:k} under the global null."""
    return 1.0 - sf(c, n, k, method=method)


def isf(alpha: float, n: int, k: int, method: str = "exact") -> float:
    """Inverse survival: smallest c such that S_{n:k}(c) <= alpha."""
    if method == "exact":
        return isf_exact(alpha, n, k)
    raise ValueError(f"Unknown method={method!r}.")


def pvalue(pvals: Iterable[float] | np.ndarray, k: int, method: str = "exact") -> float:
    """
    Compute a p-value for observed p-values by:

      1) computing t = T_{n:k}(pvals)
      2) returning S_{n:k}(t)

    Notes
    -----
    This function validates pvals and enforces 1 <= k <= n.
    """
    x = _as_1d_float_array(pvals)
    n = int(x.size)
    t = statistic(x, k)
    return sf(t, n=n, k=k, method=method)