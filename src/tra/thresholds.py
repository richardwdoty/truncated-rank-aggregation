from __future__ import annotations

import numpy as np
from scipy.stats import beta

from .backends.exact_dp import _validate_nk


def thresholds(alpha: float, n: int | None = None, k: int | None = None, method: str = "exact") -> np.ndarray:
    """
    Compute ordered rejection thresholds for TRA.

    For a given significance level alpha, this returns the ordered set

        A = {a_1, ..., a_k}

    such that the TRA test rejects when

        P_(i) <= a_i  for some i <= k.

    Parameters
    ----------
    alpha
        Significance level in (0,1).
    n
        Number of p-values.
    k
        Truncation level.
    method
        Null evaluation method. Currently only "exact" is supported.

    Returns
    -------
    ndarray
        Array of thresholds [a_1, ..., a_k].

    Currently only the exact finite-n backend is supported.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0,1).")

    if method != "exact":
        raise ValueError(f"Unknown method={method!r}.")

    if n is None or k is None:
        raise ValueError("method='exact' requires both n and k.")

    n, k = _validate_nk(n, k)
    
    from .api import ppf

    c_alpha = ppf(alpha, n=n, k=k, method="exact")

    i = np.arange(1, k + 1)
    a = beta.ppf(c_alpha, i, n - i + 1)

    a = np.clip(a, 0.0, 1.0)
    a = np.maximum.accumulate(a)

    return a