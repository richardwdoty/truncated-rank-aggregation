from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.special import betainc


def _as_1d_float_array(pvals: Iterable[float] | np.ndarray) -> np.ndarray:
    """
    Convert input to a 1D float NumPy array.

    Accepts common vector shapes such as (n,), (n,1), or (1,n).
    Raises for truly non-vector inputs.
    """
    if isinstance(pvals, np.ndarray):
        arr = np.asarray(pvals, dtype=float)
    else:
        # For general iterables (including generators), materialize once.
        arr = np.asarray(list(pvals), dtype=float)

    if arr.size == 0:
        raise ValueError("pvals must be non-empty.")

    if arr.ndim != 1:
        squeezed = np.squeeze(arr)
        if squeezed.ndim != 1:
            raise ValueError(f"pvals must be a 1D vector; got shape {arr.shape}.")
        arr = squeezed.astype(float, copy=False)

    return arr


def _validate_k(k: int, n: int) -> int:
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an int; got {type(k)}.")
    k = int(k)
    if k <= 0:
        raise ValueError(f"k must be >= 1; got {k}.")
    if k > n:
        raise ValueError(f"k must be <= n (n={n}); got k={k}.")
    return k


def statistic(pvals: Iterable[float] | np.ndarray, k: int) -> float:
    r"""
    Compute the Truncated Rank Aggregation statistic

        T_{n:k} = min_{1 <= i <= k} R_i,

    where R_i = F_{n:i}(P_(i)) is the PIT of the i-th order statistic under
    the global null, and F_{n:i} is the CDF of Beta(i, n-i+1).

    Parameters
    ----------
    pvals
        A 1D vector of p-values in [0, 1]. NaNs are not allowed.
        Common shapes (n,), (n,1), (1,n) are accepted.
    k
        Truncation level, must satisfy 1 <= k <= n.

    Returns
    -------
    float
        Observed statistic value in [0, 1].

    Raises
    ------
    ValueError
        If pvals is empty, contains NaNs, or contains values outside [0, 1],
        or if k is not in [1, n].
    TypeError
        If k is not an integer type.

    Notes
    -----
    The PIT uses the regularized incomplete beta function:

        F_{n:i}(x) = I_x(i, n-i+1),

    computed via scipy.special.betainc.
    """
    x = _as_1d_float_array(pvals)
    n = int(x.size)
    k = _validate_k(k, n)

    if np.isnan(x).any():
        raise ValueError("pvals contains NaN value(s).")

    if (x < 0.0).any() or (x > 1.0).any():
        bad = x[(x < 0.0) | (x > 1.0)]
        sample = bad[:5]
        raise ValueError(
            "pvals must lie in [0, 1]. "
            f"Found {bad.size} out-of-bounds value(s); sample={sample}."
        )

    xs = np.sort(x)
    xs_k = xs[:k]

    i = np.arange(1, k + 1, dtype=float)
    a = i
    b = (n - i + 1).astype(float)

    r_i = betainc(a, b, xs_k)

    # Numerical guard
    r_i = np.clip(r_i, 0.0, 1.0)

    return float(np.min(r_i))