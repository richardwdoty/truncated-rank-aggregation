from __future__ import annotations

import math

import numpy as np
from scipy.stats import gamma


def _validate_k_only(k: int) -> int:
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an int; got {type(k)}.")
    k = int(k)
    if k <= 0:
        raise ValueError(f"k must be >= 1; got {k}.")
    return k


def _validate_c_scalar(c: float) -> float:
    if not np.isscalar(c):
        raise TypeError(f"c must be a scalar float in [0, 1]; got {type(c)}.")
    c = float(c)
    if not (0.0 <= c <= 1.0):
        raise ValueError(f"c must be in [0, 1]; got {c}.")
    return c


def gamma_thresholds_g_grid(c: np.ndarray, k: int) -> np.ndarray:
    """
    Vectorized computation of g_i(c) = G_i^{-1}(c), i=1,...,k, with g_0(c)=0.

    Parameters
    ----------
    c
        1D array of values in [0, 1].
    k
        Truncation level.

    Returns
    -------
    ndarray
        Array of shape (len(c), k+1) containing [g_0(c), ..., g_k(c)] row-wise.
    """
    k = _validate_k_only(k)
    c = np.asarray(c, dtype=float).reshape(-1)

    if np.any(c < 0.0) or np.any(c > 1.0):
        raise ValueError("All c values must lie in [0, 1].")

    g = np.empty((c.size, k + 1), dtype=float)
    g[:, 0] = 0.0

    i = np.arange(1, k + 1)
    g_inner = gamma.ppf(c[:, None], a=i[None, :], scale=1.0)

    # Numerical guardrails
    g_inner = np.maximum(g_inner, 0.0)
    g_inner = np.maximum.accumulate(g_inner, axis=1)

    g[:, 1:] = g_inner
    return g


def sf_asymptotic_grid(c: np.ndarray, k: int) -> np.ndarray:
    r"""
    Limiting fixed-k null survival L_k(c).

    Computes

        L_k(c) = exp(-g_k(c)) * P_k(lambda_2(c), ..., lambda_k(c)),

    where g_i(c) are Gamma(i,1) quantiles and the recursion states satisfy

        F_1(0) = 1,
        F_j(s) = sum_{y=0}^s F_{j-1}(s-y) * lambda_j(c)^y / y!,

    for j=2,...,k and s=0,...,j-1, with

        P_k = sum_{s=0}^{k-1} F_k(s).
    """
    k = _validate_k_only(k)
    c = np.asarray(c, dtype=float).reshape(-1)

    out = np.empty_like(c, dtype=float)

    out[c <= 0.0] = 1.0
    out[c >= 1.0] = 0.0

    mid = (c > 0.0) & (c < 1.0)
    if not np.any(mid):
        return out

    cc = c[mid]
    t_size = cc.size

    g = gamma_thresholds_g_grid(cc, k)   # shape (t_size, k+1)
    lam = np.diff(g, axis=1)             # shape (t_size, k), columns lambda_1,...,lambda_k

    if k == 1:
        out[mid] = np.exp(-g[:, 1])
        return out

    # F_1(0) = 1
    states = np.ones((t_size, 1), dtype=float)

    for j in range(2, k + 1):
        lam_j = lam[:, j - 1]  # lambda_j, shape (t_size,)
        new = np.zeros((t_size, j), dtype=float)

        # Precompute lam_j^y / y! for y=0,...,j-1
        weights = np.empty((t_size, j), dtype=float)
        weights[:, 0] = 1.0

        if j >= 2:
            term = np.ones(t_size, dtype=float)
            for y in range(1, j):
                term *= lam_j / y
                weights[:, y] = term

        for s in range(0, j):
            total = np.zeros(t_size, dtype=float)
            y_min = max(0, s - (j - 2))
            for y in range(y_min, s + 1):
                total += states[:, s - y] * weights[:, y]
            new[:, s] = total

        states = new

    poly_part = np.sum(states, axis=1)
    out[mid] = np.exp(-g[:, k]) * poly_part
    out = np.clip(out, 0.0, 1.0)
    return out


def sf_asymptotic(c: float, k: int) -> float:
    """
    Scalar wrapper for the limiting fixed-k null survival L_k(c).
    """
    c = _validate_c_scalar(c)
    return float(sf_asymptotic_grid(np.array([c], dtype=float), k)[0])