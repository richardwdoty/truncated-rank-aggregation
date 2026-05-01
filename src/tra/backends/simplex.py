from __future__ import annotations

import math

from scipy.optimize import brentq
import numpy as np
from scipy.stats import beta

from .exact_dp import _validate_nk


def _validate_c_scalar(c: float) -> float:
    if not np.isscalar(c):
        raise TypeError(f"c must be a scalar float in [0, 1]; got {type(c)}.")
    c = float(c)
    if not (0.0 <= c <= 1.0):
        raise ValueError(f"c must be in [0, 1]; got {c}.")
    return c


def thresholds_a_simplex(c: np.ndarray, n: int, k: int) -> np.ndarray:
    """
    Vectorized threshold computation for the simplex backend.

    Parameters
    ----------
    c
        1D array of values in [0, 1].
    n, k
        TRA parameters.

    Returns
    -------
    ndarray
        Array of shape (m, k+1) containing [a_0(c), ..., a_k(c)] row-wise,
        with a_0(c) = 0.
    """
    c = np.asarray(c, dtype=float).reshape(-1)
    if np.any(c < 0.0) or np.any(c > 1.0):
        raise ValueError("All c values must lie in [0, 1].")

    a = np.empty((c.size, k + 1), dtype=float)
    a[:, 0] = 0.0

    if k == 0:
        return a

    i = np.arange(1, k + 1)
    a_inner = beta.ppf(c[:, None], i[None, :], (n - i + 1)[None, :])

    # Guardrails against tiny numerical irregularities
    a_inner = np.clip(a_inner, 0.0, 1.0)
    a_inner = np.maximum.accumulate(a_inner, axis=1)

    a[:, 1:] = a_inner
    return a


def sf_simplex_grid(c: np.ndarray, n: int, k: int) -> np.ndarray:
    r"""
    Exact null survival S_{n:k}(c) via the ordered-simplex representation.

    This implementation uses the recursion

        V_0(t) = 1,
        V_j(t) = \int_{a_j(c)}^t V_{j-1}(u)\,du,   j >= 1,

    with V_j(t)=0 for t < a_j(c), and the final representation

        S_{n:k}(c)
        = [n!/(n-k)!] \int_{a_k(c)}^1 (1-t)^{n-k} V_{k-1}(t)\,dt.

    The key computational strategy is:
    - construct the polynomial coefficients of V_{k-1}(t) on [a_k(c), 1]
    - integrate term-by-term against t^j (1-t)^(n-k) using a stable recurrence
      for the tail moments \int_z^1 t^j (1-t)^m dt.

    Parameters
    ----------
    c
        1D array-like of values in [0, 1].
    n, k
        TRA parameters.

    Returns
    -------
    ndarray
        Survival values of shape (m,).
    """
    n, k = _validate_nk(n, k)
    c = np.asarray(c, dtype=float).reshape(-1)

    out = np.empty_like(c, dtype=float)

    # Endpoints
    out[c <= 0.0] = 1.0
    out[c >= 1.0] = 0.0

    mid = (c > 0.0) & (c < 1.0)
    if not np.any(mid):
        return out

    cc = c[mid]
    zmat = thresholds_a_simplex(cc, n, k)
    z = zmat[:, k]  # a_k(c)

    # Build polynomial coefficients for V_{k-1}(t) on [a_k(c), 1]
    # coeffs[row, j] = coefficient of t^j
    coeffs = np.ones((cc.size, 1), dtype=float)  # V_0(t) = 1

    for j in range(1, k):
        aj = zmat[:, j]
        deg = coeffs.shape[1] - 1

        antideriv = np.zeros((cc.size, deg + 2), dtype=float)
        antideriv[:, 1:] = coeffs / np.arange(1, deg + 2, dtype=float)[None, :]

        # Evaluate antiderivative at a_j row-wise via Horner, then shift constant
        val = np.zeros(cc.size, dtype=float)
        for p in range(antideriv.shape[1] - 1, -1, -1):
            val = val * aj + antideriv[:, p]
        antideriv[:, 0] -= val

        coeffs = antideriv

    d = coeffs.shape[1] - 1
    m = n - k
    b = float(m + 1)

    # Tail moments I_j(z) = ∫_z^1 t^j (1-t)^m dt
    tail = np.exp(b * np.log1p(-z))  # (1-z)^(m+1)
    I = tail / b  # noqa: E741
    I_all = np.empty((cc.size, d + 1), dtype=float)
    I_all[:, 0] = I

    # Recurrence:
    # I_{j+1} = (a/(a+b)) I_j + z^a (1-z)^b / (a+b), where a = j+1
    pow_z = z.copy()
    for j in range(0, d):
        a_par = float(j + 1)
        denom = a_par + b
        I = (a_par / denom) * I + (pow_z * tail) / denom  # noqa: E741
        I_all[:, j + 1] = I
        pow_z *= z

    total_int = np.sum(coeffs * I_all, axis=1)

    log_prefactor = math.lgamma(n + 1) - math.lgamma(n - k + 1)
    prefactor = math.exp(log_prefactor)

    out_mid = prefactor * total_int
    out_mid = np.clip(out_mid, 0.0, 1.0)

    out[mid] = out_mid
    return out


def sf_simplex(c: float, n: int, k: int) -> float:
    """
    Scalar wrapper for the ordered-simplex exact survival backend.
    """
    c = _validate_c_scalar(c)
    return float(sf_simplex_grid(np.array([c], dtype=float), n, k)[0])


def isf_simplex(alpha: float, n: int, k: int) -> float:
    """
    Inverse survival for the simplex null.
    """
    n, k = _validate_nk(n, k)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1]; got {alpha}.")

    if alpha >= 1.0:
        return 0.0
    if alpha <= 0.0:
        return 1.0

    def f(c: float) -> float:
        return sf_simplex(c, n, k) - alpha

    return float(brentq(f, 0.0, 1.0, xtol=1e-12, rtol=1e-12, maxiter=200))
