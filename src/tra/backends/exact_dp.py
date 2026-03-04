from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta
from scipy.optimize import brentq


@dataclass(frozen=True)
class ExactDPResult:
    """Container for exact DP survival evaluation (mostly for debugging/tests)."""

    a: np.ndarray  # thresholds a_0..a_k
    q: np.ndarray  # conditional probs q_0..q_{k-1}
    dp_last: np.ndarray  # P_{k-1}(s), s=0..k-1
    sf: float


def _validate_nk(n: int, k: int) -> tuple[int, int]:
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"n must be an int; got {type(n)}.")
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an int; got {type(k)}.")
    n = int(n)
    k = int(k)
    if n <= 0:
        raise ValueError(f"n must be >= 1; got {n}.")
    if k <= 0:
        raise ValueError(f"k must be >= 1; got {k}.")
    if k > n:
        raise ValueError(f"k must be <= n (n={n}); got k={k}.")
    return n, k


def _log_binom_pmf(x: int, n: int, p: float) -> float:
    """log P(Bin(n,p)=x) computed stably via lgamma."""
    if x < 0 or x > n:
        return -math.inf
    if p == 0.0:
        return 0.0 if x == 0 else -math.inf
    if p == 1.0:
        return 0.0 if x == n else -math.inf
    # log(n choose x) + x log p + (n-x) log(1-p)
    return (
        math.lgamma(n + 1)
        - math.lgamma(x + 1)
        - math.lgamma(n - x + 1)
        + x * math.log(p)
        + (n - x) * math.log1p(-p)
    )


def thresholds_a(c: float, n: int, k: int) -> np.ndarray:
    """
    Compute thresholds a_i(c) = F_{n:i}^{-1}(c) for i=1..k, with a_0(c)=0.

    Here F_{n:i} is the CDF of Beta(i, n-i+1) under the global null.
    """
    n, k = _validate_nk(n, k)
    if not (0.0 <= c <= 1.0):
        raise ValueError(f"c must be in [0,1]; got {c}.")

    a = np.empty(k + 1, dtype=float)
    a[0] = 0.0
    if c == 0.0:
        a[1:] = 0.0
        return a
    if c == 1.0:
        a[1:] = 1.0
        return a

    for i in range(1, k + 1):
        a[i] = float(beta.ppf(c, i, n - i + 1))
    # Numerical guard
    a = np.clip(a, 0.0, 1.0)
    return a


def q_from_thresholds(a: np.ndarray) -> np.ndarray:
    """
    Compute q_j = pi_j / (1 - a_j) for j=0..k-1, where pi_j = a_{j+1}-a_j.
    Implements the convention q_j=0 if a_j=1.
    """
    if a.ndim != 1:
        raise ValueError("a must be a 1D array.")
    if a.size < 2:
        raise ValueError("a must have length at least 2.")
    k = a.size - 1

    q = np.empty(k, dtype=float)
    for j in range(0, k):
        denom = 1.0 - a[j]
        if denom <= 0.0:
            q[j] = 0.0
        else:
            pi_j = a[j + 1] - a[j]
            # pi_j should be >=0; small negative drift gets clipped
            if pi_j < 0.0 and pi_j > -1e-15:
                pi_j = 0.0
            if pi_j < 0.0:
                raise ValueError("Thresholds a must be nondecreasing.")
            q[j] = pi_j / denom

    # Numerical guard
    q = np.clip(q, 0.0, 1.0)
    return q


def sf_exact(c: float, n: int, k: int, *, return_details: bool = False) -> float | ExactDPResult:
    r"""
    Exact null survival S_{n:k}(c) = P(T_{n:k} > c) via the DP recursion in the paper.

    Uses thresholds a_i(c) and conditional bin probabilities q_j as defined in the manuscript,
    then evaluates:

      P_0(0) = (1-q_0)^n
      P_j(s) = sum_{r=0}^{min(j-1,s)} P_{j-1}(r) * BinPMF(n-r, q_j; s-r),   j=1..k-1, s=0..j
      S_{n:k}(c) = sum_{s=0}^{k-1} P_{k-1}(s)

    where BinPMF(m, q; x) = C(m,x) q^x (1-q)^(m-x).
    """
    n, k = _validate_nk(n, k)
    if not (0.0 <= c <= 1.0):
        raise ValueError(f"c must be in [0,1]; got {c}.")

    a = thresholds_a(c, n, k)
    q = q_from_thresholds(a)

    # DP base: j=0, only state s=0 is possible under caps
    dp_prev = np.zeros(1, dtype=float)
    dp_prev[0] = (1.0 - q[0]) ** n  # P(S0=0 and S0<=0)

    if k == 1:
        sf = float(dp_prev[0])
        if return_details:
            return ExactDPResult(a=a, q=q, dp_last=dp_prev.copy(), sf=sf)
        return sf

    # Transitions: j=1..k-1
    for j in range(1, k):
        dp_cur = np.zeros(j + 1, dtype=float)
        qj = float(q[j])

        # s ranges 0..j (cap S_j <= j)
        for s in range(0, j + 1):
            r_max = min(j - 1, s)
            total = 0.0
            for r in range(0, r_max + 1):
                # X_j = s-r ~ Bin(n-r, qj)
                m = n - r
                x = s - r
                lp = _log_binom_pmf(x, m, qj)
                if lp == -math.inf:
                    continue
                total += dp_prev[r] * math.exp(lp)
            dp_cur[s] = total

        dp_prev = dp_cur

    sf = float(np.sum(dp_prev))
    # Guard against tiny drift
    if sf < 0.0 and sf > -1e-15:
        sf = 0.0
    if sf > 1.0 and sf < 1.0 + 1e-15:
        sf = 1.0

    if return_details:
        return ExactDPResult(a=a, q=q, dp_last=dp_prev.copy(), sf=sf)
    return sf


def isf_exact(alpha: float, n: int, k: int) -> float:
    """
    Inverse survival for the exact null: smallest c in [0,1] such that S_{n:k}(c) <= alpha.
    """
    n, k = _validate_nk(n, k)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1]; got {alpha}.")

    # Boundary behavior:
    # S(0)=1, S(1)=0
    if alpha >= 1.0:
        return 0.0
    if alpha <= 0.0:
        return 1.0

    def f(c: float) -> float:
        return sf_exact(c, n, k) - alpha

    # Guaranteed sign change: f(0)=1-alpha>0, f(1)=0-alpha<0
    return float(brentq(f, 0.0, 1.0, xtol=1e-12, rtol=1e-12, maxiter=200))