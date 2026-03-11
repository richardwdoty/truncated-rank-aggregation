from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .api import isf, pvalue, sf, sf_grid, test
from .backends.exact_dp import _validate_nk
from .thresholds import thresholds


@dataclass(frozen=True)
class TRADistribution:
    """
    Null distribution object for Truncated Rank Aggregation.

    Parameters
    ----------
    n
        Number of p-values.
    k
        Truncation level.
    method
        Null evaluation method.
    """

    n: int
    k: int
    method: str = "exact"

    def __post_init__(self) -> None:
        n, k = _validate_nk(self.n, self.k)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "k", k)

    def sf(self, c: float) -> float:
        """
        Survival function S_{n:k}(c).
        """
        return sf(c, n=self.n, k=self.k, method=self.method)

    def sf_grid(self, c: Iterable[float] | np.ndarray) -> np.ndarray:
        """
        Survival function S_{n:k}(c) evaluated over a grid.
        """
        return sf_grid(c, n=self.n, k=self.k, method=self.method)

    def isf(self, alpha: float) -> float:
        """
        Inverse survival function.
        """
        return isf(alpha, n=self.n, k=self.k, method=self.method)

    def thresholds(self, alpha: float) -> np.ndarray:
        """
        Rank-wise rejection thresholds a_1, ..., a_k.
        """
        return thresholds(alpha, n=self.n, k=self.k, method=self.method)

    def pvalue(self, pvals: Iterable[float] | np.ndarray) -> float:
        """
        Compute the null p-value for an observed vector of p-values.
        """
        return pvalue(pvals, k=self.k, method=self.method)

    def test(self, pvals: Iterable[float] | np.ndarray):
        """
        Run the TRA test on an observed vector of p-values.
        """
        return test(pvals, k=self.k, method=self.method)


def null_dist(n: int, k: int, method: str = "exact") -> TRADistribution:
    """
    Construct a TRA null distribution object.

    Parameters
    ----------
    n
        Number of p-values.
    k
        Truncation level.
    method
        Null evaluation method.

    Returns
    -------
    TRADistribution
        Distribution object with methods for sf, sf_grid, isf, thresholds,
        pvalue, and test.
    """
    return TRADistribution(n=n, k=k, method=method)