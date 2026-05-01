from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .api import cdf, isf, ppf, pvalue, sf, sf_grid, test
from .backends.asymptotic import _validate_k_only
from .backends.exact_dp import _validate_nk
from .thresholds import thresholds


@dataclass(frozen=True)
class TRADistribution:
    """
    Null distribution object for Truncated Rank Aggregation.

    For finite-n methods ("exact", "simplex"), both n and k are required.
    For the fixed-k asymptotic method, only k is required.
    """

    k: int
    n: int | None = None
    method: str = "exact"

    def __post_init__(self) -> None:
        if self.method in {"exact", "simplex"}:
            if self.n is None:
                raise ValueError(f"method={self.method!r} requires n.")
            n, k = _validate_nk(self.n, self.k)
            object.__setattr__(self, "n", n)
            object.__setattr__(self, "k", k)
            return

        if self.method == "asymptotic":
            k = _validate_k_only(self.k)
            object.__setattr__(self, "k", k)
            return

        raise ValueError(f"Unknown method={self.method!r}.")

    def sf(self, c: float) -> float:
        return sf(c, n=self.n, k=self.k, method=self.method)

    def sf_grid(self, c: Iterable[float] | np.ndarray) -> np.ndarray:
        return sf_grid(c, n=self.n, k=self.k, method=self.method)

    def cdf(self, c: float) -> float:
        return cdf(c, n=self.n, k=self.k, method=self.method)

    def isf(self, alpha: float) -> float:
        return isf(alpha, n=self.n, k=self.k, method=self.method)

    def ppf(self, alpha: float) -> float:
        return ppf(alpha, n=self.n, k=self.k, method=self.method)

    def critical_value(self, alpha: float) -> float:
        """
        Level-alpha critical value (lower-tail quantile) for the test statistic.

        Consistent with the package's lower-tail inference convention, the test
        rejects when the statistic is <= critical_value(alpha).
        """
        return self.ppf(alpha)

    def thresholds(self, alpha: float) -> np.ndarray:
        return thresholds(alpha, n=self.n, k=self.k, method=self.method)

    def pvalue(self, pvals: Iterable[float] | np.ndarray) -> float:
        return pvalue(pvals, k=self.k, method=self.method)

    def test(self, pvals: Iterable[float] | np.ndarray):
        return test(pvals, k=self.k, method=self.method)


def null_dist(k: int, n: int | None = None, method: str = "exact") -> TRADistribution:
    """
    Construct a TRA null distribution object.

    Examples
    --------
    exact finite-n:
        null_dist(n=100, k=5, method="exact")

    asymptotic fixed-k:
        null_dist(k=5, method="asymptotic")
    """
    return TRADistribution(k=k, n=n, method=method)
