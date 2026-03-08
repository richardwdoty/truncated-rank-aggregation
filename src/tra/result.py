from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TRATestResult:
    """
    Result of a Truncated Rank Aggregation test.

    Attributes
    ----------
    statistic
        Observed value of T_{n:k}.
    pvalue
        Null p-value P(T_{n:k} <= observed), equivalently sf(observed).
    n
        Number of p-values used in the test.
    k
        Truncation level used in the test.
    """

    statistic: float
    pvalue: float
    n: int
    k: int