"""
Truncated Rank Aggregation (TRA)
"""

from .api import cdf, isf, pvalue, sf, test
from .result import TRATestResult
from .statistic import statistic

__all__ = [
    "TRATestResult",
    "statistic",
    "sf",
    "cdf",
    "isf",
    "pvalue",
    "test",
]