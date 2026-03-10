"""
Truncated Rank Aggregation (TRA)
"""

from .api import cdf, isf, pvalue, sf, sf_grid, test
from .result import TRATestResult
from .statistic import statistic
from .thresholds import thresholds

__all__ = [
    "TRATestResult",
    "statistic",
    "sf",
    "sf_grid",
    "cdf",
    "isf",
    "pvalue",
    "test",
    "thresholds",
]