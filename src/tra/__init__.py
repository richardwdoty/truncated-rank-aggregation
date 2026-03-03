"""
Truncated Rank Aggregation (TRA)
"""

from .api import cdf, isf, pvalue, sf
from .statistic import statistic

__all__ = [
    "statistic",
    "sf",
    "cdf",
    "isf",
    "pvalue",
]