"""
Truncated Rank Aggregation (TRA)

Python implementation of the truncated rank aggregation statistic and
its exact and asymptotic null distributions.
"""

from .api import sf, cdf, isf, pvalue

__all__ = ["sf", "cdf", "isf", "pvalue"]