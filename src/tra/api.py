"""
Public API for Truncated Rank Aggregation.
"""


def sf(c, n, k, method="exact"):
    """
    Survival function P(T_{n:k} > c).

    Parameters
    ----------
    c : float
        Threshold value.
    n : int
        Number of p-values.
    k : int
        Truncation level.
    method : {"exact", "asymptotic"}
        Evaluation method.

    Returns
    -------
    float
        Survival probability.
    """
    raise NotImplementedError


def cdf(c, n, k, method="exact"):
    """CDF of the statistic."""
    return 1.0 - sf(c, n, k, method=method)


def isf(alpha, n, k, method="exact"):
    """
    Inverse survival function.
    """
    raise NotImplementedError


def pvalue(t_obs, n, k, method="exact"):
    """
    Compute p-value for observed statistic.
    """
    return sf(t_obs, n, k, method=method)