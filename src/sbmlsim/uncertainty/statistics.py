"""The statistics of the draws of an uncertainty analysis, with their errors.

Every statistic of a Monte Carlo propagation has an error from the finite
number of draws, which is reported with it. The functions ignore the draws
which are not finite, e.g. of a point which failed, and return the number of
draws they used.

- `mean_se`: the mean and its standard error `s / sqrt(n)`
- `sd_se`: the standard deviation and its standard error by the delta method
  with the fourth central moment, `sqrt((m4 - s^4) / n) / (2 s)`
- `quantile_ci`: a quantile and its distribution-free confidence interval of
  order statistics, the ranks of the binomial distribution of the number of
  draws below the quantile (Hahn & Meeker 1991)

For quasi-random draws these errors are conservative: their convergence is
faster than that of random draws, which the errors assume.

Hahn GJ, Meeker WQ (1991). Statistical Intervals: A Guide for Practitioners.
Wiley.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def _finite(x: ArrayLike) -> np.ndarray:
    """Get the finite values of an array, flattened."""
    values = np.asarray(x, dtype=float).ravel()
    return values[np.isfinite(values)]


def mean_se(x: ArrayLike) -> tuple[float, float, int]:
    """Get the mean, its standard error and the number of draws.

    Args:
        x: the draws.

    Returns:
        The mean, its standard error and the number of finite draws; `nan`
        without draws (the error without two of them).
    """
    values = _finite(x)
    n = values.size
    if n == 0:
        return math.nan, math.nan, 0
    se = float(np.std(values, ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return float(np.mean(values)), se, n


def sd_se(x: ArrayLike) -> tuple[float, float, int]:
    """Get the standard deviation, its standard error and the number of draws.

    Args:
        x: the draws.

    Returns:
        The standard deviation, its standard error by the delta method and
        the number of finite draws; `nan` without two draws.
    """
    values = _finite(x)
    n = values.size
    if n < 2:
        return math.nan, math.nan, n
    sd = float(np.std(values, ddof=1))
    if sd == 0.0:
        return 0.0, 0.0, n
    m4 = float(np.mean((values - values.mean()) ** 4))
    se = math.sqrt(max(m4 - sd**4, 0.0) / n) / (2.0 * sd)
    return sd, se, n


def quantile_ci(
    x: ArrayLike, q: float, level: float = 0.95
) -> tuple[float, float, float, int]:
    """Get a quantile and its distribution-free confidence interval.

    The interval is the pair of order statistics `x_(r) <= x_(s)` whose ranks
    are the `(1 - level) / 2` and `(1 + level) / 2` quantiles of the binomial
    distribution `B(n, q)` of the number of draws below the quantile, which
    covers the quantile with at least about `level` for any continuous
    distribution (Hahn & Meeker 1991).

    Args:
        x: the draws.
        q: the probability of the quantile, in (0, 1).
        level: the level of the interval.

    Returns:
        The quantile, the lower and the upper end of its interval and the
        number of finite draws; an end beyond the draws is `nan`, e.g. the
        upper end of the 99 % quantile of 10 draws.
    """
    values = np.sort(_finite(x))
    n = values.size
    if n == 0:
        return math.nan, math.nan, math.nan, 0
    estimate = float(np.quantile(values, q))
    alpha = 1.0 - level
    low_rank = int(stats.binom.ppf(alpha / 2.0, n, q))
    high_rank = int(stats.binom.ppf(1.0 - alpha / 2.0, n, q)) + 1
    low = float(values[low_rank - 1]) if low_rank >= 1 else math.nan
    high = float(values[high_rank - 1]) if high_rank <= n else math.nan
    return estimate, low, high, n
