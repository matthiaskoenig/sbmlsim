"""The statistics of the draws of an uncertainty analysis, with their errors.

Every statistic of a Monte Carlo propagation has an error from the finite
number of draws, which is reported with it. The functions ignore the draws
which are `NaN`, i.e. of a point which failed, and return the number of draws
they used; an infinite draw is a value (the mean of draws with one is
infinite, their standard deviation `NaN`).

- `mean_se`: the mean and its standard error `s / sqrt(n)`
- `sd_se`: the standard deviation and its standard error by the delta method
  with the fourth central moment, `sqrt((m4 - s^4) / n) / (2 s)`
- `quantile_ci`: a quantile and its distribution-free confidence interval of
  order statistics, the ranks of the binomial distribution of the number of
  draws below the quantile (Hahn & Meeker 1991)

For quasi-random draws these errors are conservative: their convergence is
faster than that of random draws, which the errors assume. The statistics are
computed by sbmlsim-core (`sbmlsim_core.stats`).

Hahn GJ, Meeker WQ (1991). Statistical Intervals: A Guide for Practitioners.
Wiley.
"""

from __future__ import annotations

from numpy.typing import ArrayLike

from sbmlsim.engine import import_core


def mean_se(x: ArrayLike) -> tuple[float, float, int]:
    """Get the mean, its standard error and the number of draws.

    Args:
        x: the draws.

    Returns:
        The mean, its standard error and the number of draws which are not
        `NaN`; `nan` without draws (the error without two of them).
    """
    import_core()
    from sbmlsim_core import stats

    return stats.mean_se(x)


def sd_se(x: ArrayLike) -> tuple[float, float, int]:
    """Get the standard deviation, its standard error and the number of draws.

    Args:
        x: the draws.

    Returns:
        The standard deviation, its standard error by the delta method and
        the number of draws which are not `NaN`; `nan` without two draws.
    """
    import_core()
    from sbmlsim_core import stats

    return stats.sd_se(x)


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
        number of draws which are not `NaN`; an end beyond the draws is `nan`,
        e.g. the upper end of the 99 % quantile of 10 draws.
    """
    import_core()
    from sbmlsim_core import stats

    return stats.quantile_ci(x, q, level)


def first_order_given_data(
    x: ArrayLike, y: ArrayLike, n_bins: int | None = None
) -> float:
    """Get the first-order index of an input from given draws, by binning.

    The draws are split into `n_bins` bins of equal probability of the input
    (`max(2, sqrt(n) / 2)` by default), and the index is the variance of the
    means of the output in the bins over the variance of the output,
    `Var(E[y | bin]) / Var(y)` (Plischke, Borgonovo & Smith 2013), less the
    variance the bin means have by chance, `(n_bins - 1) / n` of the variance
    of the output, so an input without an effect has an index near 0. It
    needs no simulations beyond the draws of the propagation. For inputs
    which are correlated, the index of an input includes the effect of the
    inputs it is correlated with; the indices do not add up to the variance.

    Plischke E, Borgonovo E, Smith CL (2013). Global sensitivity measures from
    given data. Eur J Oper Res 226:536-550.

    Args:
        x: the draws of the input.
        y: the draws of the output, of the same draws.
        n_bins: the number of bins.

    Returns:
        The index, between 0 and 1; `nan` with fewer than 8 finite pairs or
        an output without variance.
    """
    import_core()
    from sbmlsim_core import stats

    return stats.first_order_given_data(x, y, n_bins)
