"""The statistics of the draws of an uncertainty analysis."""

import numpy as np
import pytest

from sbmlsim.uncertainty.statistics import (
    first_order_given_data,
    mean_se,
    quantile_ci,
    sd_se,
)


def test_the_standard_errors_of_the_mean_and_the_sd() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(3.0, 2.0, 10_000)
    _mean, se, n = mean_se(x)
    assert n == 10_000
    assert se == pytest.approx(2.0 / 100.0, rel=0.05)
    sd, sd_error, _ = sd_se(x)
    assert sd == pytest.approx(2.0, rel=0.03)
    # for a normal sample the SE of the SD is sd / sqrt(2 (n - 1))
    assert sd_error == pytest.approx(2.0 / np.sqrt(2 * 9_999), rel=0.1)


def test_nan_draws_are_left_out() -> None:
    x = np.array([1.0, 2.0, np.nan, 3.0])
    assert mean_se(x)[0] == pytest.approx(2.0)
    assert mean_se(x)[2] == 3
    assert np.isnan(mean_se(np.array([np.nan]))[0])


def test_the_interval_of_a_quantile_covers_it() -> None:
    """The distribution free interval covers the quantile in about 95 % of samples."""
    rng = np.random.default_rng(2)
    true = 1.6448536269514722  # the 95 % quantile of the standard normal
    covered = 0
    repeats = 2000
    for _ in range(repeats):
        _estimate, low, high, _n = quantile_ci(rng.normal(size=200), 0.95)
        covered += int(low <= true <= high)
    assert 0.93 <= covered / repeats <= 0.975


def test_a_quantile_of_few_draws_has_an_open_interval() -> None:
    estimate, _low, high, n = quantile_ci(np.arange(10.0), 0.99)
    assert n == 10
    assert np.isfinite(estimate)
    assert np.isnan(high)


def test_first_order_indices_of_given_data() -> None:
    rng = np.random.default_rng(3)
    x1, x2 = rng.normal(size=(2, 4096))
    y = x1 + 2 * x2
    assert first_order_given_data(x1, y) == pytest.approx(0.2, abs=0.03)
    assert first_order_given_data(x2, y) == pytest.approx(0.8, abs=0.03)
    # an input without an effect has an index near 0
    assert first_order_given_data(rng.normal(size=4096), y) == pytest.approx(
        0.0, abs=0.02
    )
