"""Uncertain values: a nominal value with its distribution."""

import numpy as np
import pytest
from scipy import stats

from sbmlsim import Q
from sbmlsim.simulation.sampling import LogNormal, Normal, Truncated, Uniform
from sbmlsim.simulation.sampling.distributions import Distribution
from sbmlsim.uncertainty import Uncertain
from sbmlsim.units import Quantity


def _magnitude(value: object, unit: str) -> float:
    assert isinstance(value, Quantity)
    return float(value.to(unit).magnitude)


def _at(
    distribution: Distribution, u: list[float], unit: str | None = None
) -> np.ndarray:
    """Get the values of a distribution at probabilities, in a unit."""
    x = distribution.ppf(np.array(u))
    if isinstance(x, Quantity):
        return np.asarray(x.to(unit).magnitude if unit else x.magnitude)
    return np.asarray(x)


def test_mean_and_sd_of_a_positive_value_are_log_normal() -> None:
    dose = Uncertain(Q(100, "mg"), sd=Q(20, "mg"))
    assert dose.shape == "log-normal"
    assert isinstance(dose.distribution, LogNormal)
    assert _magnitude(dose.mean(), "mg") == pytest.approx(100.0, rel=1e-9)
    assert _magnitude(dose.sd(), "mg") == pytest.approx(20.0, rel=1e-9)
    median = float(_at(dose.distribution, [0.5], "mg")[0])
    assert median == pytest.approx(100.0 / np.sqrt(1.04), rel=1e-9)
    assert dose.nominal == Q(100, "mg")


def test_a_normal_opt_in() -> None:
    dose = Uncertain(Q(100, "mg"), sd=Q(20, "mg"), shape="normal")
    assert dose.shape == "normal"
    assert isinstance(dose.distribution, Normal)
    assert _magnitude(dose.sd(), "mg") == pytest.approx(20.0)


def test_a_value_which_may_be_negative_is_normal() -> None:
    shift = Uncertain(-0.5, sd=0.1)
    assert shift.shape == "normal"
    assert shift.mean() == pytest.approx(-0.5)


def test_cv() -> None:
    clearance = Uncertain(Q(10, "l/hr"), cv=0.3)
    assert _magnitude(clearance.sd(), "l/hr") == pytest.approx(3.0, rel=1e-9)


def test_truncation() -> None:
    weight = Uncertain(Q(75, "kg"), sd=Q(12, "kg"), low=Q(45, "kg"))
    assert isinstance(weight.distribution, Truncated)
    lowest = float(_at(weight.distribution, [1e-9], "kg")[0])
    assert lowest >= 45.0
    # the truncation moves the mean up
    assert _magnitude(weight.mean(), "kg") > 75.0


def test_range_and_log_range() -> None:
    fraction = Uncertain.range(0.6, 0.9)
    assert isinstance(fraction.distribution, Uniform)
    assert fraction.nominal == pytest.approx(0.75)
    rate = Uncertain.range(Q(0.1, "1/hr"), Q(10, "1/hr"), log=True)
    assert rate.shape == "log-uniform"
    assert _magnitude(rate.nominal, "1/hr") == pytest.approx(1.0)


def test_interval_matches_its_quantiles() -> None:
    clint = Uncertain.interval(Q(12, "l/hr"), Q(30, "l/hr"), level=0.90)
    np.testing.assert_allclose(
        _at(clint.distribution, [0.05, 0.5, 0.95], "l/hr"),
        [12.0, np.sqrt(12.0 * 30.0), 30.0],
        rtol=1e-9,
    )
    shift = Uncertain.interval(-1.0, 3.0, level=0.95)
    assert shift.shape == "normal"
    assert float(_at(shift.distribution, [0.975])[0]) == pytest.approx(3.0)


def test_se_with_a_count() -> None:
    auc = Uncertain.se(Q(40, "mg*hr/l"), Q(2, "mg*hr/l"), n=16)
    assert _magnitude(auc.sd(), "mg*hr/l") == pytest.approx(8.0, rel=1e-9)


def test_exactly_one_spread() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Uncertain(Q(1, "mg"))
    with pytest.raises(ValueError, match="exactly one"):
        Uncertain(Q(1, "mg"), sd=Q(0.1, "mg"), cv=0.1)


def test_units_convert() -> None:
    dose = Uncertain(Q(100, "mg"), sd=Q(0.02, "g"), shape="normal")
    assert _magnitude(dose.sd(), "mg") == pytest.approx(20.0)
    with pytest.raises(ValueError, match="unit"):
        Uncertain(Q(100, "mg"), sd=Q(2, "l"))


def test_a_given_distribution() -> None:
    ka = Uncertain(
        Q(1.2, "1/hr"), distribution=LogNormal(median=Q(1.2, "1/hr"), cv=0.5)
    )
    assert ka.shape == "given"
    sigma = np.sqrt(np.log(1.25))
    expected = stats.lognorm(s=sigma, scale=1.2).mean()
    assert _magnitude(ka.mean(), "1/hr") == pytest.approx(expected, rel=1e-6)


def test_the_kind_and_the_name() -> None:
    weight = Uncertain(Q(75, "kg"), cv=0.15, kind="variability", name="BW")
    assert (weight.kind, weight.name) == ("variability", "BW")
    with pytest.raises(ValueError, match="kind"):
        Uncertain(Q(75, "kg"), cv=0.15, kind="noise")  # ty: ignore[invalid-argument-type]


def test_a_round_trip_of_the_dict() -> None:
    for value in (
        Uncertain(Q(100, "mg"), sd=Q(20, "mg"), low=Q(10, "mg"), name="dose"),
        Uncertain.range(0.6, 0.9),
        Uncertain.interval(Q(12, "l/hr"), Q(30, "l/hr"), level=0.9, kind="variability"),
    ):
        again = Uncertain.from_dict(value.to_dict())
        assert again.to_dict() == value.to_dict()
        np.testing.assert_allclose(
            _at(again.distribution, [0.3]), _at(value.distribution, [0.3])
        )
