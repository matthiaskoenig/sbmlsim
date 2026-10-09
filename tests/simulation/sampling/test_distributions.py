"""The distributions of the sampler."""

import json
from collections.abc import Callable

import numpy as np
import pytest
from scipy import stats

from sbmlsim import Q
from sbmlsim.simulation.sampling import (
    Distribution,
    Empirical,
    Fixed,
    LogNormal,
    LogUniform,
    Normal,
    Truncated,
    Uniform,
)
from sbmlsim.units import Quantity


def plain(
    distribution: Distribution, u: np.ndarray, reference: float | None = None
) -> np.ndarray:
    values = distribution.ppf(u, reference)
    assert isinstance(values, np.ndarray)
    return values


def quantity(
    distribution: Distribution, u: np.ndarray, reference: Quantity | None = None
) -> Quantity:
    values = distribution.ppf(u, reference)
    assert isinstance(values, Quantity)
    return values


U = np.array([0.0, 0.1, 0.5, 0.9, 1.0])


def test_uniform_and_relative_uniform() -> None:
    np.testing.assert_allclose(plain(Uniform(2.0, 4.0), U), 2.0 + 2.0 * U)
    np.testing.assert_allclose(plain(Uniform(relative=0.5), U, 10.0), 5.0 + 10.0 * U)
    assert Uniform(relative=0.5).is_relative and not Uniform(2.0, 4.0).is_relative


def test_degenerate_bounds_are_allowed() -> None:
    np.testing.assert_array_equal(plain(Uniform(3.0, 3.0), U), np.full(5, 3.0))
    np.testing.assert_allclose(plain(LogUniform(3.0, 3.0), U), np.full(5, 3.0))
    np.testing.assert_array_equal(
        Uniform(3.0, 3.0).cdf(np.array([2.0, 3.0])), [0.0, 1.0]
    )
    np.testing.assert_array_equal(
        LogUniform(3.0, 3.0).cdf(np.array([2.0, 3.0])), [0.0, 1.0]
    )


def test_the_formulas_of_the_start_values_of_a_fit() -> None:
    lower, upper = 1e-3, 5.0
    expected = 10.0 ** (np.log10(lower) + U * (np.log10(upper) - np.log10(lower)))
    np.testing.assert_array_equal(plain(LogUniform(lower, upper), U), expected)
    np.testing.assert_array_equal(
        plain(Uniform(lower, upper), U), lower + U * (upper - lower)
    )


def test_cdf_inverts_ppf() -> None:
    u = U[1:-1]
    for distribution in [
        Uniform(2.0, 4.0),
        LogUniform(1e-2, 1e2),
        Normal(3.0, 2.0),
        LogNormal(5.0, 0.2),
        Truncated(Normal(0.0, 1.0), lower=0.0),
        Truncated(LogNormal(5.0, 0.2), lower=3.0, upper=9.0),
    ]:
        np.testing.assert_allclose(distribution.cdf(plain(distribution, u)), u)
    np.testing.assert_allclose(
        Uniform(relative=0.5).cdf(plain(Uniform(relative=0.5), u, 4.0), 4.0), u
    )
    np.testing.assert_allclose(
        LogUniform(factor=3.0).cdf(plain(LogUniform(factor=3.0), u, 4.0), 4.0), u
    )


def test_log_uniform_is_uniform_in_log10() -> None:
    values = plain(LogUniform(1e-2, 1e2), U)
    np.testing.assert_allclose(np.log10(values), -2.0 + 4.0 * U)
    np.testing.assert_allclose(
        plain(LogUniform(factor=10.0), U, 5.0), 10 ** (np.log10(0.5) + 2.0 * U)
    )


def test_normal_and_lognormal_against_scipy() -> None:
    u = U[1:-1]
    np.testing.assert_allclose(plain(Normal(3.0, 2.0), u), stats.norm(3.0, 2.0).ppf(u))
    np.testing.assert_allclose(
        plain(Normal(cv=0.1), u, 10.0), stats.norm(10.0, 1.0).ppf(u)
    )
    sigma = np.sqrt(np.log(1.0 + 0.2**2))
    np.testing.assert_allclose(
        plain(LogNormal(5.0, 0.2), u), stats.lognorm(sigma, scale=5.0).ppf(u)
    )
    np.testing.assert_allclose(
        plain(LogNormal(cv=0.2), u, 5.0), stats.lognorm(sigma, scale=5.0).ppf(u)
    )


def test_the_probabilities_zero_and_one_are_clipped_for_unbounded_distributions() -> (
    None
):
    values = plain(Normal(0.0, 1.0), np.array([0.0, 1.0]))
    assert np.isfinite(values).all()
    assert values[0] < -6.0 and values[1] > 6.0


def test_truncated_restricts_the_probabilities() -> None:
    truncated = Truncated(Normal(0.0, 1.0), lower=0.0)
    values = plain(truncated, np.linspace(0.0, 1.0, 11))
    assert values.min() >= 0.0
    np.testing.assert_allclose(
        plain(truncated, np.array([0.5])), stats.halfnorm.ppf(0.5)
    )


def test_empirical_and_fixed() -> None:
    empirical = Empirical([3.0, 1.0, 2.0])
    np.testing.assert_array_equal(
        plain(empirical, np.array([0.0, 0.4, 0.99, 1.0])), [1.0, 2.0, 3.0, 3.0]
    )
    np.testing.assert_array_equal(
        empirical.cdf(np.array([0.5, 1.0, 2.5, 3.0])), [0.0, 1 / 3, 2 / 3, 1.0]
    )
    np.testing.assert_array_equal(plain(Fixed(7.0), U), np.full(5, 7.0))
    np.testing.assert_array_equal(Fixed(7.0).cdf(np.array([6.0, 7.0])), [0.0, 1.0])


def test_quantities_keep_their_unit() -> None:
    values = quantity(Normal(Q(75.0, "kg"), Q(12.0, "kg")), np.array([0.5]))
    assert values.units == Q(1.0, "kg").units
    assert values.magnitude[0] == pytest.approx(75.0)
    mixed = Truncated(Normal(Q(75.0, "kg"), Q(12.0, "kg")), lower=40.0)
    assert quantity(mixed, np.array([0.0])).magnitude[0] >= 40.0
    grams = quantity(Uniform(Q(1.0, "kg"), Q(2000.0, "g")), np.array([1.0]))
    assert grams.to("kg").magnitude[0] == pytest.approx(2.0)
    relative = quantity(LogNormal(cv=0.1), np.array([0.5]), Q(5.0, "mg"))
    assert str(relative.units) == "milligram"
    empirical = quantity(Empirical(Q(np.array([3.0, 1.0]), "mg")), np.array([0.0]))
    assert str(empirical.units) == "milligram" and empirical.magnitude[0] == 1.0
    assert str(quantity(Fixed(Q(3.0, "mg")), U).units) == "milligram"


def test_a_relative_distribution_needs_a_reference() -> None:
    with pytest.raises(ValueError, match="model="):
        plain(LogNormal(cv=0.1), U)


def test_a_log_distribution_needs_a_positive_reference() -> None:
    with pytest.raises(ValueError, match="positive"):
        plain(LogNormal(cv=0.1), U, -1.0)
    with pytest.raises(ValueError, match="positive"):
        plain(LogUniform(factor=2.0), U, 0.0)


def test_a_negative_reference() -> None:
    values = plain(Uniform(relative=0.5), np.array([0.0, 1.0]), -10.0)
    np.testing.assert_allclose(values, [-15.0, -5.0])
    assert plain(Normal(cv=0.1), np.array([0.5]), -10.0)[0] == pytest.approx(-10.0)


@pytest.mark.parametrize(
    "make",
    [
        lambda: Uniform(4.0, 2.0),
        lambda: Uniform(2.0),
        lambda: Uniform(2.0, 4.0, relative=0.1),
        lambda: LogUniform(0.0, 1.0),
        lambda: LogUniform(factor=1.0),
        lambda: Normal(1.0),
        lambda: Normal(sd=1.0, cv=0.1),
        lambda: Normal(1.0, -1.0),
        lambda: LogNormal(cv=-0.1),
        lambda: LogNormal(5.0),
        lambda: Empirical([]),
        lambda: Truncated(Normal(0.0, 1.0)),
        lambda: Truncated(Normal(0.0, 1.0), lower=2.0, upper=1.0),
        lambda: Truncated(Normal(0.0, 1.0), lower=20.0),
    ],
)
def test_invalid_distributions_raise(make: Callable[[], Distribution]) -> None:
    with pytest.raises(ValueError):
        make()


def test_the_distributions_serialize() -> None:
    distributions = [
        Uniform(1.0, 2.0),
        Uniform(relative=0.1),
        LogUniform(factor=3.0),
        Normal(Q(75.0, "kg"), Q(12.0, "kg")),
        LogNormal(cv=0.2),
        Truncated(Normal(0.0, 1.0), lower=0.0),
        Empirical([1.0, 2.0]),
        Fixed(Q(3.0, "mg")),
    ]
    data = json.loads(json.dumps([d.to_dict() for d in distributions]))
    assert data[0] == {"type": "Uniform", "lower": 1.0, "upper": 2.0, "relative": None}
    assert data[3]["mean"] == {"value": 75.0, "unit": "kilogram"}
    assert data[5]["distribution"]["type"] == "Normal"
