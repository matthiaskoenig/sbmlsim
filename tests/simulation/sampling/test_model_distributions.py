"""The distributions of the parameters of a model, from the formulas of distrib."""

import math

import numpy as np
import pytest

from sbmlsim.simulation.sampling import (
    LogNormal,
    Normal,
    Truncated,
    Uniform,
    distribution_from_formula,
)


@pytest.mark.parametrize(
    ("formula", "kind"),
    [
        ("lognormal(0.0, 0.5)", LogNormal),
        ("normal(1, 0.2)", Normal),
        ("uniform(1, 2)", Uniform),
        ("lognormal(0.0, 0.5, 0.1, 3)", Truncated),
    ],
)
def test_the_distributions_of_distrib(formula: str, kind: type) -> None:
    assert isinstance(distribution_from_formula(formula), kind)


@pytest.mark.parametrize("formula", ["gamma(1, 2)", "normal(1)", "normal(a, 1)", "x"])
def test_an_unsupported_distribution_raises(formula: str) -> None:
    with pytest.raises(ValueError, match="distribution"):
        distribution_from_formula(formula)


def test_the_lognormal_of_distrib_has_its_median_and_sigma() -> None:
    d = distribution_from_formula("lognormal(1.0, 0.5)")
    median, upper = d.ppf(np.array([0.5, 0.8413447460685429]))
    assert median == pytest.approx(math.e)
    assert upper == pytest.approx(math.exp(1.5))


def test_a_truncated_normal_keeps_its_bounds() -> None:
    d = distribution_from_formula("normal(1.0, 0.2, 0.8, 1.2)")
    assert isinstance(d, Truncated)
    assert isinstance(d.distribution, Normal)
    draws = np.asarray(d.ppf(np.linspace(0.01, 0.99, 20)), dtype=float)
    assert np.all((draws >= 0.8) & (draws <= 1.2))
