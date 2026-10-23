"""The prior of a parameter, the priors of PEtab v2 truncated at the bounds."""

import json
import math
import re

import numpy as np
import pytest
import scipy.stats

from sbmlsim import Q
from sbmlsim.fit.objects import FitParameter, Prior, PriorDistribution
from sbmlsim.simulation.sampling import (
    Fixed,
    LogNormal,
    LogUniform,
    Normal,
    Truncated,
    Uniform,
)
from sbmlsim.simulation.sampling.distributions import Distribution
from sbmlsim.uncertainty import Uncertain

#: the `log_prior` of the case 0024 of the PEtab v2 test suite at `5.0`, the
#: parameters have the bounds `[0, 10]`
CASE_0024 = {
    ("uniform", (2.0, 8.0)): -1.79175946922805,
    ("normal", (4.0, 2.0)): -1.71269052620973,
    ("log-normal", (5.0, 2.0)): -2.23619159741645,
    ("cauchy", (3.0, 5.0)): -2.15728237835358,
    ("chisquare", (4.0,)): -2.23558885282238,
    ("exponential", (3.0,)): -2.72895309496787,
    ("gamma", (3.0, 5.0)): -2.17348344324312,
    ("laplace", (3.0, 5.0)): -2.19557833470717,
    ("log-laplace", (3.0, 5.0)): -3.35750526098019,
    ("log-uniform", (3.0, 5.0)): -0.93771092034198,
    ("rayleigh", (3.0,)): -1.9728021416671,
}


@pytest.mark.parametrize(("prior", "expected"), CASE_0024.items())
def test_log_density_of_the_test_suite(
    prior: tuple[str, tuple[float, ...]], expected: float
) -> None:
    """The priors of PEtab v2 are truncated at the bounds of the parameter."""
    distribution, parameters = prior
    density = Prior(PriorDistribution(distribution), parameters).log_density(
        5.0, 0.0, 10.0
    )
    assert density == pytest.approx(expected, abs=1e-13)


def test_a_truncated_normal_prior() -> None:
    """The density is normalized over the bounds."""
    prior = Prior(PriorDistribution.NORMAL, (4.0, 2.0))
    expected = scipy.stats.truncnorm.logpdf(
        5.0, (0.0 - 4.0) / 2.0, (10.0 - 4.0) / 2.0, loc=4.0, scale=2.0
    )
    assert prior.log_density(5.0, 0.0, 10.0) == pytest.approx(expected, rel=1e-12)
    # without bounds it is the normal distribution
    expected = scipy.stats.norm.logpdf(5.0, loc=4.0, scale=2.0)
    assert prior.log_density(5.0, -math.inf, math.inf) == pytest.approx(expected)


def test_a_prior_coerces_its_fields() -> None:
    """The distribution is the enum and the parameters a tuple of floats."""
    prior = Prior("normal", [4, 2])  # ty: ignore[invalid-argument-type]
    assert prior.distribution is PriorDistribution.NORMAL
    assert prior.parameters == (4.0, 2.0)
    assert prior == Prior(PriorDistribution.NORMAL, (4.0, 2.0))


def test_a_prior_requires_a_distribution_of_petab() -> None:
    """The error names the distribution and the ones of PEtab."""
    with pytest.raises(ValueError, match=re.escape("'beta'")):
        Prior("beta", (1.0, 2.0))  # ty: ignore[invalid-argument-type]


def test_a_fit_parameter_with_a_prior() -> None:
    """The prior is part of the parameter and of its serialization."""
    prior = Prior(PriorDistribution.NORMAL, (4.0, 2.0))
    parameter = FitParameter(
        pid="k", lower_bound=0.0, upper_bound=10.0, unit="dimensionless", prior=prior
    )
    assert parameter.prior == prior
    assert parameter.to_dict()["prior"] == {
        "distribution": "normal",
        "parameters": [4.0, 2.0],
    }
    restored = FitParameter.from_json(json.dumps(parameter.to_dict()))
    assert restored == parameter
    assert restored.prior == prior
    assert parameter != FitParameter(
        pid="k", lower_bound=0.0, upper_bound=10.0, unit="dimensionless"
    )


@pytest.mark.parametrize(
    "distribution",
    [
        Normal(mean=Q(4.0, "1/hr"), sd=Q(0.5, "1/hr")),
        LogNormal(median=Q(1.2, "1/hr"), cv=0.4),
        Uniform(lower=Q(0.5, "1/hr"), upper=Q(8.0, "1/hr")),
        LogUniform(lower=Q(0.1, "1/hr"), upper=Q(10.0, "1/hr")),
    ],
)
def test_a_prior_of_a_distribution(distribution: Distribution) -> None:
    """The prior is the distribution: the density agrees, the round trip holds."""
    prior = Prior.from_distribution(distribution, "1/min")
    values = np.array([0.6, 1.0, 2.0, 4.0, 7.0]) / 60.0
    densities = np.exp([prior.log_density(float(v), 1e-6, 100.0) for v in values])
    # the density of the distribution in 1/min, from its cumulative distribution
    h = 1e-7
    upper = distribution.cdf(Q(values + h, "1/min"))
    lower = distribution.cdf(Q(values - h, "1/min"))
    # a central difference of the distribution function, accurate to about 1e-3
    np.testing.assert_allclose(densities, (upper - lower) / (2 * h), rtol=1e-3)
    again = prior.to_distribution("1/min")
    np.testing.assert_allclose(
        again.cdf(Q(values, "1/min")), distribution.cdf(Q(values, "1/min")), rtol=1e-9
    )


def test_a_fit_parameter_with_an_uncertain_prior() -> None:
    parameter = FitParameter(
        pid="ka",
        lower_bound=0.01,
        upper_bound=100.0,
        unit="1/hr",
        prior=Uncertain(Q(1.2, "1/hr"), cv=0.3),
    )
    assert parameter.prior is not None
    assert parameter.prior.distribution is PriorDistribution.LOG_NORMAL
    sigma = math.sqrt(math.log1p(0.09))
    mu = math.log(1.2 / math.sqrt(1.09))
    assert parameter.prior.parameters == pytest.approx((mu, sigma))


def test_a_prior_which_is_no_prior_of_petab_raises() -> None:
    with pytest.raises(ValueError, match="Normal, LogNormal, Uniform, LogUniform"):
        Prior.from_distribution(Fixed(value=1.0), None)
    with pytest.raises(ValueError, match="relative"):
        Prior.from_distribution(Normal(cv=0.2), None)


def test_a_truncation_is_the_bounds_of_the_parameter() -> None:
    """A truncation inside the bounds of the parameter would be lost."""
    inner = Normal(mean=4.0, sd=1.0)
    truncated = Truncated(inner, lower=0.0, upper=10.0)
    assert Prior.from_distribution(truncated, None) == Prior.from_distribution(
        inner, None
    )
    # a truncation at the bounds of the parameter is the prior of PEtab
    parameter = FitParameter(
        pid="k",
        lower_bound=0.0,
        upper_bound=10.0,
        unit="dimensionless",
        prior=truncated,
    )
    assert parameter.prior == Prior.from_distribution(inner, None)
    with pytest.raises(ValueError, match="bounds"):
        FitParameter(
            pid="k",
            lower_bound=0.0,
            upper_bound=10.0,
            unit="dimensionless",
            prior=Truncated(inner, lower=2.0, upper=6.0),
        )


def test_a_prior_in_a_unit_which_does_not_fit_raises() -> None:
    inner = Normal(mean=Q(4.0, "mg"), sd=Q(1.0, "mg"))
    with pytest.raises(ValueError, match="does not fit"):
        FitParameter(
            pid="k",
            lower_bound=0.0,
            upper_bound=10.0,
            unit="mmol",
            prior=Truncated(inner, lower=Q(0.0, "mg"), upper=Q(10.0, "mg")),
        )


def test_a_prior_of_quantities_for_a_unit_id_of_a_model_raises() -> None:
    """A unit id of a model is resolved later, a prior of numbers is in it."""
    with pytest.raises(ValueError, match=r"no unit of pint.*numbers"):
        FitParameter(
            pid="k",
            lower_bound=0.0,
            upper_bound=10.0,
            unit="mg_per_min",
            prior=Normal(mean=Q(4.0, "mg/min"), sd=Q(1.0, "mg/min")),
        )
    parameter = FitParameter(
        pid="k",
        lower_bound=0.0,
        upper_bound=10.0,
        unit="mg_per_min",
        prior=Normal(mean=4.0, sd=1.0),
    )
    assert parameter.prior == Prior.from_distribution(Normal(mean=4.0, sd=1.0), None)
