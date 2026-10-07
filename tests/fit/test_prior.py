"""The prior of a parameter, the priors of PEtab v2 truncated at the bounds."""

import json
import math
import re

import pytest
import scipy.stats

from sbmlsim.fit.objects import FitParameter, Prior, PriorDistribution

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
