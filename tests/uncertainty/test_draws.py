"""Correlated quasi-random draws of uncertain values."""

import numpy as np
import pytest
from scipy import stats

from sbmlsim import Q
from sbmlsim.simulation.sampling import LogNormal, Normal, Uniform
from sbmlsim.uncertainty import Covariance, Variable
from sbmlsim.uncertainty.draws import draw, normal_score_correlation
from sbmlsim.units import Quantity


def _variable(key: str, distribution: object) -> Variable:
    return Variable(key, (key,), distribution, "uncertainty", "parameter")  # ty: ignore[invalid-argument-type]


def _magnitude(values: object, unit: str | None = None) -> np.ndarray:
    if isinstance(values, Quantity):
        return np.asarray(values.to(unit).magnitude if unit else values.magnitude)
    return np.asarray(values)


def test_the_moments_of_the_draws() -> None:
    variables = [
        _variable("a", Normal(mean=Q(5.0, "mg"), sd=Q(2.0, "mg"))),
        _variable("b", LogNormal(median=10.0, cv=0.5)),
        _variable("c", Uniform(lower=0.0, upper=1.0)),
    ]
    values = draw(variables, [], 4096, seed=1)
    a = _magnitude(values["a"], "mg")
    assert a.mean() == pytest.approx(5.0, rel=0.01)
    assert a.std(ddof=1) == pytest.approx(2.0, rel=0.01)
    b = _magnitude(values["b"])
    assert np.median(b) == pytest.approx(10.0, rel=0.01)
    assert b.std(ddof=1) / b.mean() == pytest.approx(0.5, rel=0.02)
    assert _magnitude(values["c"]).mean() == pytest.approx(0.5, abs=0.01)


def test_a_correlation_of_normal_scores() -> None:
    variables = [
        _variable("a", Normal(mean=0.0, sd=1.0)),
        _variable("b", LogNormal(median=1.0, cv=0.3)),
    ]
    corr = Covariance.correlation({("a", "b"): 0.7})
    values = draw(variables, [corr], 4096, seed=2)
    scores = np.vstack([_magnitude(values["a"]), np.log(_magnitude(values["b"]))])
    assert np.corrcoef(scores)[0, 1] == pytest.approx(0.7, abs=0.02)


def test_a_linear_covariance_of_log_normals() -> None:
    """A covariance on the linear scale is the covariance of the values."""
    means, cvs = np.array([10.0, 50.0]), np.array([0.3, 0.2])
    sds = means * cvs
    rho = 0.6
    matrix = np.array(
        [[sds[0] ** 2, rho * sds[0] * sds[1]], [rho * sds[0] * sds[1], sds[1] ** 2]]
    )
    variables = [
        _variable(key, LogNormal(median=float(m / np.sqrt(1 + cv**2)), cv=float(cv)))
        for key, m, cv in zip(["CL", "V"], means, cvs, strict=True)
    ]
    values = draw(variables, [Covariance(["CL", "V"], matrix=matrix)], 2**14, seed=3)
    sample = np.cov(np.vstack([_magnitude(values["CL"]), _magnitude(values["V"])]))
    np.testing.assert_allclose(sample, matrix, rtol=0.03)


def test_a_covariance_whose_sds_disagree_raises() -> None:
    variables = [
        _variable("a", Normal(mean=0.0, sd=1.0)),
        _variable("b", Normal(mean=0.0, sd=2.0)),
    ]
    cov = Covariance(["a", "b"], matrix=[[1.0, 0.5], [0.5, 1.0]])
    with pytest.raises(ValueError, match=r"'b'.*standard deviation"):
        normal_score_correlation(variables, [cov])


def test_a_covariance_of_an_unknown_target_raises() -> None:
    variables = [_variable("a", Normal(mean=0.0, sd=1.0))]
    with pytest.raises(ValueError, match="'z'"):
        normal_score_correlation(variables, [Covariance.correlation({("a", "z"): 0.5})])


def test_batches_continue_the_sequence() -> None:
    variables = [
        _variable("a", Normal(mean=0.0, sd=1.0)),
        _variable("b", Normal(mean=1.0, sd=1.0)),
    ]
    corr = [Covariance.correlation({("a", "b"): 0.4})]
    whole = draw(variables, corr, 512, seed=4)
    first = draw(variables, corr, 256, seed=4, start=0)
    second = draw(variables, corr, 256, seed=4, start=256)
    for key in ("a", "b"):
        np.testing.assert_array_equal(
            np.concatenate([_magnitude(first[key]), _magnitude(second[key])]),
            _magnitude(whole[key]),
        )


@pytest.mark.parametrize("design", ["lhs", "random"])
def test_other_designs(design: str) -> None:
    variables = [_variable("a", Normal(mean=0.0, sd=1.0))]
    values = _magnitude(draw(variables, [], 2048, seed=5, design=design)["a"])  # ty: ignore[invalid-argument-type]
    assert stats.kstest(values, "norm").pvalue > 0.01
    again = _magnitude(draw(variables, [], 2048, seed=5, design=design)["a"])  # ty: ignore[invalid-argument-type]
    np.testing.assert_array_equal(values, again)
