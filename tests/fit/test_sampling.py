"""Tests of the sampling of the start values of a fit."""

import logging

import numpy as np
import pytest
from scipy.stats import qmc

from sbmlsim.fit import FitParameter
from sbmlsim.fit.sampling import SamplingType, create_samples

SIZE = 7
SEED = 1234


def _bounded() -> list[FitParameter]:
    return [
        FitParameter("p1", 100.0, lower_bound=10.0, upper_bound=1e4, unit="mM"),
        FitParameter("p2", 0.0, lower_bound=-2.0, upper_bound=3.0, unit="mM"),
        FitParameter("p3", None, lower_bound=1e-3, upper_bound=1.0, unit="mM"),
    ]


def _unbounded() -> list[FitParameter]:
    return [
        FitParameter("w1", -0.5, unit="dimensionless"),
        FitParameter("w2", 0.0, lower_bound=-np.inf, upper_bound=1.0, unit="mM"),
        FitParameter("w3", 2.5, lower_bound=0.0, upper_bound=np.inf, unit="mM"),
    ]


def _expected(parameters: list[FitParameter], sampling: SamplingType) -> np.ndarray:
    """Get the samples as they were drawn before parameters could be unbounded.

    This is the sampling of `create_samples` of the release 0.7.2 for bounded
    parameters, written down once more: the unit hypercube of all parameters
    is drawn, and every column is stretched to the bounds of its parameter.
    """
    rng = np.random.default_rng(SEED)
    if sampling.is_lhs:
        x = qmc.LatinHypercube(d=len(parameters), rng=rng).random(n=SIZE)
    else:
        x = rng.random(size=(SIZE, len(parameters)))
    for k, p in enumerate(parameters):
        lb, ub = float(p.lower_bound), float(p.upper_bound)
        if sampling.is_log:
            lb = 1e-10 if lb <= 0.0 else lb
            x[:, k] = np.power(10, np.log10(lb) + x[:, k] * np.log10(ub / lb))
        else:
            x[:, k] = lb + x[:, k] * (ub - lb)
    return x


@pytest.mark.parametrize("sampling", list(SamplingType))
def test_the_samples_of_bounded_parameters_did_not_change(
    sampling: SamplingType,
) -> None:
    """The samples of a fit with bounds are the ones of the release before."""
    parameters = _bounded()
    samples = create_samples(parameters, size=SIZE, sampling=sampling, seed=SEED)
    assert list(samples.columns) == ["p1", "p2", "p3"]
    np.testing.assert_allclose(
        samples.to_numpy(), _expected(parameters, sampling), rtol=1e-14
    )
    for p in parameters:
        lower = 1e-10 if sampling.is_log and p.lower_bound <= 0 else p.lower_bound
        assert np.all(samples[p.pid] >= lower)
        assert np.all(samples[p.pid] <= p.upper_bound)


@pytest.mark.parametrize("sampling", list(SamplingType))
def test_a_parameter_without_a_bound_is_not_sampled(
    sampling: SamplingType, caplog: pytest.LogCaptureFixture
) -> None:
    """A parameter with an infinite bound starts from its start value."""
    parameters = [*_bounded(), *_unbounded()]
    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.sampling"):
        samples = create_samples(parameters, size=SIZE, sampling=sampling, seed=SEED)
    for p in _unbounded():
        np.testing.assert_array_equal(samples[p.pid], np.full(SIZE, p.start_value))
    # the bound is not replaced by a number which is sampled
    assert "infinite" not in caplog.text
    assert np.all(np.isfinite(samples.to_numpy()))
    # the repeats differ in the parameters which have bounds
    assert samples["p1"].nunique() == SIZE


@pytest.mark.parametrize("sampling", list(SamplingType))
def test_the_samples_do_not_depend_on_the_parameters_without_bounds(
    sampling: SamplingType,
) -> None:
    """The columns of the bounded parameters are the ones of the hypercube."""
    bounded = _bounded()
    w1, w2, w3 = _unbounded()
    parameters = [w1, bounded[0], w2, bounded[1], bounded[2], w3]
    samples = create_samples(parameters, size=SIZE, sampling=sampling, seed=SEED)
    expected = _expected(
        [
            FitParameter(p.pid, None, 1.0, 2.0, unit="mM")
            if np.isinf(p.lower_bound) or np.isinf(p.upper_bound)
            else p
            for p in parameters
        ],
        sampling,
    )
    for k in (1, 3, 4):
        np.testing.assert_allclose(samples.iloc[:, k], expected[:, k], rtol=1e-14)


def test_a_parameter_without_a_bound_needs_a_start_value() -> None:
    """There is nothing to start from without a bound and a start value."""
    parameters = [*_bounded(), FitParameter("w1", None, unit="dimensionless")]
    with pytest.raises(ValueError, match=r"'w1'.*\[-inf - inf\].*'start_value'"):
        create_samples(parameters, size=SIZE, seed=SEED)


def test_the_sampling_is_checked() -> None:
    """The size and the parameters of a sampling are given."""
    with pytest.raises(ValueError, match="'size' must be a positive integer"):
        create_samples(_bounded(), size=0)
    with pytest.raises(ValueError, match="'parameters' must not be empty"):
        create_samples([], size=SIZE)
    negative = [FitParameter("p1", -2.0, lower_bound=-3.0, upper_bound=-1.0, unit="mM")]
    with pytest.raises(ValueError, match=r"'p1'.*positive upper bound"):
        create_samples(negative, size=SIZE, sampling=SamplingType.LOGUNIFORM)
