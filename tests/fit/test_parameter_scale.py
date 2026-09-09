"""Tests of the parameter scale of a fit.

The scale is the space the optimizer searches the parameters in. It is a
property of the optimization and not of the model or of the data, which is why
it is part of the `FitSettings` and why PEtab v2 removed the `parameterScale`
of its parameter table.
"""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from sbmlsim.fit import FitSettings
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType

#: values which span orders of magnitude, i.e. what a log scale is for
VALUES = np.array([1e-6, 1.0, 25.0, 1e3])


@pytest.mark.parametrize("scale", list(ParameterScaleType))
def test_the_scale_round_trips(scale: ParameterScaleType) -> None:
    """A parameter is the same after the way there and back."""
    scaled = scale.to_scale(VALUES)
    assert np.allclose(scale.from_scale(scaled), VALUES)


def test_the_scales_are_what_they_say() -> None:
    """The transformations are the logarithm, the log10 and the identity."""
    assert np.allclose(ParameterScaleType.LINEAR.to_scale(VALUES), VALUES)
    assert np.allclose(ParameterScaleType.LOG10.to_scale(VALUES), np.log10(VALUES))
    assert np.allclose(ParameterScaleType.LOG.to_scale(VALUES), np.log(VALUES))

    assert ParameterScaleType.LOG10.is_log
    assert ParameterScaleType.LOG.is_log
    assert not ParameterScaleType.LINEAR.is_log


def test_the_default_is_log10() -> None:
    """A rate constant spans orders of magnitude, so the default is log10."""
    assert FitSettings().parameter_scale is ParameterScaleType.LOG10


def test_the_scale_is_stored_with_the_settings() -> None:
    """The settings of a fit carry the scale, so a report uses the same one."""
    settings = FitSettings(parameter_scale=ParameterScaleType.LINEAR)
    restored = FitSettings.from_dict(settings.to_dict())
    assert restored.parameter_scale is ParameterScaleType.LINEAR
    assert restored == settings

    # settings which were stored before the scale existed are log10, which is
    # what the optimization did
    old = settings.to_dict()
    del old["parameter_scale"]
    assert FitSettings.from_dict(old).parameter_scale is ParameterScaleType.LOG10


def test_the_problem_transforms_with_its_scale(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The problem searches the space its settings name."""
    for scale in ParameterScaleType:
        problem = op_hctz_pkiv
        problem.initialize(replace(fit_settings, parameter_scale=scale))
        assert problem.parameter_scale is scale

        x = np.asarray(problem.x0, dtype=float)
        assert np.allclose(problem.from_scale(problem.to_scale(x)), x)


def test_the_cost_does_not_depend_on_the_scale(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The same parameters have the same cost in every space.

    The scale is how the optimizer walks, not what it optimizes.
    """
    costs = []
    for scale in ParameterScaleType:
        problem = op_hctz_pkiv
        problem.initialize(replace(fit_settings, parameter_scale=scale))
        x = np.asarray(problem.x0, dtype=float)
        costs.append(problem.cost_least_square(problem.to_scale(x)))

    assert costs[0] == pytest.approx(costs[1])
    assert costs[0] == pytest.approx(costs[2])


def test_a_logarithm_needs_positive_bounds(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The bounds have to suit the space the optimizer searches.

    A logarithm of a bound which is not positive is not a number, so the
    problem says so instead of optimizing NaN. On the linear scale the same
    bounds are fine.
    """
    problem = op_hctz_pkiv
    # the definition of the example shares its `FitParameter` objects between
    # the problems, so the test works on copies of them
    problem.parameters = [deepcopy(p) for p in problem.parameters]
    problem.parameters[0].lower_bound = -1.0
    problem.parameters[0].start_value = 0.5

    for scale in [ParameterScaleType.LOG10, ParameterScaleType.LOG]:
        with pytest.raises(ValueError, match="positive"):
            problem.initialize(replace(fit_settings, parameter_scale=scale), force=True)

    # the linear scale searches the interval as it is
    problem.initialize(
        replace(fit_settings, parameter_scale=ParameterScaleType.LINEAR),
        force=True,
    )
    assert problem.parameter_scale is ParameterScaleType.LINEAR


def test_an_infinite_bound_is_never_allowed(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """An optimizer cannot search an interval which has no end."""
    problem = op_hctz_pkiv
    problem.parameters = [deepcopy(p) for p in problem.parameters]
    problem.parameters[0].upper_bound = np.inf

    for scale in ParameterScaleType:
        with pytest.raises(ValueError, match="finite"):
            problem.initialize(replace(fit_settings, parameter_scale=scale), force=True)
