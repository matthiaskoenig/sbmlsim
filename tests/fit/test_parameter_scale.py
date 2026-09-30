"""Tests of the parameter scale of a fit.

The scale is the space the optimizer searches the parameters in. It is a
property of the optimization and not of the model or of the data, which is why
it is part of the `FitSettings` and why PEtab v2 removed the `parameterScale`
of its parameter table.
"""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from sbmlsim.fit import FitParameter, FitSettings, ParameterSet
from sbmlsim.fit.fisher import fisher_information
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
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The problem searches the space its settings name."""
    for scale in ParameterScaleType:
        problem = op_hctz_pk
        problem.initialize(replace(fit_settings, parameter_scale=scale))
        assert problem.parameter_scale is scale

        x = np.asarray(problem.x0, dtype=float)
        assert np.allclose(problem.from_scale(problem.to_scale(x)), x)


def test_the_cost_does_not_depend_on_the_scale(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The same parameters have the same cost in every space.

    The scale is how the optimizer walks, not what it optimizes.
    """
    costs = []
    for scale in ParameterScaleType:
        problem = op_hctz_iv
        problem.initialize(replace(fit_settings, parameter_scale=scale))
        x = np.asarray(problem.x0, dtype=float)
        costs.append(problem.cost_least_square(problem.to_scale(x)))

    assert costs[0] == pytest.approx(costs[1])
    assert costs[0] == pytest.approx(costs[2])


def test_a_logarithm_needs_positive_bounds(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The bounds have to suit the space the optimizer searches.

    A logarithm of a bound which is not positive is not a number, so the
    problem says so instead of optimizing NaN. On the linear scale the same
    bounds are fine.
    """
    problem = op_hctz_pk
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
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """An optimizer cannot search an interval which has no end."""
    problem = op_hctz_pk
    problem.parameters = [deepcopy(p) for p in problem.parameters]
    problem.parameters[0].upper_bound = np.inf

    for scale in ParameterScaleType:
        with pytest.raises(ValueError, match="finite"):
            problem.initialize(replace(fit_settings, parameter_scale=scale), force=True)


# --- THE SCALE OF A PARAMETER ---


def test_a_parameter_has_the_scale_of_the_settings_by_default() -> None:
    """Every existing definition means what it means without a scale."""
    parameter = FitParameter("k", 1.0, 0.1, 10.0, unit="1/min")
    assert parameter.scale is None
    assert parameter.to_dict()["scale"] is None
    assert FitParameter(**parameter.to_dict()) == parameter


@pytest.mark.parametrize("scale", list(ParameterScaleType))
def test_the_scale_of_a_parameter_is_stored(scale: ParameterScaleType) -> None:
    """The scale is a part of the parameter and of its serialization."""
    parameter = FitParameter("k", 1.0, 0.1, 10.0, unit="1/min", scale=scale)
    assert parameter.scale is scale
    assert parameter.to_dict()["scale"] == scale.name
    assert FitParameter(**parameter.to_dict()).scale is scale
    restored = FitParameter.from_json(str(parameter.to_json()))
    assert restored.scale is scale
    assert restored == parameter
    assert parameter != FitParameter("k", 1.0, 0.1, 10.0, unit="1/min")
    assert (
        FitParameter("k", 1.0, 0.1, 10.0, unit="1/min", scale=scale.name) == parameter
    )


@pytest.mark.parametrize("scale", ["lin", "LOG2", 1, 2.0, ParameterScaleType])
def test_a_scale_which_is_not_a_scale(scale: object) -> None:
    """A scale is a `ParameterScaleType` or the name of one."""
    with pytest.raises(ValueError, match=r"FitParameter 'k': the scale"):
        FitParameter("k", 1.0, 0.1, 10.0, unit="1/min", scale=scale)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lower_bound": np.nan}, "the 'lower_bound' is 'nan'"),
        ({"upper_bound": np.nan}, "the 'upper_bound' is 'nan'"),
        ({"upper_bound": None}, "the 'upper_bound' is 'None'"),
        ({"start_value": np.nan}, "the start value 'nan' is not a finite number"),
        ({"start_value": np.inf}, "the start value 'inf' is not a finite number"),
    ],
)
def test_the_values_of_a_parameter_are_numbers(kwargs: dict, message: str) -> None:
    """A value which is no number is an error and not a bound which holds."""
    arguments: dict[str, Any] = {
        "start_value": 1.0,
        "lower_bound": 0.1,
        "upper_bound": 10.0,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=rf"FitParameter 'k': {message}"):
        FitParameter("k", unit="1/min", **arguments)


def _with_scales(
    problem: OptimizationProblem, scales: list[ParameterScaleType | None]
) -> OptimizationProblem:
    """Get the problem with copies of its parameters which have the scales."""
    problem.parameters = [deepcopy(p) for p in problem.parameters]
    for parameter, scale in zip(problem.parameters, scales, strict=True):
        parameter.scale = scale
    return problem


def test_the_problem_transforms_every_parameter_with_its_scale(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A parameter without a scale has the one of the settings."""
    problem = _with_scales(
        op_hctz_pk, [ParameterScaleType.LINEAR, None, ParameterScaleType.LOG]
    )
    with pytest.raises(ValueError, match="must be initialized first"):
        problem.to_scale([1.0, 2.0, 3.0])

    problem.initialize(fit_settings)
    assert problem.parameter_scale is ParameterScaleType.LOG10
    assert problem.scales_initialized == [
        ParameterScaleType.LINEAR,
        ParameterScaleType.LOG10,
        ParameterScaleType.LOG,
    ]
    x = np.array([0.5, 100.0, np.e])
    scaled = problem.to_scale(x)
    np.testing.assert_allclose(scaled, [0.5, 2.0, 1.0])
    np.testing.assert_allclose(problem.from_scale(scaled), x)

    for values in ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0], 1.0, [[1.0, 2.0, 3.0]]):
        with pytest.raises(ValueError, match="one value per parameter"):
            problem.to_scale(values)
        with pytest.raises(ValueError, match="one value per parameter"):
            problem.from_scale(values)


def test_the_cost_does_not_depend_on_the_scales_of_the_parameters(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The scales are how the optimizer walks, not what it optimizes."""
    problem = op_hctz_iv
    problem.initialize(fit_settings)
    x = np.asarray(problem.x0, dtype=float)
    cost = problem.cost_least_square(problem.to_scale(x))

    n = len(problem.parameters)
    scales: list[ParameterScaleType | None] = [
        [ParameterScaleType.LINEAR, ParameterScaleType.LOG, None][k % 3]
        for k in range(n)
    ]
    problem = _with_scales(problem, scales)
    problem.initialize(fit_settings, force=True)
    assert problem.cost_least_square(problem.to_scale(x)) == pytest.approx(
        cost, rel=1e-12
    )


def test_a_parameter_on_the_linear_scale_may_be_negative(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The scale of a parameter decides which bounds it may have."""
    problem = _with_scales(op_hctz_pk, [ParameterScaleType.LINEAR, None, None])
    problem.parameters[0].lower_bound = -1.0
    problem.initialize(fit_settings)
    lower = problem.to_scale([p.lower_bound for p in problem.parameters])
    assert lower[0] == -1.0
    assert lower[1] == pytest.approx(np.log10(problem.parameters[1].lower_bound))

    problem.parameters[1].lower_bound = -1.0
    with pytest.raises(
        ValueError, match=rf"'LOG10'.*positive.*'{problem.parameters[1].pid}'"
    ):
        problem.initialize(fit_settings, force=True)


def test_the_fisher_information_uses_the_scales_of_the_parameters(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The intervals are transformed back with the scale of every parameter."""
    problem = op_hctz_iv
    n = len(problem.parameters)
    problem = _with_scales(problem, [ParameterScaleType.LINEAR] + [None] * (n - 1))
    problem.initialize(fit_settings)
    pset = ParameterSet.from_fit_parameters(
        problem.parameters, x=np.asarray(problem.x0, dtype=float), sid="start"
    )
    fim = fisher_information(problem, fit_settings, pset)
    assert fim.scale is ParameterScaleType.LOG10
    assert fim.parameter_scales == problem.scales_initialized
    assert fim.to_dict()["scales"] == ["LINEAR"] + ["LOG10"] * (n - 1)
    np.testing.assert_allclose(fim.from_scale(fim.to_scale(fim.values)), fim.values)
    lower, upper = fim.confidence_intervals()
    errors = fim.standard_errors
    if np.isfinite(errors[0]):
        # symmetric on the linear scale, and not on a logarithmic one
        assert fim.values[0] - lower[0] == pytest.approx(upper[0] - fim.values[0])

    with pytest.raises(ValueError, match="one scale per parameter"):
        replace(fim, scales=[ParameterScaleType.LINEAR])
    assert replace(fim, scales=[]).parameter_scales == [ParameterScaleType.LOG10] * n
