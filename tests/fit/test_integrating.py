"""Other integrator settings of an initialized problem (`OptimizationProblem.integrating`).

The checks of the verification of a credibility assessment evaluate the
adopted fits with tightened tolerances and a second integration method. They
evaluate the initialized problem of the fit within `integrating` instead of a
problem initialized with these settings, and the numbers must be the ones of
such a problem, and the ones of the problem after the context the ones before.
"""

import copy
import dataclasses
from typing import Any

import numpy as np
import pytest

from sbmlsim.engine import Engine
from sbmlsim.fit import FitSettings
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.model.tolerances import AbsoluteTolerance


def _evaluation(problem: OptimizationProblem) -> dict[str, list[Any]]:
    """Get the complete data of the problem at its start values."""
    evaluation = problem.residuals(
        xlog=problem.to_scale(np.array(problem.x0, dtype=float)), complete_data=True
    )
    assert isinstance(evaluation, dict)
    return evaluation


def _assert_equal(one: dict[str, list[Any]], other: dict[str, list[Any]]) -> None:
    """Check that two evaluations are equal number for number."""
    assert one.keys() == other.keys()
    for key, values in one.items():
        assert len(values) == len(other[key]), key
        for value, expected in zip(values, other[key], strict=True):
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, expected, err_msg=key)
            else:
                assert value == expected or (value != value and expected != expected)


def _tightened(settings: FitSettings) -> FitSettings:
    """Get the settings with the tolerances divided by 100."""
    tolerance = AbsoluteTolerance.of(settings.absolute_tolerance)
    return dataclasses.replace(
        settings,
        relative_tolerance=settings.relative_tolerance / 100,
        absolute_tolerance=AbsoluteTolerance(
            amount=tolerance.amount / 100,
            concentration=tolerance.concentration / 100,
            other=tolerance.other / 100,
        ),
    )


@pytest.mark.parametrize("engine", [Engine.CORE, Engine.ROADRUNNER])
def test_integrating_is_the_problem_initialized_with_the_settings(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings, engine: Engine
) -> None:
    op_hctz_iv.engine = engine
    definition = copy.deepcopy(op_hctz_iv)
    op_hctz_iv.initialize(fit_settings)
    models = list({id(m): m for m in op_hctz_iv.models}.values())
    states = [model.integrator_state() for model in models]
    reference = _evaluation(op_hctz_iv)
    tightened = _tightened(fit_settings)

    with op_hctz_iv.integrating(tightened, maximum_bdf_order=2):
        alternative = _evaluation(op_hctz_iv)
    fresh = copy.deepcopy(definition)
    fresh.initialize(tightened)
    for model in {id(m): m for m in fresh.models}.values():
        model.set_integrator_settings(maximum_bdf_order=2)
    _assert_equal(alternative, _evaluation(fresh))
    # the second method changes the numbers, so the comparison compares
    with pytest.raises(AssertionError):
        _assert_equal(alternative, reference)

    # the problem integrates as before, its models have their states again
    assert [model.integrator_state() for model in models] == states
    _assert_equal(_evaluation(op_hctz_iv), reference)


def test_integrating_sets_the_settings_back_after_an_error(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    op_hctz_iv.initialize(fit_settings)
    models = list({id(m): m for m in op_hctz_iv.models}.values())
    states = [model.integrator_state() for model in models]
    with (
        pytest.raises(RuntimeError, match="stop"),
        op_hctz_iv.integrating(_tightened(fit_settings), maximum_bdf_order=2),
    ):
        assert [model.integrator_state() for model in models] != states
        raise RuntimeError("stop")
    assert [model.integrator_state() for model in models] == states
    # a setting the engine does not have changes nothing
    with (
        pytest.raises(ValueError, match="no settings"),
        op_hctz_iv.integrating(fit_settings, unknown_setting=1.0),
    ):
        pass
    assert [model.integrator_state() for model in models] == states


def test_integrating_needs_an_initialized_problem(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    with (
        pytest.raises(ValueError, match="must be initialized"),
        op_hctz_iv.integrating(fit_settings),
    ):
        pass
