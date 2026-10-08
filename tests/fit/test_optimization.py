"""Tests of the evaluation of an optimization problem."""

from dataclasses import replace

import numpy as np
import pytest

from sbmlsim.fit import FitParameter, FitSettings
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.optimization import OptimizationProblem


@pytest.mark.parametrize("variable_step_size", [True, False])
def test_the_residuals_do_not_depend_on_earlier_evaluations(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    variable_step_size: bool,
) -> None:
    """The first evaluation of a problem equals every later one, bit for bit.

    The absolute tolerance of the simulator is scaled by the volumes of the
    compartments; the volumes of the state after a simulation must not change
    the tolerance of the next one.
    """
    problem = op_hctz_pk
    problem.initialize(replace(fit_settings, variable_step_size=variable_step_size))
    x0 = problem.to_scale(problem.x0)
    residuals = [np.asarray(problem.residuals(x0), dtype=float) for _ in range(3)]
    np.testing.assert_array_equal(residuals[0], residuals[1])
    np.testing.assert_array_equal(residuals[1], residuals[2])


def test_the_initial_time_step_reaches_the_models(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The initial time step of the settings is the one of every model of a fit."""
    problem = op_hctz_pk
    problem.initialize(replace(fit_settings, initial_time_step=1e-9))
    for model in problem.models:
        integrator = model.r_loaded.getIntegrator()
        assert integrator.getValue("initial_time_step") == pytest.approx(1e-9)


def test_a_parameter_whose_target_is_not_an_entity(
    definition_hctz_pk: FitDefinition, fit_settings: FitSettings
) -> None:
    """A misspelled target is named, not a symbol roadrunner does not know."""
    typo = FitParameter(
        pid="Ka_dis",
        start_value=1.0,
        lower_bound=1e-4,
        upper_bound=10,
        unit="1/hr",
        target="Ka_dis_hctzz",
    )
    definition = replace(
        definition_hctz_pk, parameters=[*definition_hctz_pk.parameters[1:], typo]
    )
    problem = definition.problem(opid="hctz_typo")
    with pytest.raises(
        ValueError,
        match=r"'hctz_typo': FitParameter 'Ka_dis' writes 'Ka_dis_hctzz', which "
        r"is not an entity of the model .*'sciml:<id>'",
    ):
        problem.initialize(fit_settings)
