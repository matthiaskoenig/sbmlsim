"""Tests of the evaluation of an optimization problem."""

from dataclasses import replace

import numpy as np
import pytest

from sbmlsim.fit import FitSettings
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
