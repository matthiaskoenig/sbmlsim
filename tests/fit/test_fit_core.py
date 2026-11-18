"""The request of a fit of the engine core, built from an optimization problem (#556).

The engine fits a problem itself: its optimizer, the runs of its starts and
the profiles of its identifiability run in rust on the threads of one batch,
where an optimizer in python evaluates one point at a time and waits for the
engine between the steps. A problem reaches that path when its whole
objective is in a request of the contract `fit/1`, see `sbmlsim.fit.fit_core`.

The request is checked against the problem itself: the residuals the engine
computes from the request are the ones the problem computes for the same
parameters, every point of them, which is what makes the fit of the engine
the fit of the problem. The iv data of the HCTZ example is the problem of
these tests, four training mappings of one simulation, with the weighting of
a real fit (`fit_settings`: normalized residuals, the points of the curves
and the errors of the data).
"""

import dataclasses
from typing import Any

import numpy as np
import pytest

from sbmlsim.engine import Engine
from sbmlsim.fit import FitSettings
from sbmlsim.fit.fit_core import fit_request, gaps_of, loaded_model
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import LossFunctionType, ObjectiveType, ResidualType
from sbmlsim.simulator import Simulator
from tests.engines import require_core_fit


def on_core(problem: OptimizationProblem, settings: FitSettings) -> OptimizationProblem:
    """Initialize a problem on the engine core, the engine which fits itself."""
    require_core_fit()
    problem.initialize(settings)
    problem.set_simulator(Simulator(n_workers=1, engine=Engine.CORE))
    assert problem.engine is Engine.CORE
    return problem


def core_problem(problem: OptimizationProblem) -> Any:
    """Get the objective of the engine for the request of a problem."""
    return loaded_model(problem).fit_problem(fit_request(problem), threads=1)


def test_the_request_of_a_problem_is_its_simulations_and_its_training_data(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A condition is a simulation group of the problem, a curve a training mapping."""
    problem = on_core(op_hctz_iv, fit_settings)

    request = fit_request(problem)

    assert request["contract"] == "fit/1"
    assert len(request["conditions"]) == len(problem.mapping_groups)
    assert [p["id"] for p in request["parameters"]] == [
        parameter.target_id for parameter in problem.parameters
    ]
    training = problem.indices(MappingKind.TRAINING)
    assert len(request["curves"]) == len(training)
    assert [curve["id"] for curve in request["curves"]] == [
        problem.mapping_keys[k] for k in training
    ]
    conditions = {condition["id"] for condition in request["conditions"]}
    for curve in request["curves"]:
        assert curve["condition"] in conditions
        assert len(curve["times"]) == len(curve["values"]) == len(curve["noise"])
        assert all(noise > 0.0 for noise in curve["noise"])


def test_the_residuals_of_the_engine_are_the_ones_of_the_problem(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The engine computes the residuals of the problem, point for point."""
    problem = on_core(op_hctz_iv, fit_settings)
    core = core_problem(problem)

    for factor in (1.0, 1.3, 0.7):
        x = problem.to_scale(problem.xmodel * factor)
        expected = np.asarray(problem.residuals(x), dtype=float)

        residuals = np.asarray(core.residuals(x), dtype=float)

        assert residuals.shape == expected.shape
        np.testing.assert_allclose(residuals, expected, rtol=1e-9, atol=1e-12)
        assert core.cost(x) == pytest.approx(
            0.5 * float(np.sum(np.square(expected))), rel=1e-9
        )


def test_the_start_and_the_bounds_of_the_request_are_the_ones_of_the_parameters(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The search space of the engine is the search space of the problem."""
    problem = on_core(op_hctz_iv, fit_settings)
    core = core_problem(problem)

    np.testing.assert_allclose(core.start, problem.to_scale(problem.xmodel), rtol=1e-12)
    lower, upper = core.bounds
    np.testing.assert_allclose(
        lower,
        problem.to_scale([p.lower_bound for p in problem.parameters]),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        upper,
        problem.to_scale([p.upper_bound for p in problem.parameters]),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"objective": ObjectiveType.LIKELIHOOD}, "objective"),
        ({"loss_function": LossFunctionType.SOFT_L1}, "loss_function"),
        ({"residual": ResidualType.ABSOLUTE_TO_BASELINE}, "residual"),
    ],
)
def test_an_objective_the_engine_has_not_is_a_gap_of_the_problem(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    change: dict[str, object],
    code: str,
) -> None:
    """A problem whose objective the request cannot carry names what keeps it out."""
    problem = on_core(op_hctz_iv, dataclasses.replace(fit_settings, **change))

    gaps = gaps_of(problem)

    assert [gap.code for gap in gaps] == [code]
    assert gaps[0].message


def test_a_problem_of_another_engine_is_no_fit_of_the_engine(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Only a problem which simulates in the engine core fits in it."""
    op_hctz_iv.initialize(fit_settings)
    assert op_hctz_iv.engine is Engine.ROADRUNNER

    gaps = gaps_of(op_hctz_iv)

    assert [gap.code for gap in gaps] == ["engine"]


def test_a_problem_without_gaps_builds_a_request(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The gaps of the problem of a fit of the engine are none."""
    problem = on_core(op_hctz_iv, fit_settings)

    assert gaps_of(problem) == []
