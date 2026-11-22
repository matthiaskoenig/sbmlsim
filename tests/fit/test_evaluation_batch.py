"""The evaluation of several parameter vectors as one batch of the engine (#556).

One evaluation of the objective of a fit is one simulation per condition, and
the Jacobian of a step of a least squares optimizer is that evaluation once
per parameter: `(parameters + 1) * conditions` simulations which do not depend
on each other. sbmlsim-core simulates such a batch on threads of its own,
so a problem of that engine evaluates a batch of vectors as one batch of
simulations instead of one simulation per call, see
`OptimizationProblem.evaluate_residuals_many`.

The batch is the serial evaluation bitwise: the engine holds no state between
two requests and every thread simulates on an instance of its own, see
`tests/simulator/test_scan_core.py` of the scan, which is the same fan out.
"""

import numpy as np
import pytest

from sbmlsim.engine import Engine
from sbmlsim.fit import FitSettings
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.simulator import Simulator
from tests.engines import require


@pytest.fixture
def problem_core(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> OptimizationProblem:
    """Get the initialized problem of the pharmacokinetics on sbmlsim-core."""
    require(Engine.CORE)
    op_hctz_pk.initialize(fit_settings)
    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine=Engine.CORE))
    return op_hctz_pk


def _vectors(problem: OptimizationProblem) -> list[np.ndarray]:
    """Get the parameter vectors of a batch, the model values and two steps."""
    x = problem.to_scale(problem.xmodel)
    return [x, x * 1.001, x * 0.999]


def test_the_residuals_of_a_batch_are_the_ones_of_its_vectors(
    problem_core: OptimizationProblem,
) -> None:
    """A batch evaluates the residuals of every vector, bitwise."""
    vectors = _vectors(problem_core)
    serial = [problem_core.evaluate_residuals(x) for x in vectors]
    batched = problem_core.evaluate_residuals_many(vectors)

    assert len(batched) == len(vectors)
    for k, ((residuals, cost), (r_many, cost_many)) in enumerate(
        zip(serial, batched, strict=True)
    ):
        assert np.array_equal(residuals, r_many), (
            f"the residuals of the vector {k} of the batch are not the ones of "
            f"its evaluation"
        )
        assert cost == cost_many, k


def test_the_costs_of_a_batch_are_the_ones_of_its_vectors(
    problem_core: OptimizationProblem,
) -> None:
    """A batch evaluates the cost of every vector, bitwise."""
    vectors = _vectors(problem_core)
    serial = [problem_core.evaluate_cost(x) for x in vectors]

    assert problem_core.evaluate_cost_many(vectors) == serial


def test_a_batch_simulates_every_vector_in_the_batch(
    problem_core: OptimizationProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every simulation of a batch runs in the batch, none of its own.

    The point of the batch: the simulations of the vectors fan out over the
    threads of the engine at once, so the evaluation of a vector runs no
    simulation while it reads its results.
    """
    from sbmlsim.fit import optimization

    vectors = _vectors(problem_core)
    expected = problem_core.evaluate_residuals_many(vectors)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a vector of the batch simulated a plan of its own")

    monkeypatch.setattr(optimization, "execute_plan", refuse)
    batched = problem_core.evaluate_residuals_many(vectors)

    for (residuals, cost), (r_many, cost_many) in zip(expected, batched, strict=True):
        assert np.array_equal(residuals, r_many)
        assert cost == cost_many


def test_a_batch_of_one_vector_is_its_evaluation(
    problem_core: OptimizationProblem,
) -> None:
    """A batch of one vector gives the residuals of that vector."""
    x = problem_core.to_scale(problem_core.xmodel)

    ((residuals, cost),) = problem_core.evaluate_residuals_many([x])
    one, cost_one = problem_core.evaluate_residuals(x)

    assert np.array_equal(one, residuals)
    assert cost_one == cost


def test_an_empty_batch_is_no_evaluation(problem_core: OptimizationProblem) -> None:
    """A batch without a vector simulates nothing and gives nothing."""
    assert problem_core.evaluate_residuals_many([]) == []
    assert problem_core.evaluate_cost_many([]) == []
