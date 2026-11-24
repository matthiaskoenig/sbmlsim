"""The evaluation of several parameter vectors as one batch of the engine (#556).

One evaluation of the objective of a fit is one simulation per condition, and
the Jacobian of a step of a least squares optimizer is that evaluation once
per parameter: `(parameters + 1) * conditions` simulations which do not depend
on each other. sbmlsim-core simulates such a batch on threads of its own,
so a problem of that engine evaluates a batch of vectors as one batch of
simulations instead of one simulation per call, see
`OptimizationProblem.evaluate_residuals_many`.

One evaluation on its own is a batch as well: its conditions do not depend on
each other either, so they run on the threads of the engine instead of one
simulation per call on the thread of the caller, see
`OptimizationProblem._batched_evaluation`. A worker of a pool keeps its one
thread, the processes of the pool already use the CPUs of the machine.

The batch is the serial evaluation bitwise: the engine holds no state between
two requests and every thread simulates on an instance of its own, see
`tests/simulator/test_scan_core.py` of the scan, which is the same fan out.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import sbmlsim.fit.optimization as optimization
from sbmlsim import parallel
from sbmlsim.engine import Engine
from sbmlsim.fit import FitSettings
from sbmlsim.fit.objects import MappingKind
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


def _recorded_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[list[list[str]], int]]:
    """Record the batches of the engine and fail a simulation outside one."""
    batches: list[tuple[list[list[str]], int]] = []
    execute_plans = optimization.execute_plans

    def recorded(
        points: Sequence[tuple[Any, Sequence[str]]], model: Any, threads: int
    ) -> Any:
        batches.append(([list(columns) for _, columns in points], threads))
        return execute_plans(points, model, threads)

    def unbatched(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a point was simulated outside the batch")

    monkeypatch.setattr(optimization, "execute_plans", recorded)
    monkeypatch.setattr(optimization, "execute_plan", unbatched)
    return batches


def test_one_evaluation_simulates_its_groups_as_one_batch(
    problem_core: OptimizationProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The groups of one evaluation run as one batch, on the threads of the engine."""
    batches = _recorded_batches(monkeypatch)
    problem_core.residuals(problem_core.to_scale(problem_core.xmodel))

    assert len(batches) == 1, "the groups of the evaluation are not one batch"
    columns, threads = batches[0]
    groups = sum(
        1
        for group in problem_core.mapping_groups
        if any(problem_core.mapping_kinds[k] is MappingKind.TRAINING for k in group)
    )
    assert groups > 1
    assert len(columns) == groups
    assert threads > 1, "the batch of an evaluation runs on one thread"


def test_the_complete_data_batches_the_curves_of_the_groups(
    problem_core: OptimizationProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The curves of the complete data are points of the batch of the evaluation."""
    at_data = _recorded_batches(monkeypatch)
    problem_core.residuals(problem_core.to_scale(problem_core.xmodel))
    points = len(at_data[0][0])

    complete = _recorded_batches(monkeypatch)
    problem_core.residuals(
        problem_core.to_scale(problem_core.xmodel), complete_data=True
    )
    assert len(complete) == 1
    # the validation data of the groups and the curve of every group
    assert len(complete[0][0]) > points


def test_a_batched_evaluation_is_the_serial_one(
    problem_core: OptimizationProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The residuals of a batched evaluation are the ones of its simulations."""
    x = problem_core.to_scale(problem_core.xmodel)
    batched = np.asarray(problem_core.residuals(x))

    monkeypatch.setattr(parallel, "in_worker", lambda: True)
    assert problem_core._threads_of_an_evaluation(set(range(100))) == 1
    serial = np.asarray(problem_core.residuals(x))

    assert np.array_equal(batched, serial), (
        "the residuals of the batch of an evaluation are not the serial ones"
    )


def test_an_evaluation_in_a_worker_keeps_its_thread(
    problem_core: OptimizationProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker of a pool simulates its evaluation one point per call.

    The processes of the pool already use the CPUs of the machine: a worker
    which took them for the threads of its batch would oversubscribe them.
    """
    monkeypatch.setattr(parallel, "in_worker", lambda: True)
    batches: list[int] = []
    execute_plans = optimization.execute_plans

    def recorded(points: Any, model: Any, threads: int) -> Any:
        batches.append(threads)
        return execute_plans(points, model, threads)

    monkeypatch.setattr(optimization, "execute_plans", recorded)
    problem_core.residuals(problem_core.to_scale(problem_core.xmodel))
    assert not batches, "a worker simulated its evaluation as a batch"
