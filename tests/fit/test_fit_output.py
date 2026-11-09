"""A fit simulates what its training data needs and not more."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import sbmlsim.fit.optimization as optimization
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings
from sbmlsim.simulator.plan import Plan


def test_a_fit_ends_at_the_last_time_of_its_data(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The simulation of a group which outputs the times of its data ends there.

    The curve of the evaluation keeps the whole simulation.
    """
    problem = op_hctz_pk
    problem.initialize(fit_settings)
    x = np.asarray(problem.x0, dtype=float)
    ended = 0
    for k_group, compiled in enumerate(problem.plans):
        plan = problem.evaluated_plan(k_group, x)
        if problem._group_ends[k_group] is None:
            assert plan.end == compiled.end
            continue
        ended += 1
        assert plan.end == compiled.times[-1] < compiled.end
        assert problem.evaluated_plan(k_group, x, curve=True).end == compiled.end
    assert ended


def test_a_fit_selects_what_its_training_data_reads(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group with training and other data selects what its training data reads."""
    problem = op_hctz_pk
    problem.initialize(fit_settings)
    executed: list[list[str]] = []
    execute_plan = optimization.execute_plan

    def recorded(plan: Plan, model: Any, selections: Sequence[str]) -> Any:
        executed.append(list(selections))
        return execute_plan(plan, model, selections)

    monkeypatch.setattr(optimization, "execute_plan", recorded)
    problem.residuals(problem.to_scale(np.asarray(problem.x0, dtype=float)))

    expected = []
    mixed = False
    for group in problem.mapping_groups:
        training = [
            k for k in group if problem.mapping_kinds[k] is MappingKind.TRAINING
        ]
        if not training:
            continue
        needed = sorted({s for k in training for s in problem.selections[k]})
        mixed |= needed != sorted({s for k in group for s in problem.selections[k]})
        expected.append(needed)
    # the problem has a group whose other data reads more
    assert mixed
    assert executed == expected
