"""The evaluation of a fit problem with the engine of its simulator (#219).

The tests of the evaluation with COPASI and AMICI are in
`test_evaluation_engines.py`; these are the ones of the switch of the engine
which need no engine besides roadrunner, or COPASI as any other engine.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sbmlsim.engine import Engine
from sbmlsim.fit import (
    FitSettings,
)
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.simulator import Simulator
from tests.fit.scalar_experiment import scalar_problem


def test_set_simulator_of_the_same_engine_keeps_the_models(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    op_hctz_pk.initialize(fit_settings)
    models = list(op_hctz_pk.models)
    plans = list(op_hctz_pk.plans)
    simulator = Simulator(n_workers=1)
    op_hctz_pk.set_simulator(simulator)
    assert op_hctz_pk.engine is Engine.ROADRUNNER
    assert all(a is b for a, b in zip(op_hctz_pk.models, models, strict=True))
    assert all(a is b for a, b in zip(op_hctz_pk.plans, plans, strict=True))
    assert op_hctz_pk.runner_initialized.simulator is simulator


def test_an_unsupported_feature_raises_instead_of_failing_the_evaluation(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feature the engine does not simulate is raised, not a failed point.

    `NotImplementedError` is a `RuntimeError`, which an evaluation turns into
    the residuals of a failed integration; a feature COPASI does not simulate
    fails for every parameter, so it is raised. Both kinds of groups raise
    it: the groups of the HCTZ problem output the times of their data
    (`_simulate_at_data`), the group of the cmax keeps the output of its
    simulation (`_simulate_points`).
    """
    import sbmlsim.fit.optimization as optimization

    op_hctz_pk.initialize(fit_settings)
    assert all(graph is None for graph in op_hctz_pk._group_graphs)
    scalar = scalar_problem(tmp_path)
    scalar.initialize(fit_settings)
    assert all(graph is not None for graph in scalar._group_graphs)

    def unsupported(*args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("the engine does not simulate it")

    monkeypatch.setattr(optimization, "execute_plan", unsupported)
    for problem in (op_hctz_pk, scalar):
        x = problem.to_scale(problem.xmodel)
        with pytest.raises(NotImplementedError, match="does not simulate"):
            problem.residuals(x)
        with pytest.raises(NotImplementedError, match="does not simulate"):
            problem.residuals(x, complete_data=True)


def test_a_problem_of_roadrunner_keeps_the_settings_of_cvode(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    op_hctz_pk.initialize(fit_settings)
    simulator = op_hctz_pk.runner_initialized.simulator
    assert simulator is not None
    assert simulator.integrator_settings["variable_step_size"] is (
        fit_settings.variable_step_size
    )
    assert simulator.integrator_settings["initial_time_step"] == (
        fit_settings.initial_time_step
    )


def test_a_failed_set_simulator_leaves_the_problem_as_it_was(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op_hctz_pk.initialize(fit_settings)
    x = op_hctz_pk.to_scale(op_hctz_pk.xmodel)
    r = np.asarray(op_hctz_pk.residuals(x), dtype=float)
    runner = op_hctz_pk.runner_initialized
    models, plans = list(op_hctz_pk.models), list(op_hctz_pk.plans)
    runner_models, runner_simulator = runner.models, runner.simulator
    experiments = {
        sid: (experiment._models, experiment.simulator)
        for sid, experiment in runner.experiments.items()
    }

    def fail(self: OptimizationProblem) -> None:
        raise ValueError("the plans do not compile")

    monkeypatch.setattr(OptimizationProblem, "_compile_plans", fail)
    with pytest.raises(ValueError, match="do not compile"):
        op_hctz_pk.set_simulator(Simulator(n_workers=1, engine="copasi"))
    monkeypatch.undo()

    assert op_hctz_pk.engine is Engine.ROADRUNNER
    assert all(a is b for a, b in zip(op_hctz_pk.models, models, strict=True))
    assert all(a is b for a, b in zip(op_hctz_pk.plans, plans, strict=True))
    assert runner.models is runner_models
    assert runner.simulator is runner_simulator
    assert runner.engine is Engine.ROADRUNNER
    for sid, experiment in runner.experiments.items():
        assert experiment._models is experiments[sid][0]
        assert experiment.simulator is experiments[sid][1]
    np.testing.assert_array_equal(np.asarray(op_hctz_pk.residuals(x), float), r)
