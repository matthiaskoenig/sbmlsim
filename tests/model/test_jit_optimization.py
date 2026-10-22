"""The optimization level the code of a model is compiled with (#331)."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import roadrunner

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel, JitOptimization, RoadrunnerSBMLModel
from sbmlsim.model.model_roadrunner import (
    DEFAULT_JIT_OPTIMIZATION,
    JIT_OPTIMIZATION_VARIABLE,
    resolve_jit_optimization,
)
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.worker import ModelSpec
from sbmlsim.task import Task
from tests.simulator.models import sbml


def test_the_level_without_the_variable_is_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(JIT_OPTIMIZATION_VARIABLE, raising=False)
    assert resolve_jit_optimization() is DEFAULT_JIT_OPTIMIZATION
    assert RoadrunnerSBMLModel(source=sbml()).jit_optimization is (
        DEFAULT_JIT_OPTIMIZATION
    )


def test_the_variable_sets_the_level_of_a_model_without_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(JIT_OPTIMIZATION_VARIABLE, "O0")
    assert RoadrunnerSBMLModel(source=sbml()).jit_optimization is JitOptimization.O0
    assert (
        RoadrunnerSBMLModel(source=sbml(), jit_optimization="O3").jit_optimization
        is JitOptimization.O3
    )


def test_an_unknown_level_names_where_it_was_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match=r"jit_optimization 'fast'.*'O0', 'O1'"):
        RoadrunnerSBMLModel(source=sbml(), jit_optimization="fast")
    monkeypatch.setenv(JIT_OPTIMIZATION_VARIABLE, "o0")
    with pytest.raises(ValueError, match=f"{JIT_OPTIMIZATION_VARIABLE} 'o0'"):
        RoadrunnerSBMLModel(source=sbml())


def test_a_model_is_compiled_with_lljit_at_its_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    levels: list[int] = []
    set_level = roadrunner.LoadSBMLOptions.setLLJitOptimizationLevel

    def spy(options: roadrunner.LoadSBMLOptions, level: int) -> None:
        levels.append(level)
        set_level(options, level)

    monkeypatch.setattr(roadrunner.LoadSBMLOptions, "setLLJitOptimizationLevel", spy)
    for level in JitOptimization:
        RoadrunnerSBMLModel(source=sbml(), jit_optimization=level)
    assert levels == [
        roadrunner.LoadSBMLOptions.NONE,
        roadrunner.LoadSBMLOptions.LESS,
        roadrunner.LoadSBMLOptions.DEFAULT,
        roadrunner.LoadSBMLOptions.AGGRESSIVE,
    ]


def test_the_levels_simulate_the_same() -> None:
    simulator = Simulator(n_workers=1)
    simulation = Simulation(end=10, steps=20)
    results = [
        simulator.simulate(
            RoadrunnerSBMLModel(source=sbml(), jit_optimization=level), simulation
        )
        for level in JitOptimization
    ]
    for result in results[1:]:
        np.testing.assert_allclose(result.values, results[0].values, rtol=1e-12)


def test_a_worker_compiles_a_model_at_the_level_of_the_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RoadrunnerSBMLModel(source=sbml(), jit_optimization="O0")
    spec = ModelSpec.of(model)
    assert spec.jit_optimization is JitOptimization.O0
    other = RoadrunnerSBMLModel(source=sbml(), jit_optimization="O2")
    assert ModelSpec.of(other).key != spec.key
    # the environment of a worker is not the one of the parent
    monkeypatch.setenv(JIT_OPTIMIZATION_VARIABLE, "O2")
    loaded = pickle.loads(pickle.dumps(spec)).load()
    assert loaded.jit_optimization is JitOptimization.O0
    assert ModelSpec.of(loaded) == spec


class _Experiment(SimulationExperiment):
    model_path: Path

    def models(self) -> dict:
        return {"m": AbstractModel(self.model_path)}

    def simulations(self) -> dict:
        return {"sim": Simulation(end=1, steps=1)}

    def tasks(self) -> dict:
        return {"task": Task(model="m", simulation="sim")}

    def data(self) -> dict:
        return {"A": Data("[A]", task="task")}


def test_a_runner_compiles_its_models_at_its_level(tmp_path: Path) -> None:
    _Experiment.model_path = tmp_path / "model.xml"
    _Experiment.model_path.write_text(sbml())
    runner = ExperimentRunner(
        _Experiment,
        base_path=tmp_path,
        data_path=tmp_path,
        on_error="raise",
        jit_optimization="O1",
    )
    assert runner.jit_optimization is JitOptimization.O1
    assert [model.jit_optimization for model in runner.models.values()] == [
        JitOptimization.O1
    ]
