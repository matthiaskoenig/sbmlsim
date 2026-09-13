"""Tests of running a simulation experiment."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.experiment.runner import model_key
from sbmlsim.fit import FitData, FitMapping
from sbmlsim.model import AbstractModel
from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.simulator.simulation_serial import SimulatorSerial
from sbmlsim.task import Task
from sbmlsim.units import Quantity


class FitMappingExperiment(SimulationExperiment):
    """An experiment whose fit mapping observes what `data()` does not."""

    def models(self) -> dict:
        return {"m": AbstractModel(source=REPRESSILATOR_SBML)}

    def simulations(self) -> dict:
        return {"sim": TimecourseSim([Timecourse(start=0, end=20, steps=20)])}

    def tasks(self) -> dict:
        return {"task": Task(model="m", simulation="sim")}

    def data(self) -> dict:
        return {"X": Data(index="[X]", task="task")}

    def fit_mappings(self) -> dict:
        return {
            "fm": FitMapping(
                self,
                reference=FitData(self, task="task", xid="time", yid="[X]"),
                observable=FitData(self, task="task", xid="time", yid="[Y]"),
            )
        }

    def figures(self) -> dict:
        return {}


def _runner(experiment_class: type[SimulationExperiment]) -> ExperimentRunner:
    """Get a runner of a single experiment class."""
    return ExperimentRunner(
        experiment_classes=[experiment_class],
        simulator=SimulatorSerial(),
        base_path=Path("."),
        data_path=Path("."),
    )


def test_the_selections_cover_the_fit_mappings() -> None:
    """A fit mapping is simulated even when `data()` does not name it.

    The reduced selections are what a run simulates. A `FitData` builds its
    `Data` when it is resolved and does not register it, so an observable
    which is not also declared in `data()` used to be missing from the
    results and the evaluation of the mapping raised.
    """
    runner = _runner(FitMappingExperiment)
    experiment = runner.experiments["FitMappingExperiment"]

    assert "[Y]" in experiment._selections_of_model("m")
    experiment.run(runner.simulator, reduced_selections=True)

    variables = set(experiment.results["task"].xds.data_vars)
    assert {"[X]", "[Y]"} <= variables
    # and the reduction still happens, i.e. not everything is selected
    assert "[Z]" not in variables


def test_the_selections_of_another_model_are_not_added() -> None:
    """The selections of a model are the data of its own tasks."""
    runner = _runner(FitMappingExperiment)
    experiment = runner.experiments["FitMappingExperiment"]
    assert experiment._selections_of_model("other") == {"time"}


# ---------------------------------------------------------------------------
# the loaded models are shared
# ---------------------------------------------------------------------------
class ModelA(FitMappingExperiment):
    """An experiment of the unchanged model."""


class ModelB(FitMappingExperiment):
    """Another experiment of the same model, which must share it."""


class ModelChanged(FitMappingExperiment):
    """The same source with a change, which must not share the model."""

    def models(self) -> dict:
        return {
            "m": AbstractModel(
                source=REPRESSILATOR_SBML,
                changes={"X": Quantity(20.0, "dimensionless")},
            )
        }


@pytest.fixture
def model_loads(monkeypatch: pytest.MonkeyPatch) -> list[AbstractModel]:
    """Record every model which is loaded and compiled."""
    loads: list[AbstractModel] = []
    original = RoadrunnerSBMLModel.from_abstract_model

    def counted(abstract_model: AbstractModel, **kwargs: Any) -> RoadrunnerSBMLModel:
        loads.append(abstract_model)
        return original(abstract_model, **kwargs)

    monkeypatch.setattr(
        RoadrunnerSBMLModel, "from_abstract_model", staticmethod(counted)
    )
    return loads


def _runner_of(*classes: type[SimulationExperiment]) -> ExperimentRunner:
    """Get a runner of several experiment classes."""
    return ExperimentRunner(
        experiment_classes=list(classes),
        simulator=SimulatorSerial(),
        base_path=Path("."),
        data_path=Path("."),
    )


def test_the_same_model_is_loaded_once(model_loads: list[AbstractModel]) -> None:
    """Experiments which name the same model share the loaded instance.

    An experiment builds its own `AbstractModel`, which has no equality of its
    own, so a cache on the object never found anything and every experiment
    compiled the model again.
    """
    runner = _runner_of(ModelA, ModelB)
    assert len(model_loads) == 1
    assert (
        runner.experiments["ModelA"]._models["m"]
        is (runner.experiments["ModelB"]._models["m"])
    )


def test_a_model_with_other_changes_is_its_own(
    model_loads: list[AbstractModel],
) -> None:
    """The changes are part of the model, so they are part of its key."""
    runner = _runner_of(ModelA, ModelChanged)
    assert len(model_loads) == 2
    assert (
        runner.experiments["ModelA"]._models["m"]
        is not (runner.experiments["ModelChanged"]._models["m"])
    )


def test_the_key_of_a_model_describes_it() -> None:
    """Two descriptions of the same model have the same key."""
    a = AbstractModel(source=REPRESSILATOR_SBML)
    b = AbstractModel(source=REPRESSILATOR_SBML)
    changed = AbstractModel(
        source=REPRESSILATOR_SBML, changes={"X": Quantity(20.0, "dimensionless")}
    )

    # the objects are not equal, their keys are
    assert a != b
    assert model_key(a) == model_key(b)
    assert model_key(a) != model_key(changed)


# ---------------------------------------------------------------------------
# the figures are created when they are used
# ---------------------------------------------------------------------------
def test_the_figures_are_not_created_when_nothing_uses_them() -> None:
    """A run which neither shows nor saves the figures does not draw them."""
    runner = _runner(FitMappingExperiment)
    experiment = runner.experiments["FitMappingExperiment"]
    experiment.run(runner.simulator, show_figures=False)

    assert experiment._mpl_figures == {}
    # the simulation still ran
    assert np.asarray(experiment.results["task"].xds["[X]"]).size > 0


def test_the_figures_are_created_for_the_output(tmp_path: Path) -> None:
    """A run with an output path draws and saves them."""
    runner = _runner(FitMappingExperiment)
    experiment = runner.experiments["FitMappingExperiment"]
    experiment.run(runner.simulator, output_path=tmp_path)

    assert (tmp_path / f"{experiment.sid}.json").exists()
