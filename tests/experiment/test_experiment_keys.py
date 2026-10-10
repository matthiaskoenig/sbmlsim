"""Tests of the sid of an experiment, its keys and the ids of its data."""

from pathlib import Path

import pytest

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.simulator import Simulator
from tests.experiment.test_experiment_observables import PKExperiment, _runner


class Plain(SimulationExperiment):
    """An experiment without definitions."""


def test_a_passed_sid_is_kept() -> None:
    assert Plain(sid="X", base_path=Path("."), data_path=Path(".")).sid == "X"
    assert Plain(base_path=Path("."), data_path=Path(".")).sid == "Plain"


def test_a_runner_experiment_has_the_class_name_as_sid() -> None:
    runner = ExperimentRunner(
        experiment_classes=[Plain],
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
    )
    assert runner.experiments["Plain"].sid == "Plain"


@pytest.mark.parametrize("key", ["5-hydroxyomeprazole", "a.b", "1a", "a b", "a-"])
def test_a_key_must_be_a_full_sid(key: str) -> None:
    class Bad(PKExperiment):
        def datasets(self) -> dict:
            return {key: None}

    with pytest.raises(ValueError, match=r"\[a-zA-Z_\]\[a-zA-Z0-9_\]\*.*" + key):
        _runner(Bad)


def test_an_amount_and_its_concentration_have_different_sids() -> None:
    amount = Data("C", task="t")
    conc = Data("[C]", task="t")
    assert amount.sid == "t__C" and conc.sid == "t__conc__C"
    assert conc.selection == "[C]"
    assert Data("pk.cmax", task="t").sid == "t__pk__cmax"
    assert Data("dose.PODOSE", dataset="d").sid == "d__dose__PODOSE"
    assert Data("[C]", task="t", sid="mine").sid == "mine"


def test_an_amount_and_a_concentration_are_both_kept() -> None:
    class Both(PKExperiment):
        def data(self) -> dict:
            self.add_selections_data(["time", "C", "[C]"])
            return {}

    runner = _runner(Both)
    experiment = runner.experiments["Both"]
    assert {"task_sim__C", "task_sim__conc__C"} <= set(experiment._data)
    assert experiment._data["task_sim__conc__C"].selection == "[C]"
    experiment.run(runner.simulator)
    names = set(experiment.results["task_sim"].ds.data_vars)
    assert {"C", "[C]"} <= names
