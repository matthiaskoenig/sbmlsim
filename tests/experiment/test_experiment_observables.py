"""An experiment declares observables and every task computes the ones its data read."""

import json
from pathlib import Path

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.simulation import (
    PK,
    Change,
    Dimension,
    Formula,
    Observable,
    Scan,
    Simulation,
)
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from tests.simulator.models import sbml_pk


class PKExperiment(SimulationExperiment):
    """A dosed one-compartment model, once and over three doses."""

    def models(self) -> dict:
        return {"m": AbstractModel(source=sbml_pk())}

    def simulations(self) -> dict:
        simulation = Simulation(
            end=48, steps=96, changes=[Change(0, {"PODOSE": Q(100, "mg")})]
        )
        doses = Dimension("dose", values={"PODOSE": Q([50.0, 100.0, 200.0], "mg")})
        return {"sim": simulation, "scan": Scan(simulation, [doses])}

    def observables(self) -> dict[str, Observable]:
        return {
            "pk": PK("pk", "[C]", dose="PODOSE", route="oral"),
            "cmax": Formula("cmax", "max([C])"),
            # never read by a data: never compiled, so its unknown symbol is no error
            "unused": Formula("unused", "nope * 2"),
        }

    def tasks(self) -> dict:
        return {
            "task_sim": Task(model="m", simulation="sim"),
            "task_scan": Task(model="m", simulation="scan"),
        }

    def data(self) -> dict:
        self.add_selections_data(["time", "[C]"])
        return {
            "cmax_scan": Data("cmax", task="task_scan"),
            "pk_cmax_scan": Data("pk.cmax", task="task_scan"),
            "dose_scan": Data("PODOSE", task="task_scan"),
        }


def _runner(experiment_class: type[SimulationExperiment]) -> ExperimentRunner:
    return ExperimentRunner(
        experiment_classes=[experiment_class],
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
    )


@pytest.fixture(scope="module")
def experiment() -> SimulationExperiment:
    runner = _runner(PKExperiment)
    experiment = runner.experiments["PKExperiment"]
    experiment.run(runner.simulator)
    return experiment


def test_a_task_computes_the_observables_its_data_read(
    experiment: SimulationExperiment,
) -> None:
    scan = experiment.results["task_scan"]
    assert set(scan.ds.data_vars) == {"cmax", "pk.cmax", "[C]"}
    assert scan["cmax"].dims == ("dose",) and scan["[C]"].dims == ("dose", "time")
    np.testing.assert_allclose(scan["cmax"].values, scan["[C]"].max("time").values)
    single = experiment.results["task_sim"]
    assert "cmax" not in single.ds.data_vars and "[C]" in single.ds.data_vars


def test_the_data_of_an_observable_is_a_labelled_array(
    experiment: SimulationExperiment,
) -> None:
    cmax = Data("pk.cmax", task="task_scan").get_data(experiment)
    assert cmax.dims == ("dose",)
    np.testing.assert_allclose(cmax["PODOSE"].values, [50.0, 100.0, 200.0])
    assert np.all(np.diff(cmax.values) > 0)
    assert cmax.attrs["units"]
    middle = Data("cmax", task="task_scan", sel={"dose": 1}).get_data(experiment)
    assert middle.dims == ()


def test_an_unknown_index_raises_at_initialize() -> None:
    class Unknown(PKExperiment):
        def data(self) -> dict:
            return {"x": Data("nope", task="task_sim")}

    with pytest.raises(ValueError, match=r"'nope'.*observable.*task_sim.*model 'm'"):
        _runner(Unknown)


def test_a_parameter_of_an_observable_which_is_no_pk_raises() -> None:
    class NotPK(PKExperiment):
        def data(self) -> dict:
            return {"x": Data("cmax.value", task="task_sim")}

    with pytest.raises(ValueError, match=r"'cmax\.value'"):
        _runner(NotPK)


def test_the_key_of_an_observable_is_its_id() -> None:
    class WrongKey(PKExperiment):
        def observables(self) -> dict[str, Observable]:
            return {"peak": Formula("cmax", "max([C])")}

    with pytest.raises(ValueError, match=r"key 'peak'.*id 'cmax'"):
        _runner(WrongKey)


def test_an_observable_id_must_not_clash_with_a_key() -> None:
    class Clash(PKExperiment):
        def observables(self) -> dict[str, Observable]:
            return {"task_sim": Formula("task_sim", "max([C])")}

    with pytest.raises(ValueError, match="Duplicate key 'task_sim'"):
        _runner(Clash)


def test_without_reduced_selections_every_selection_is_kept() -> None:
    runner = _runner(PKExperiment)
    full = runner.experiments["PKExperiment"]
    full.run(runner.simulator, reduced_selections=False)
    names = set(full.results["task_scan"].ds.data_vars)
    assert {"cmax", "pk.cmax", "[C]"} <= names and len(names) > 3


class ValuesOnly(PKExperiment):
    """A task whose data read values per simulation only, with time registered."""

    def data(self) -> dict:
        self.add_selections_data(["time"], task_ids=["task_scan"])
        return {"cmax_scan": Data("cmax", task="task_scan")}


def test_a_task_of_values_per_simulation_runs_with_time_registered() -> None:
    runner = _runner(ValuesOnly)
    experiment = runner.experiments["ValuesOnly"]
    experiment.run(runner.simulator)
    result = experiment.results["task_scan"]
    assert set(result.ds.data_vars) == {"cmax"} and "time" not in result.ds.dims


class FigureData(PKExperiment):
    """A figure reads a selection which data() does not register."""

    def data(self) -> dict:
        return {}

    def figures(self) -> dict:
        figure = Figure(experiment=self, sid="fig", num_rows=1, num_cols=1)
        plot = figure.create_plots(xaxis=Axis("time"), yaxis=Axis("C"))[0]
        plot.curve(x=Data("time", task="task_sim"), y=Data("[C]", task="task_sim"))
        return {"fig": figure}


def test_the_data_of_figures_count() -> None:
    runner = _runner(FigureData)
    experiment = runner.experiments["FigureData"]
    experiment.run(runner.simulator)
    assert "[C]" in experiment.results["task_sim"].ds.data_vars


def test_the_json_has_the_observables(experiment: SimulationExperiment) -> None:
    d = json.loads(experiment.to_json())
    assert set(d["observables"]) == {"pk", "cmax", "unused"}
    assert d["observables"]["pk"]["type"]
    assert d["tasks"]["task_scan"]["observables"] == ["pk", "cmax"]
    assert d["tasks"]["task_sim"]["observables"] == []


class SharedTarget(PKExperiment):
    """A plain task reads PODOSE, the scan over PODOSE without observables."""

    def data(self) -> dict:
        return {
            "plain": Data("PODOSE", task="task_sim"),
            "scan_time": Data("time", task="task_scan"),
            "scan_c": Data("[C]", task="task_scan"),
            "scan_dose": Data("PODOSE", task="task_scan"),
        }


@pytest.mark.parametrize("reduced", [True, False])
def test_a_changed_target_is_a_coordinate_of_its_own_scan(reduced: bool) -> None:
    runner = _runner(SharedTarget)
    experiment = runner.experiments["SharedTarget"]
    experiment.run(runner.simulator, reduced_selections=reduced)
    scan = experiment.results["task_scan"]
    assert scan["PODOSE"].dims == ("dose",)
    assert experiment.results["task_sim"]["PODOSE"].dims == ("time",)
