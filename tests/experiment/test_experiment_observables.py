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
from sbmlsim.resources import REPRESSILATOR_SBML
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
            "dose_scan": Data("dose.PODOSE", task="task_scan"),
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
    np.testing.assert_allclose(cmax["dose.PODOSE"].values, [50.0, 100.0, 200.0])
    assert "PODOSE" not in cmax.coords
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


def test_a_bare_pk_id_raises_at_initialize() -> None:
    class BarePK(PKExperiment):
        def data(self) -> dict:
            return {"x": Data("pk", task="task_sim")}

    with pytest.raises(ValueError, match=r"'pk'.*pk\.<parameter>.*pk\.cmax"):
        _runner(BarePK)


def test_an_unknown_simulation_of_a_task_raises_at_initialize() -> None:
    class UnknownSimulation(PKExperiment):
        def tasks(self) -> dict:
            return {"task_sim": Task(model="m", simulation="nope")}

        def data(self) -> dict:
            return {}

    with pytest.raises(
        ValueError, match=r"task 'task_sim'.*simulation 'nope'.*\['scan', 'sim'\]"
    ):
        _runner(UnknownSimulation)


def test_an_unknown_model_of_a_task_raises_at_initialize() -> None:
    class UnknownModel(PKExperiment):
        def tasks(self) -> dict:
            return {"task_sim": Task(model="nope", simulation="sim")}

        def data(self) -> dict:
            return {}

    with pytest.raises(ValueError, match=r"task 'task_sim'.*model 'nope'.*\['m'\]"):
        _runner(UnknownModel)


def test_an_unknown_dimension_of_sel_raises_at_initialize() -> None:
    class DimensionTypo(PKExperiment):
        def data(self) -> dict:
            return {"x": Data("cmax", task="task_scan", sel={"dsoe": 1})}

    with pytest.raises(ValueError, match=r"dimension 'dsoe'.*\['dose'"):
        _runner(DimensionTypo)


def test_an_unknown_label_of_sel_raises_at_initialize() -> None:
    class LabelTypo(PKExperiment):
        def figures(self) -> dict:
            figure = Figure(experiment=self, sid="fig", num_rows=1, num_cols=1)
            plot = figure.create_plots(xaxis=Axis("time"), yaxis=Axis("C"))[0]
            plot.curve(
                x=Data("time", task="task_scan", sel={"dose": 7}),
                y=Data("[C]", task="task_scan", sel={"dose": 7}),
            )
            return {"fig": figure}

    with pytest.raises(ValueError, match=r"\[7\] of the dimension 'dose'.*\[0, 1, 2\]"):
        _runner(LabelTypo)


def test_a_selection_of_the_time_is_not_checked_at_initialize() -> None:
    class TimeSel(PKExperiment):
        def data(self) -> dict:
            return {"x": Data("[C]", task="task_scan", sel={"time": 0.0, "dose": 1})}

    experiment = _runner(TimeSel).experiments["TimeSel"]
    assert "x" in experiment._data


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
    with pytest.raises(KeyError, match=r"no time.*values per simulation"):
        Data("time", task="task_scan").get_data(experiment)


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
    """HCTZ-like: PODOSE is registered for every task of a shared model."""

    def data(self) -> dict:
        self.add_selections_data(["time", "PODOSE"])
        return {"scan_dose": Data("dose.PODOSE", task="task_scan")}


@pytest.mark.parametrize("reduced", [True, False])
def test_the_plain_name_of_a_changed_target_is_its_timecourse(reduced: bool) -> None:
    runner = _runner(SharedTarget)
    experiment = runner.experiments["SharedTarget"]
    experiment.run(runner.simulator, reduced_selections=reduced)
    scan = experiment.results["task_scan"]
    assert scan["PODOSE"].dims == ("dose", "time")
    assert scan["dose.PODOSE"].dims == ("dose",)
    assert experiment.results["task_sim"]["PODOSE"].dims == ("time",)
    values = Data("dose.PODOSE", task="task_scan").get_data(experiment)
    assert values.dims == ("dose",) and values.attrs["units"]
    np.testing.assert_allclose(values.values, [50.0, 100.0, 200.0])
    curve = Data("PODOSE", task="task_scan").get_data(experiment)
    assert curve.dims == ("dose", "time")
    # the dose is a timecourse which falls from the value the scan sets
    np.testing.assert_allclose(curve.values[:, 0], [50.0, 100.0, 200.0])
    assert np.all(curve.values[:, -1] < curve.values[:, 0])


class SpeciesScan(PKExperiment):
    """A scan over the initial concentration of a species."""

    def simulations(self) -> dict:
        simulation = Simulation(end=48, steps=96)
        init = Dimension("init", values={"[C]": Q([1.0, 2.0, 4.0], "mg/litre")})
        return {"sim": simulation, "scan": Scan(simulation, [init])}

    def data(self) -> dict:
        self.add_selections_data(["time", "[C]"], task_ids=["task_scan"])
        return {"values": Data("init.[C]", task="task_scan")}


def test_a_scanned_species_is_a_timecourse_and_its_values_are_qualified() -> None:
    runner = _runner(SpeciesScan)
    experiment = runner.experiments["SpeciesScan"]
    experiment.run(runner.simulator)
    timecourse = Data("[C]", task="task_scan").get_data(experiment)
    assert timecourse.dims == ("init", "time")
    assert np.all(timecourse.values[:, -1] < timecourse.values[:, 0])
    values = Data("init.[C]", task="task_scan").get_data(experiment)
    assert values.dims == ("init",)
    np.testing.assert_allclose(values.values, [1.0, 2.0, 4.0])


def test_a_target_the_dimension_does_not_change_raises() -> None:
    runner = _runner(SpeciesScan)
    experiment = runner.experiments["SpeciesScan"]
    experiment.run(runner.simulator)
    with pytest.raises(KeyError, match="'init'"):
        Data("init.nope", task="task_scan").get_data(experiment)


def test_a_plain_name_which_is_only_the_values_of_a_dimension_raises(
    experiment: SimulationExperiment,
) -> None:
    with pytest.raises(
        KeyError, match=r"'PODOSE'.*Data\('dose\.PODOSE'\).*timecourse.*regist"
    ):
        Data("PODOSE", task="task_scan").get_data(experiment)
    values = Data("dose.PODOSE", task="task_scan").get_data(experiment)
    assert values.dims == ("dose",) and values.name == "task_scan__dose__PODOSE"
    assert list(values.coords) == ["dose", "dose.PODOSE"]


class TwoScanTasks(PKExperiment):
    """Two tasks of one scan, one of which reads the scanned dose as a selection."""

    def tasks(self) -> dict:
        return {
            "task_scan": Task(model="m", simulation="scan"),
            "task_scan2": Task(model="m", simulation="scan"),
        }

    def data(self) -> dict:
        return {
            "cmax_scan": Data("cmax", task="task_scan"),
            "cmax_scan2": Data("cmax", task="task_scan2"),
            "dose_scan2": Data("PODOSE", task="task_scan2"),
        }


@pytest.mark.parametrize("reduced", [True, False])
def test_every_array_of_a_task_names_the_values_of_a_dimension_alike(
    reduced: bool,
) -> None:
    runner = _runner(TwoScanTasks)
    experiment = runner.experiments["TwoScanTasks"]
    experiment.run(runner.simulator, reduced_selections=reduced)
    for task in ("task_scan", "task_scan2"):
        cmax = Data("cmax", task=task).get_data(experiment)
        assert "dose.PODOSE" in cmax.coords and "PODOSE" not in cmax.coords
        np.testing.assert_allclose(cmax["dose.PODOSE"].values, [50.0, 100.0, 200.0])
        values = Data("dose.PODOSE", task=task).get_data(experiment)
        np.testing.assert_allclose(values.values, [50.0, 100.0, 200.0])
    curve = Data("PODOSE", task="task_scan2").get_data(experiment)
    assert curve.dims == ("dose", "time") and "dose.PODOSE" in curve.coords


class AddedParameterScan(PKExperiment):
    """A scan of a parameter added to the model, which is no default selection."""

    def models(self) -> dict:
        return {"m": AbstractModel(source=sbml_pk(), parameters={"kadd": 1.0})}

    def simulations(self) -> dict:
        simulation = Simulation(end=48, steps=96)
        k = Dimension("k", values={"kadd": np.array([1.0, 2.0, 4.0])})
        return {"sim": simulation, "scan": Scan(simulation, [k])}

    def data(self) -> dict:
        return {"kadd_scan": Data("kadd", task="task_scan")}


@pytest.mark.parametrize("reduced", [True, False])
def test_a_task_selects_what_its_data_read_also_without_reduced_selections(
    reduced: bool,
) -> None:
    runner = _runner(AddedParameterScan)
    experiment = runner.experiments["AddedParameterScan"]
    model = experiment._models["m"]
    model.set_selections(None)
    assert "kadd" not in (model.selections or [])
    experiment.run(runner.simulator, reduced_selections=reduced)
    timecourse = Data("kadd", task="task_scan").get_data(experiment)
    assert timecourse.dims == ("k", "time")
    np.testing.assert_allclose(timecourse.values[:, -1], [1.0, 2.0, 4.0])
    values = Data("k.kadd", task="task_scan").get_data(experiment)
    np.testing.assert_allclose(values.values, [1.0, 2.0, 4.0])


class DocsObservableExperiment(SimulationExperiment):
    """The experiment of the observables block of docs/data.md."""

    def models(self) -> dict:
        return {"model": REPRESSILATOR_SBML}

    def simulations(self) -> dict:
        return {
            "scan": Scan(
                Simulation(end=100, steps=100),
                [Dimension("x0", values={"X": np.array([10.0, 20.0, 40.0])})],
            )
        }

    def observables(self) -> dict[str, Observable]:
        return {"xmax": Formula("xmax", "max([X])")}

    def tasks(self) -> dict:
        return {"task_scan": Task(model="model", simulation="scan")}

    def data(self) -> dict:
        return {
            "data_xmax": Data("xmax", task="task_scan"),
            "x": Data("[X]", task="task_scan"),
        }


@pytest.mark.parametrize("reduced", [True, False])
def test_the_values_of_the_docs_block_have_one_name(reduced: bool) -> None:
    runner = _runner(DocsObservableExperiment)
    experiment = runner.experiments["DocsObservableExperiment"]
    experiment.run(runner.simulator, reduced_selections=reduced)
    xmax = Data("xmax", task="task_scan").get_data(experiment)
    np.testing.assert_allclose(xmax["x0.X"].values, [10.0, 20.0, 40.0])
    x0 = Data("x0.X", task="task_scan").get_data(experiment)
    np.testing.assert_allclose(x0.values, [10.0, 20.0, 40.0])
    x = Data("[X]", task="task_scan", sel={"x0": 2}).get_data(experiment)
    assert float(x.max()) == float(xmax.sel(x0=2))
