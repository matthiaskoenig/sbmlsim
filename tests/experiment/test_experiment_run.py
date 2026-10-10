"""Tests of running a simulation experiment."""

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from sbmlsim.data import Data, DataSet, to_quantity
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.experiment.runner import model_key
from sbmlsim.fit import FitData, FitMapping
from sbmlsim.model import AbstractModel
from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from sbmlsim.plot import Axis, Curve, Figure, Plot, SubPlot
from sbmlsim.plot.padding import first_curve, without_padding
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.result import ScanResult
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from sbmlsim.units import Quantity, _create_registry


class FitMappingExperiment(SimulationExperiment):
    """An experiment whose fit mapping observes what `data()` does not."""

    def models(self) -> dict:
        return {"m": AbstractModel(source=REPRESSILATOR_SBML)}

    def simulations(self) -> dict:
        return {"sim": Simulation(end=20, steps=20)}

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
        simulator=Simulator(),
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

    variables = set(experiment.results["task"].ds.data_vars)
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
        simulator=Simulator(),
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
    assert np.asarray(experiment.results["task"].ds["[X]"]).size > 0


def test_the_figures_are_created_for_the_output(tmp_path: Path) -> None:
    """A run with an output path draws and saves them."""
    runner = _runner(FitMappingExperiment)
    experiment = runner.experiments["FitMappingExperiment"]
    experiment.run(runner.simulator, output_path=tmp_path)

    assert (tmp_path / f"{experiment.sid}.json").exists()


class ScanExperiment(SimulationExperiment):
    """An experiment of a scan of the initial amount of X."""

    def models(self) -> dict:
        return {"m": AbstractModel(source=REPRESSILATOR_SBML)}

    def simulations(self) -> dict:
        return {
            "scan": Scan(
                Simulation(end=10, steps=10),
                [Dimension("d", values={"X": np.array([1.0, 2.0, 3.0])})],
            )
        }

    def tasks(self) -> dict:
        return {"task": Task(model="m", simulation="scan")}

    def data(self) -> dict:
        return {"Y": Data(index="[Y]", task="task")}


def test_the_data_of_a_scan_has_the_time_last() -> None:
    """A Data of a task is the variable of the result, the time last.

    A changed target which is no selection is a coordinate over its
    dimension.
    """
    runner = _runner(ScanExperiment)
    experiment = runner.experiments["ScanExperiment"]
    experiment.run(runner.simulator)

    y = Data("[Y]", task="task").get_data(experiment)
    assert y.shape == (3, 11)
    assert y.attrs["units"] == "dimensionless"
    time = Data("time", task="task").get_data(experiment, to_units="second")
    np.testing.assert_allclose(time.values, np.linspace(0, 10, 11))
    x = Data("X", task="task").get_data(experiment)
    np.testing.assert_allclose(x.values, [1.0, 2.0, 3.0])


class RegistryExperiment(FitMappingExperiment):
    """An experiment whose data combines a task and a dataset."""

    def datasets(self) -> dict:
        df = pd.DataFrame({"time": [0.0, 10.0, 20.0], "X": [1.0, 4.0, 2.0]})
        return {
            "ds": DataSet.from_df(
                df, udict={"time": "second", "X": "dimensionless"}, ureg=self.ureg
            )
        }


def test_the_data_of_a_task_is_in_the_registry_of_the_experiment() -> None:
    """A runner with its own registry combines task data with dataset data."""
    # a registry with the definitions of the package, but another one
    ureg = _create_registry()
    runner = ExperimentRunner(
        experiment_classes=[RegistryExperiment],
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
        ureg=ureg,
    )
    experiment = runner.experiments["RegistryExperiment"]
    experiment.run(runner.simulator)

    ratio = Data(
        "ratio",
        function="x / max(d)",
        variables={"x": Data("[X]", task="task"), "d": Data("X", dataset="ds")},
    ).get_data(experiment)
    x = Data("[X]", task="task").get_data(experiment)
    assert experiment.ureg is ureg
    assert to_quantity(x, experiment.ureg)._REGISTRY is ureg
    np.testing.assert_allclose(ratio.values, x.values / 4.0)


class RaggedScanExperiment(ScanExperiment):
    """The scan with the steps of the integrator, a ragged result."""

    def simulations(self) -> dict:
        return {
            "scan": Scan(
                Simulation(end=10),
                [Dimension("d", values={"X": np.array([1.0, 2.0, 30.0])})],
            )
        }


def test_the_data_of_a_ragged_scan_has_its_points_last() -> None:
    """A Data of a ragged result is over `(*dims, _point)`, padded with NaN.

    A curve draws the first point of the scan without its padding.
    """
    runner = _runner(RaggedScanExperiment)
    experiment = runner.experiments["RaggedScanExperiment"]
    experiment.run(runner.simulator)
    result = experiment.results["task"]
    assert result.ragged

    y = Data("[Y]", task="task").get_data(experiment)
    time = Data("time", task="task").get_data(experiment)
    assert y.values.shape == time.values.shape == (3, result.ds.sizes["_point"])
    padded = np.isnan(time.values)
    assert padded.any()
    np.testing.assert_array_equal(np.isnan(y.values), padded)

    x, curve = without_padding(first_curve(time.values), first_curve(y.values))
    native = Simulator().simulate(
        experiment._models["m"], Simulation(end=10, preinit_changes={"X": 1.0})
    )
    np.testing.assert_allclose(x, native.time)
    np.testing.assert_allclose(curve, native["[Y]"])


def test_the_data_of_a_variable_which_was_not_selected_raises() -> None:
    """A Data whose variable is not in the result names the variables."""
    runner = _runner(ScanExperiment)
    experiment = runner.experiments["ScanExperiment"]
    experiment.run(runner.simulator)

    with pytest.raises(KeyError, match=r"'\[Z\]' is not in the result.*\[Y\]"):
        Data("[Z]", task="task").get_data(experiment)


def test_an_experiment_without_a_simulator_raises() -> None:
    """A run needs a simulator."""
    runner = _runner(ScanExperiment)
    experiment = runner.experiments["ScanExperiment"]

    with pytest.raises(ValueError, match="has no simulator"):
        experiment.run(None)


def test_the_results_are_written_as_netcdf(tmp_path: Path) -> None:
    """A run which saves its results writes every task as netCDF, no TSV."""
    runner = _runner(FitMappingExperiment)
    experiment = runner.experiments["FitMappingExperiment"]
    experiment.run(runner.simulator, output_path=tmp_path, save_results=True)
    path = tmp_path / "FitMappingExperiment_task.nc"
    assert ScanResult.from_netcdf(path)["[X]"].size > 0
    assert not list(tmp_path.glob("FitMappingExperiment_task.tsv"))


# ---------------------------------------------------------------------------
# the format decides the backend
# ---------------------------------------------------------------------------
class FigureExperiment(FitMappingExperiment):
    """An experiment with one figure of the figure model."""

    def figures(self) -> dict:
        plot = Plot(
            sid="p",
            xaxis=Axis("time", unit="second"),
            yaxis=Axis("X", unit="dimensionless"),
        )
        plot.curves.append(
            Curve(
                x=Data(index="time", task="task"),
                y=Data(index="[X]", task="task"),
                name="X",
            )
        )
        return {
            "fig": Figure(
                experiment=self,
                sid="fig",
                num_rows=1,
                num_cols=1,
                subplots=[SubPlot(plot=plot, row=1, col=1)],
            )
        }


def _run_with(tmp_path: Path, formats: list[str]) -> Path:
    """Run the figure experiment with the given output formats."""
    runner = _runner(FigureExperiment)
    experiment = runner.experiments["FigureExperiment"]
    experiment.run(runner.simulator, output_path=tmp_path, figure_formats=formats)
    return tmp_path


def test_a_static_format_is_drawn_by_matplotlib(tmp_path: Path) -> None:
    """`svg` writes an image and no interactive page."""
    out = _run_with(tmp_path, ["svg"])
    assert (out / "FigureExperiment_fig.svg").exists()
    assert not (out / "FigureExperiment_fig.html").exists()
    assert not (out / "plotly.min.js").exists()


def test_the_interactive_format_is_drawn_by_plotly(tmp_path: Path) -> None:
    """`html` writes an interactive page and no image.

    The javascript is written next to the page, so the report needs no
    network, and matplotlib is not asked for a figure at all.
    """
    pytest.importorskip("plotly")
    runner = _runner(FigureExperiment)
    experiment = runner.experiments["FigureExperiment"]
    experiment.run(runner.simulator, output_path=tmp_path, figure_formats=["html"])

    page = tmp_path / "FigureExperiment_fig.html"
    assert page.exists()
    assert (tmp_path / "plotly.min.js").exists()
    assert not (tmp_path / "FigureExperiment_fig.svg").exists()
    # matplotlib was not used, so nothing was rendered and closed
    assert experiment._mpl_figures == {}

    html = page.read_text(encoding="utf-8")
    assert "plotly-graph-div" in html
    # the page loads its javascript from next to it and not from the network
    assert 'src="plotly.min.js"' in html
    assert "https://" not in html.split("<script")[1][:400]


def test_both_formats_are_written(tmp_path: Path) -> None:
    """An experiment asks for the image and the page in one run."""
    pytest.importorskip("plotly")
    out = _run_with(tmp_path, ["svg", "html"])
    assert (out / "FigureExperiment_fig.svg").exists()
    assert (out / "FigureExperiment_fig.html").exists()


def test_the_interactive_figures_need_plotly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Without plotly the run says so and writes no page, it does not raise."""
    monkeypatch.setitem(sys.modules, "sbmlsim.plot.serialization_plotly", None)
    runner = _runner(FigureExperiment)
    experiment = runner.experiments["FigureExperiment"]

    with caplog.at_level(logging.ERROR):
        experiment.run(runner.simulator, output_path=tmp_path, figure_formats=["html"])

    assert not (tmp_path / "FigureExperiment_fig.html").exists()
    assert any("plotly" in record.getMessage() for record in caplog.records)
