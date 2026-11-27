"""A Data resolves to a labelled array with the dimensions of its source."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sbmlsim.data import ROW, Data, DataSet, to_quantity
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from sbmlsim.units import DimensionalityError


class ArrayExperiment(SimulationExperiment):
    """A scan of the initial amount of X on a grid, a ragged scan and a dataset."""

    def datasets(self) -> dict:
        df = pd.DataFrame(
            {"group": ["a", "b", "b"], "time": [0.0, 5.0, 10.0], "X": [1.0, 4.0, 2.0]}
        )
        return {
            "ds": DataSet.from_df(
                df, udict={"time": "second", "X": "dimensionless"}, ureg=self.ureg
            )
        }

    def models(self) -> dict:
        return {"m": AbstractModel(source=REPRESSILATOR_SBML)}

    def simulations(self) -> dict:
        dim = Dimension(
            "d", values={"X": np.array([1.0, 2.0, 30.0])}, labels=["lo", "mid", "hi"]
        )
        return {
            "sim_grid": Scan(Simulation(end=10, steps=10), [dim]),
            "sim_ragged": Scan(Simulation(end=10), [dim]),
        }

    def tasks(self) -> dict:
        return {
            "grid": Task(model="m", simulation="sim_grid"),
            "ragged": Task(model="m", simulation="sim_ragged"),
        }

    def data(self) -> dict:
        self.add_selections_data(["time", "[X]", "[Y]"])
        return {}


@pytest.fixture(scope="module")
def experiment() -> SimulationExperiment:
    runner = ExperimentRunner(
        experiment_classes=[ArrayExperiment],
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
        on_error="raise",
    )
    experiment = runner.experiments["ArrayExperiment"]
    experiment.run(runner.simulator)
    return experiment


def test_task_data_keeps_the_dimensions_and_coordinates(
    experiment: SimulationExperiment,
) -> None:
    y = Data("[Y]", task="grid").get_data(experiment)
    assert isinstance(y, xr.DataArray)
    assert y.dims == ("d", "time") and y.shape == (3, 11)
    assert y.name == "grid__conc__Y"
    assert y["d"].values.tolist() == ["lo", "mid", "hi"]
    np.testing.assert_allclose(y["d.X"].values, [1.0, 2.0, 30.0])
    assert "X" not in y.coords
    assert y.attrs["units"] is not None
    x = Data("d.X", task="grid").get_data(experiment)
    assert x.dims == ("d",)
    np.testing.assert_allclose(x.values, [1.0, 2.0, 30.0])
    with pytest.raises(KeyError, match=r"Data\('d\.X'\)"):
        Data("X", task="grid").get_data(experiment)
    labels = Data("d", task="grid").get_data(experiment)
    assert (
        labels.values.tolist() == ["lo", "mid", "hi"] and labels.attrs["units"] is None
    )


def test_sel_by_label(experiment: SimulationExperiment) -> None:
    one = Data("[Y]", task="grid", sel={"d": "mid"}).get_data(experiment)
    assert one.dims == ("time",)
    two = Data("[Y]", task="grid", sel={"d": ["lo", "hi"]}).get_data(experiment)
    assert two.dims == ("d", "time") and two["d"].values.tolist() == ["lo", "hi"]
    with pytest.raises(ValueError, match=r"dimension 'nope'.*'time'"):
        Data("[Y]", task="grid", sel={"nope": 0}).get_data(experiment)
    with pytest.raises(ValueError, match=r"\['zz'\].*'d'.*\['lo', 'mid', 'hi'\]"):
        Data("[Y]", task="grid", sel={"d": "zz"}).get_data(experiment)


def test_sel_skips_a_dimension_of_the_scan_the_data_has_not(
    experiment: SimulationExperiment,
) -> None:
    time = Data("time", task="grid", sel={"d": "mid"}).get_data(experiment)
    assert time.dims == ("time",)
    np.testing.assert_allclose(time.values, np.linspace(0, 10, 11))


def test_sel_of_a_ragged_result_picks_one_simulation(
    experiment: SimulationExperiment,
) -> None:
    time = Data("time", task="ragged", sel={"d": "lo"}).get_data(experiment)
    y = Data("[Y]", task="ragged", sel={"d": "lo"}).get_data(experiment)
    assert time.dims == y.dims == ("_point",)
    np.testing.assert_array_equal(np.isnan(time.values), np.isnan(y.values))
    native = Simulator().simulate(
        experiment._models["m"], Simulation(end=10, preinit_changes={"X": 1.0})
    )
    keep = ~np.isnan(time.values)
    np.testing.assert_allclose(time.values[keep], native.time)
    np.testing.assert_allclose(y.values[keep], native["[Y]"])


def test_dataset_data_is_over_rows(experiment: SimulationExperiment) -> None:
    x = Data("X", dataset="ds").get_data(experiment)
    assert x.dims == (ROW,) and x[ROW].values.tolist() == [0, 1, 2]
    assert x.attrs["units"] == "dimensionless"
    b = Data("X", dataset="ds", sel={"group": "b"}).get_data(experiment)
    assert b.values.tolist() == [4.0, 2.0] and b[ROW].values.tolist() == [1, 2]
    a = Data("X", dataset="ds", sel={"group": ["a"]}).get_data(experiment)
    assert a.dims == (ROW,) and a.values.tolist() == [1.0]
    with pytest.raises(ValueError, match=r"column 'nope'"):
        Data("X", dataset="ds", sel={"nope": 1}).get_data(experiment)
    with pytest.raises(ValueError, match="no row"):
        Data("X", dataset="ds", sel={"group": "c"}).get_data(experiment)


def test_a_reduction_of_a_dataset_runs_over_its_rows(
    experiment: SimulationExperiment,
) -> None:
    ratio = Data(
        "ratio",
        function="x / max(d)",
        variables={
            "x": Data("[X]", task="grid", sel={"d": "lo"}),
            "d": Data("X", dataset="ds"),
        },
    ).get_data(experiment)
    x = Data("[X]", task="grid", sel={"d": "lo"}).get_data(experiment)
    assert ratio.dims == ("time",)
    np.testing.assert_allclose(ratio.values, x.values / 4.0)


def test_to_units_keeps_the_dimensions(experiment: SimulationExperiment) -> None:
    # the time of the repressilator is in minutes, its unit definition `time`
    time = Data("time", task="grid").get_data(experiment, to_units="second")
    assert time.dims == ("time",) and time.attrs["units"] == "second"
    np.testing.assert_allclose(time.values, np.linspace(0, 10, 11) * 60.0)
    with pytest.raises(DimensionalityError):
        Data("time", task="grid").get_data(experiment, to_units="mol")


def test_to_quantity(experiment: SimulationExperiment) -> None:
    time = Data("time", task="grid").get_data(experiment)
    quantity = to_quantity(time, experiment.ureg)
    assert quantity.to("minute").magnitude.tolist() == pytest.approx(
        np.linspace(0, 10, 11).tolist()
    )
    with pytest.raises(ValueError, match="labels"):
        to_quantity(Data("d", task="grid").get_data(experiment), experiment.ureg)


def test_the_selection_is_serialized() -> None:
    assert Data("[Y]", task="grid", sel={"d": "lo"}).to_dict()["sel"] == {"d": "lo"}
    assert Data("[Y]", task="grid").to_dict()["sel"] is None


def test_a_function_of_a_scan_and_a_reference_has_the_time_last(
    experiment: SimulationExperiment,
) -> None:
    ratio = Data(
        "ratio",
        function="b_scan / a_ref",
        variables={
            "a_ref": Data("[Y]", task="grid", sel={"d": "lo"}),
            "b_scan": Data("[Y]", task="grid"),
        },
    ).get_data(experiment)
    assert ratio.dims == ("d", "time")


@pytest.mark.parametrize("task", ["grid", "ragged"])
@pytest.mark.parametrize(
    ("index", "to_units"), [("[Y]", None), ("time", None), ("time", "second")]
)
@pytest.mark.parametrize("sel", [{"d": "mid"}, {"d": "hi"}, {"d": ["lo", "hi"]}])
def test_a_selection_of_the_whole_array_is_the_data_of_the_selection(
    experiment: SimulationExperiment,
    task: str,
    index: str,
    to_units: str | None,
    sel: dict,
) -> None:
    """The lines of the draws of a figure select from one array, read once.

    Every line read, renamed and converted the whole array of its task, which
    was most of the time of the figures of an uncertainty run (sbmlsim#664);
    the selection of the array gives the same data, its unit included.
    """
    arrays: dict = {}
    for _ in range(2):
        data = Data(index, task=task, sel=sel)
        selected = data.get_data(experiment, to_units, arrays=arrays)
        plain = Data(index, task=task, sel=sel)
        expected = plain.get_data(experiment, to_units)
        xr.testing.assert_identical(selected, expected)
        assert data.unit == plain.unit
    assert list(arrays) == [(task, index, to_units)]
    # the whole array keeps its name, also when a selection is all of it
    ((whole, _),) = arrays.values()
    assert whole.name == Data(index, task=task).sid


def test_a_label_is_selected_by_its_position(experiment: SimulationExperiment) -> None:
    """A single label is selected by `isel` of its position, as `sel` selects it."""
    from sbmlsim.data import _select

    result = experiment.results["grid"]
    array = result["[Y]"]
    data = Data("[Y]", task="grid")
    for label in ("lo", "mid", "hi"):
        xr.testing.assert_identical(
            _select(array, {"d": label}, data, result.ds), array.sel(d=label)
        )
    xr.testing.assert_identical(
        _select(array, {"d": ["hi", "lo"]}, data, result.ds),
        array.sel(d=["hi", "lo"]),
    )
