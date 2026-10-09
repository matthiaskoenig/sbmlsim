"""The comparison of the results of two simulators."""

import warnings

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from sbmlsim.comparison.diff import DataSetsComparison, to_dataframe
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator

SIMULATOR = Simulator(n_workers=1)


@pytest.fixture
def model() -> RoadrunnerSBMLModel:
    """Get the repressilator with a few selections."""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    model.set_selections(["time", "PX", "[X]"])
    return model


def _frame(model: RoadrunnerSBMLModel, simulation: Simulation) -> pd.DataFrame:
    """Get the table of a simulation, one column per selection."""
    result = SIMULATOR.simulate(model, simulation)
    return pd.DataFrame(result.values, columns=list(result.columns))


def test_data_frames_are_compared() -> None:
    """A value outside of the tolerances is a difference."""
    df = pd.DataFrame({"time": [0.0, 1.0], "X": [1.0, 2.0]})
    other = df.assign(X=[1.0, 2.1])
    assert DataSetsComparison({"a": df, "b": df.copy()}).is_equal()
    comparison = DataSetsComparison({"a": df, "b": other})
    assert not comparison.is_equal()
    assert comparison.columns == ["time", "X"]


def test_a_dataset_is_compared_with_a_data_frame(model: RoadrunnerSBMLModel) -> None:
    """The dataset of a result on a grid has the time as a column."""
    simulation = Simulation(end=10, steps=10)
    res = SIMULATOR.run(model, simulation)
    comparison = DataSetsComparison({"ds": res.ds, "df": _frame(model, simulation)})
    assert comparison.columns == ["time", "PX", "[X]"]
    assert comparison.is_equal()
    np.testing.assert_array_equal(comparison.dfs[0]["time"], np.arange(11.0))


def test_a_ragged_dataset_is_compared_without_its_points(
    model: RoadrunnerSBMLModel,
) -> None:
    """The dimension `_point` of the steps of the integrator is no column."""
    simulation = Simulation(end=10)
    res = SIMULATOR.run(model, simulation)
    assert res.ragged
    comparison = DataSetsComparison({"ds": res.ds, "df": _frame(model, simulation)})
    assert comparison.columns == ["time", "PX", "[X]"]
    assert comparison.is_equal()


def test_a_point_of_a_scan_is_compared(model: RoadrunnerSBMLModel) -> None:
    """A dataset of a scan is compared one point at a time, without its labels."""
    simulation = Simulation(end=10, steps=10)
    res = SIMULATOR.run(
        model, Scan(simulation, [Dimension("dim_n", values={"n": [2.0, 3.0]})])
    )
    with pytest.raises(ValueError, match="dim_n"):
        DataSetsComparison({"ds": res.ds, "df": _frame(model, simulation)})

    for k, n in enumerate([2.0, 3.0]):
        point = res.ds.isel(dim_n=k)
        df = _frame(model, simulation.with_values({"n": n}))
        comparison = DataSetsComparison({"ds": point, "df": df})
        assert comparison.columns == ["time", "PX", "[X]"]
        assert comparison.is_equal()
    # the points differ, the comparison of one with the other is no equality
    first, second = res.ds.isel(dim_n=0), res.ds.isel(dim_n=1)
    assert not DataSetsComparison({"a": first, "b": second}).is_equal()


def test_a_point_of_a_ragged_scan_is_compared_without_its_padding(
    model: RoadrunnerSBMLModel,
) -> None:
    """The point with fewer steps of the integrator has no rows of padding."""
    simulation = Simulation(end=10)
    res = SIMULATOR.run(
        model, Scan(simulation, [Dimension("dim_n", values={"n": [2.0, 6.0]})])
    )
    assert res.ragged
    lengths = []
    for k, n in enumerate([2.0, 6.0]):
        df = _frame(model, simulation.with_values({"n": n}))
        table = to_dataframe(res.ds.isel(dim_n=k), "point")
        assert not table["time"].isna().any()
        lengths.append(len(table))
        comparison = DataSetsComparison({"ds": res.ds.isel(dim_n=k), "df": df})
        assert comparison.is_equal()
    # one of the points was padded
    assert min(lengths) < res.ds.sizes["_point"]


def test_the_frames_of_the_caller_are_not_changed() -> None:
    """The selections and factors of a comparison work on copies."""
    a = pd.DataFrame({"time": [0.0, 1.0], "x": [1.0, 2.0]})
    b = pd.DataFrame({"time": [0.0, 1.0], "y": [2.0, 4.0]})
    frames = {"a": a, "b": b}
    copies = {key: df.copy() for key, df in frames.items()}
    comparison = DataSetsComparison(
        frames,
        selections={"a": ["time", "x"], "b": ["time", "y"]},
        factors={"a": [1.0, 2.0], "b": [1.0, 1.0]},
    )
    assert comparison.is_equal()
    assert list(frames) == ["a", "b"]
    for key, df in frames.items():
        pd.testing.assert_frame_equal(df, copies[key])


@pytest.mark.parametrize("other", [[1.0, 2.0, 3.0], [1.0, 2.5, 3.0]])
def test_the_report_warns_nothing(other: list[float]) -> None:
    """The figure of the report is drawn without warnings, also without differences."""
    df = pd.DataFrame(
        {"time": [0.0, 1.0, 2.0], "X": [1.0, 2.0, 3.0], "Y": [0.0, 1.0, 0.5]}
    )
    comparison = DataSetsComparison({"a": df, "b": df.assign(X=other)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        figure = comparison.report()
    plt.close(figure)
