"""The comparison of the results of two simulators."""

import numpy as np
import pandas as pd
import pytest

from sbmlsim.comparison.diff import DataSetsComparison
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

    point = res.ds.isel(dim_n=0)
    comparison = DataSetsComparison({"ds": point, "df": _frame(model, simulation)})
    assert comparison.columns == ["time", "PX", "[X]"]
    assert comparison.is_equal()
