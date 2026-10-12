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


def test_every_result_is_compared_with_the_reference() -> None:
    """A later result cannot hide behind an earlier matching result."""
    reference = pd.DataFrame({"time": [0.0, 1.0], "X": [1.0, 2.0]})
    comparison = DataSetsComparison(
        {
            "reference": reference,
            "matching": reference.copy(),
            "different": reference.assign(X=[99.0, 99.0]),
        }
    )
    assert not comparison.is_equal()
    np.testing.assert_array_equal(comparison.diff_abs.X, [98.0, 97.0])
    np.testing.assert_array_equal(comparison.diff_tol_bool.X, [True, True])
    assert "FALSE" in comparison.report_str()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plt.close(comparison.plot_diff())


def test_multiple_results_show_the_largest_difference_per_value() -> None:
    """Different candidates can have the largest deviation at different rows."""
    reference = pd.DataFrame({"time": [0.0, 1.0], "X": [10.0, 10.0]})
    comparison = DataSetsComparison(
        {
            "reference": reference,
            "first": reference.assign(X=[8.0, 11.0]),
            "second": reference.assign(X=[9.0, 14.0]),
        }
    )
    np.testing.assert_array_equal(comparison.diff.X, [2.0, -4.0])
    np.testing.assert_array_equal(comparison.diff_abs.X, [2.0, 4.0])
    np.testing.assert_allclose(comparison.diff_rel.X, [4.0 / 18.0, 8.0 / 24.0])
    assert not comparison.is_equal()


def test_row_labels_do_not_change_a_result_comparison() -> None:
    """The numeric rows are compared in order, independent of pandas labels."""
    reference = pd.DataFrame({"time": [0.0, 1.0], "X": [1.0, 2.0]})
    other = reference.set_axis([10, 11])
    assert DataSetsComparison({"reference": reference, "other": other}).is_equal()
    assert other.index.tolist() == [10, 11]


def test_unsigned_results_do_not_wrap_their_differences() -> None:
    """Subtraction compares values rather than wrapping unsigned integers."""
    reference = pd.DataFrame({"time": [0.0], "X": np.array([0], dtype=np.uint8)})
    candidate = reference.assign(X=np.array([255], dtype=np.uint8))
    comparison = DataSetsComparison({"reference": reference, "candidate": candidate})
    assert not comparison.is_equal()
    np.testing.assert_array_equal(comparison.diff.X, [-255.0])


def test_multiple_equal_results_agree() -> None:
    """Checking more than two matching results preserves equality."""
    reference = pd.DataFrame({"time": [0.0, 1.0], "X": [1.0, 2.0]})
    assert DataSetsComparison({str(k): reference.copy() for k in range(4)}).is_equal()


def test_an_invalid_later_result_is_not_hidden() -> None:
    """A finite result cannot hide a candidate's undefined values."""
    reference = pd.DataFrame({"time": [0.0, 1.0], "X": [1.0, 2.0]})
    comparison = DataSetsComparison(
        {
            "reference": reference,
            "invalid": reference.assign(X=[np.nan, 2.0]),
            "matching": reference.copy(),
        }
    )
    assert not comparison.is_equal()
    assert comparison.diff_tol_bool.X.tolist() == [True, False]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("reference_is_invalid", [False, True])
def test_nonfinite_values_never_prove_equality(
    value: float, reference_is_invalid: bool
) -> None:
    """An infinite tolerance cannot turn an invalid reference into agreement."""
    first = pd.DataFrame({"time": [0.0], "X": [value]})
    second = first.assign(X=[1.0])
    if not reference_is_invalid:
        first, second = second, first
    comparison = DataSetsComparison({"reference": first, "candidate": second})
    assert not comparison.is_equal()
    assert comparison.diff_tol_bool.X.tolist() == [True]


def test_results_without_time_can_be_compared() -> None:
    """Scalar tables need no time column when it is explicitly optional."""
    reference = pd.DataFrame({"X": [1.0, 2.0]})
    comparison = DataSetsComparison(
        {"reference": reference, "other": reference.copy()}, time_column=False
    )
    assert comparison.is_equal()
    assert comparison.columns == ["X"]
    plt.close(comparison.report())


def test_a_column_of_durations_is_not_compared() -> None:
    """A timedelta column is no number, the other columns are compared."""
    reference = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "X": [1.0, 2.0],
            "duration": pd.to_timedelta([1, 2], unit="s"),
        }
    )
    comparison = DataSetsComparison(
        {"reference": reference, "candidate": reference.copy()}
    )
    assert comparison.columns == ["time", "X"]
    assert comparison.is_equal()


@pytest.mark.parametrize("rows", [0, 1])
def test_a_comparison_requires_two_results(rows: int) -> None:
    """A missing candidate gives a useful error before differences are computed."""
    reference = pd.DataFrame({"time": [0.0] * rows, "X": [1.0] * rows})
    with pytest.raises(ValueError, match="at least two"):
        DataSetsComparison({"reference": reference})


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


def test_integer_columns_can_be_scaled_by_fractional_factors() -> None:
    """A fractional conversion changes the dtype without changing the input."""
    a = pd.DataFrame({"time": [0, 1], "X": [1, 2]})
    b = pd.DataFrame({"time": [0, 1], "Y": [2, 4]})
    comparison = DataSetsComparison(
        {"a": a, "b": b},
        selections={"a": ["time", "X"], "b": ["time", "Y"]},
        factors={"a": [1.0, 0.5], "b": [1.0, 0.25]},
    )
    assert comparison.is_equal()
    assert a.X.tolist() == [1, 2]
    assert a.X.dtype == np.dtype("int64")


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
