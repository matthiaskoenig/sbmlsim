"""Test the result of a timecourse simulation and the results built from it."""

import numpy as np
import pandas as pd
import pytest

from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.result import TimecourseResult, XResult
from sbmlsim.simulation import Dimension, ScanSim, Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial


def _result(offset: float = 0.0) -> TimecourseResult:
    """Get a result with three time points and two variables."""
    return TimecourseResult(
        columns=("time", "[X]", "Y"),
        values=np.array(
            [
                [0.0 + offset, 1.0, 10.0],
                [1.0 + offset, 2.0, 20.0],
                [2.0 + offset, 3.0, 30.0],
            ]
        ),
    )


def test_column_by_name() -> None:
    """A column is the array of its values."""
    result = _result()
    np.testing.assert_array_equal(result["[X]"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.time, [0.0, 1.0, 2.0])
    assert len(result) == 3


def test_unknown_column() -> None:
    """A column which is not selected raises a KeyError."""
    with pytest.raises(KeyError):
        _result()["Z"]


def test_duplicate_column_is_the_first() -> None:
    """A name which appears twice gives the first column of that name."""
    result = TimecourseResult(
        columns=("time", "time"), values=np.array([[0.0, 5.0], [1.0, 6.0]])
    )
    np.testing.assert_array_equal(result["time"], [0.0, 1.0])


def test_values_must_match_the_columns() -> None:
    """The values have one column per name."""
    with pytest.raises(ValueError, match="columns"):
        TimecourseResult(columns=("time", "X"), values=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="2"):
        TimecourseResult(columns=("time",), values=np.zeros(3))


def test_from_timecourses_without_scan() -> None:
    """Several results without a scan are entries of the `_dfs` dimension."""
    xres = XResult.from_timecourses([_result(), _result(offset=0.0)])
    assert xres.xds["[X]"].dims == ("_point", "_dfs")
    assert xres.xds.sizes["_dfs"] == 2
    np.testing.assert_array_equal(xres.xds["time"].values[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(xres.xds["Y"].values[:, 1], [10.0, 20.0, 30.0])


def test_from_timecourses_without_time() -> None:
    """The time is the coordinate of the results, a result without it is refused."""
    result = TimecourseResult(columns=("X",), values=np.zeros((3, 1)))
    with pytest.raises(ValueError, match="time"):
        XResult.from_timecourses([result])


def test_from_timecourses_of_different_lengths() -> None:
    """Results of different lengths are padded, see `tests/result/test_xresult.py`."""
    short = TimecourseResult(columns=("time", "X"), values=np.zeros((2, 2)))
    long = TimecourseResult(columns=("time", "X"), values=np.zeros((3, 2)))
    xres = XResult.from_timecourses([long, short])
    assert np.isnan(xres.xds["X"].values[2, 1])


def test_from_dfs_is_from_timecourses() -> None:
    """DataFrames give the same result as the arrays they hold."""
    frames = [
        pd.DataFrame(_result().values, columns=list(_result().columns)),
        pd.DataFrame(_result().values * 2, columns=list(_result().columns)),
    ]
    expected = XResult.from_timecourses(
        [
            _result(),
            TimecourseResult(columns=_result().columns, values=_result().values * 2),
        ]
    )
    xres = XResult.from_dfs(dfs=frames)
    for key in expected.xds:
        np.testing.assert_array_equal(xres.xds[key].values, expected.xds[key].values)


def test_scan_places_every_simulation() -> None:
    """The result of a simulation of a scan is at the indices of its changes."""
    simulator = SimulatorSerial(REPRESSILATOR_SBML)
    scan = ScanSim(
        simulation=TimecourseSim(Timecourse(start=0, end=10, steps=10)),
        dimensions=[
            Dimension("dim_n", changes={"n": np.array([2.0, 3.0, 4.0])}),
            Dimension("dim_y", changes={"Y": np.array([10.0, 30.0])}),
        ],
    )
    xres = simulator.run_scan(scan)
    assert xres.xds["Y"].dims == ("_time", "dim_n", "dim_y")
    assert xres.xds["Y"].shape == (11, 3, 2)
    np.testing.assert_array_equal(xres.xds["n"].values[0], [[2, 2], [3, 3], [4, 4]])
    np.testing.assert_array_equal(xres.xds["Y"].values[0], [[10, 30]] * 3)


def test_simulator_returns_arrays() -> None:
    """A timecourse simulation is the array of its selections, with no DataFrame."""
    simulator = SimulatorSerial(REPRESSILATOR_SBML)
    simulator.set_timecourse_selections(["time", "[X]", "Y"])
    simulation = TimecourseSim(
        [
            Timecourse(start=0, end=5, steps=5, discard=True),
            Timecourse(start=0, end=10, steps=10),
            Timecourse(start=0, end=10, steps=10),
        ]
    )
    simulation.normalize(uinfo=simulator.uinfo)
    result = simulator._timecourses([simulation])[0]

    assert isinstance(result, TimecourseResult)
    assert result.columns == ("time", "[X]", "Y")
    # the pre-simulation is discarded and the times of the second timecourse
    # continue where the first one ended
    assert len(result) == 22
    np.testing.assert_allclose(result.time[:11], np.linspace(0, 10, 11))
    np.testing.assert_allclose(result.time[11:], np.linspace(10, 20, 11))


def test_simulator_all_discarded() -> None:
    """A simulation whose timecourses are all discarded has no result."""
    simulator = SimulatorSerial(REPRESSILATOR_SBML)
    simulation = TimecourseSim([Timecourse(start=0, end=5, steps=5, discard=True)])
    simulation.normalize(uinfo=simulator.uinfo)
    with pytest.raises(ValueError, match="discarded"):
        simulator._timecourses([simulation])
