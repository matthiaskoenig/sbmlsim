"""Test the result of a timecourse simulation."""

import numpy as np
import pytest

from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.result import TimecourseResult
from sbmlsim.result.timecourse import plot_times
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulator import Simulator


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


def test_simulator_returns_arrays() -> None:
    """A simulation is the array of its selections, with no DataFrame."""
    simulator = Simulator(n_workers=1)
    model = simulator.load(REPRESSILATOR_SBML)
    model.set_selections(["time", "[X]", "Y"])
    simulation = Simulation(
        start=-5, end=20, changes=[Change(10, {"Y": 5.0})], times=range(21)
    )
    result = simulator.simulate(model, simulation)

    assert isinstance(result, TimecourseResult)
    assert result.columns == ("time", "[X]", "Y")
    # the output are the times, the time of the change twice: the states
    # before and after it
    assert len(result) == 22
    np.testing.assert_allclose(result.time, np.insert(np.arange(21), 10, 10))
    assert result["Y"][10] != 5.0
    assert result["Y"][11] == 5.0


def test_few_time_points_are_kept() -> None:
    times = np.array([[0.0, 1.0, 2.0, np.nan], [0.0, 1.5, 2.0, 2.0]])
    np.testing.assert_array_equal(plot_times(times), [0.0, 1.0, 1.5, 2.0])


def test_the_time_points_of_a_figure_follow_the_steps() -> None:
    """Dense steps after a dose give dense time points, a few hundred in all."""
    rng = np.random.default_rng(1)
    # 50 simulations: dense steps in the first hour, sparse to 100 hr
    steps = [
        np.concatenate(
            [[0.0], np.sort(rng.uniform(0, 1, 400)), np.linspace(1, 100, 40)]
        )
        for _ in range(50)
    ]
    grid = plot_times(np.array(steps), n=200, keep=[24.0])
    assert grid.size <= 201
    assert grid[0] == 0.0 and grid[-1] == 100.0
    assert np.all(np.diff(grid) > 0)
    assert 24.0 in grid
    # most of the points in the first hour, every phase covered
    assert np.mean(grid <= 1.0) > 0.5
    assert np.max(np.diff(grid)) <= 100 / (200 // 4 - 1) + 1e-9


def test_a_figure_needs_two_time_points() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        plot_times(np.arange(10.0), n=1)
