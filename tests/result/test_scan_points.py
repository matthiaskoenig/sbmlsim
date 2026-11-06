"""The functions of a point of a scan result, on a decay `A -> ` with the rate `k`."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.result import ScanResult
from sbmlsim.result.scan import (
    end_values,
    point_band,
    point_timecourse,
    scan_dimensions,
    single_dimension,
    start_values,
    state_after_changes,
)
from sbmlsim.result.timecourse import PLOT_POINTS
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml

#: A in the compartment Vc decays with the rate k, in mmole, liter and minute
TOY = """
model toy
  unit substance = 1e-3 mole
  unit extent = 1e-3 mole
  unit time_unit = 60 second
  unit volume = litre
  unit mM = 1e-3 mole / litre
  unit per_min = 1 / 60 second
  compartment Vc = 1 volume
  species A in Vc = 1 mM
  k = 0.2 per_min
  DECAY: A -> ; k*A*Vc
end
"""

DECAY = Simulation(end=10, steps=50)
#: a second dose of A at 5 min: the output has the states before and after it
REDOSE = Simulation(end=10, steps=50, changes=[Change([5.0], {"[A]": Q(1.0, "mM")})])
RATES = Dimension("dim_k", values={"k": Q([0.1, 0.2], "1/min")})
#: the draws of the rate, the dimension a band reduces
DRAWS = Dimension("draws", values={"k": Q([0.1, 0.15, 0.2, 0.25, 0.3], "1/min")})


#: the file of the model, written by the fixture
TOY_PATH: dict[str, Path] = {}


@pytest.fixture(scope="module", autouse=True)
def _toy(tmp_path_factory: pytest.TempPathFactory) -> None:
    path = tmp_path_factory.mktemp("toy") / "toy.xml"
    path.write_text(sbml(TOY), encoding="utf-8")
    TOY_PATH["path"] = path


def run(simulation: Simulation, dimensions: Sequence[Dimension] = ()) -> ScanResult:
    return Simulator(n_workers=1).run(
        TOY_PATH["path"], Scan(simulation, list(dimensions))
    )


def test_the_dimension_of_a_scan() -> None:
    assert single_dimension(run(DECAY, [RATES])) == "dim_k"


def test_a_simulation_has_no_dimension() -> None:
    with pytest.raises(ValueError, match="one dimension"):
        single_dimension(run(DECAY))


def test_the_timecourse_of_a_point_without_changes() -> None:
    res = run(DECAY, [RATES])
    time, values = point_timecourse(res, "[A]", dim_k=1)
    assert np.array_equal(time, res.ds["time"].values)
    assert np.array_equal(values, res["[A]"].isel(dim_k=1).values)


@pytest.mark.parametrize("name", ["res", "key"])
def test_a_dimension_can_have_the_name_of_an_argument(name: str) -> None:
    res = run(DECAY, [Dimension(name, values={"k": Q([0.1, 0.2], "1/min")})])
    time, values = point_timecourse(res, "[A]", **{name: 1})
    assert np.array_equal(time, res.ds["time"].values)
    assert np.array_equal(values, res["[A]"].isel({name: 1}).values)


def test_the_state_before_a_change_is_left_out() -> None:
    res = run(REDOSE, [RATES])
    assert res.ds.sizes["time"] == 52
    time, values = point_timecourse(res, "[A]", dim_k=0)
    assert time.size == 51
    assert np.all(np.diff(time) > 0)
    assert values[time == 5.0] == pytest.approx([1.0])


def test_a_simulation_needs_no_position() -> None:
    time, _ = point_timecourse(run(REDOSE), "[A]")
    assert time.size == 51


def test_a_ragged_result_loses_its_padding() -> None:
    lengths = Dimension(
        "dim_sim",
        simulations={
            "short": Simulation(end=5, steps=10),
            "long": Simulation(end=10, steps=20),
        },
    )
    res = run(DECAY, [lengths])
    assert res.ragged
    time, values = point_timecourse(res, "[A]", dim_sim=0)
    assert time.size == values.size == 11
    assert time[-1] == 5.0


def test_every_dimension_needs_a_position() -> None:
    res = run(DECAY, [RATES])
    with pytest.raises(ValueError, match="dim_k"):
        point_timecourse(res, "[A]")
    with pytest.raises(ValueError, match="dim_x"):
        point_timecourse(res, "[A]", dim_k=0, dim_x=0)


def test_a_variable_which_is_no_timecourse() -> None:
    with pytest.raises(ValueError, match="no timecourse"):
        point_timecourse(run(DECAY, [RATES]), "missing", dim_k=0)


def test_the_state_after_a_change_has_every_time_once() -> None:
    res = run(REDOSE)
    assert res.ds.sizes["time"] == 52
    after = state_after_changes(res)
    time = after.ds["time"].values
    assert time.size == 51
    assert np.all(np.diff(time) > 0)
    assert after["[A]"].values[time == 5.0] == pytest.approx([1.0])


def test_the_state_after_a_change_is_the_state_of_the_point_timecourse() -> None:
    res = run(REDOSE, [RATES])
    after = state_after_changes(res)
    for k in range(2):
        time, values = point_timecourse(res, "[A]", dim_k=k)
        assert np.array_equal(after.ds["time"].values, time)
        assert np.allclose(after["[A]"].isel(dim_k=k).values, values)


def test_a_result_without_a_change_is_unchanged() -> None:
    res = run(DECAY, [RATES])
    after = state_after_changes(res)
    assert np.array_equal(after.ds["time"].values, res.ds["time"].values)
    assert np.allclose(after["[A]"].values, res["[A]"].values)


def test_the_state_after_a_change_of_a_ragged_result() -> None:
    lengths = Dimension(
        "dim_sim",
        simulations={
            "short": Simulation(end=5, steps=10),
            "long": Simulation(
                end=10, steps=20, changes=[Change([3.0], {"[A]": Q(1.0, "mM")})]
            ),
        },
    )
    res = run(DECAY, [lengths])
    assert res.ragged
    after = state_after_changes(res)
    for k in range(2):
        time, values = point_timecourse(res, "[A]", dim_sim=k)
        got = after.isel(dim_sim=k)
        finite = np.isfinite(got.ds["time"].values)
        assert np.allclose(got.ds["time"].values[finite][: time.size], time)
        assert np.allclose(got["[A]"].values[finite][: time.size], values)


def test_the_state_after_a_change_of_a_scan_of_two_dimensions() -> None:
    doses = Dimension("dim_dose", values={"[A]": Q([1.0, 2.0, 3.0], "mM")})
    res = run(REDOSE, [RATES, doses])
    after = state_after_changes(res)
    assert after["[A]"].shape[:2] == (2, 3)
    for i in range(2):
        for j in range(3):
            time, values = point_timecourse(res, "[A]", dim_k=i, dim_dose=j)
            assert np.array_equal(after.ds["time"].values, time)
            assert np.allclose(after["[A]"].isel(dim_k=i, dim_dose=j).values, values)


def test_the_start_and_end_values_on_a_grid() -> None:
    res = run(DECAY, [RATES])
    assert not res.ragged
    start = start_values(res, "[A]")
    end = end_values(res, "[A]")
    assert start.dims == end.dims == ("dim_k",)
    np.testing.assert_array_equal(start.values, res["[A]"].isel(time=0).values)
    np.testing.assert_array_equal(end.values, res["[A]"].isel(time=-1).values)


def test_the_start_and_end_values_of_the_steps_of_the_integrator() -> None:
    """A scan without steps is ragged, the end is the last time of every point."""
    res = run(Simulation(end=10), [RATES])
    assert res.ragged
    start = start_values(res, "[A]")
    end = end_values(res, "[A]")
    assert start.dims == end.dims == ("dim_k",)
    for k in range(2):
        _, values = point_timecourse(res, "[A]", dim_k=k)
        assert start.values[k] == values[0]
        assert end.values[k] == values[-1]
    # the faster decay ends lower
    assert end.values[1] < end.values[0]


def test_the_end_values_of_simulations_of_different_lengths() -> None:
    lengths = Dimension(
        "dim_sim",
        simulations={
            "short": Simulation(end=5, steps=10),
            "long": Simulation(end=10, steps=20),
        },
    )
    res = run(DECAY, [lengths])
    end = end_values(res, "[A]")
    for k in range(2):
        _, values = point_timecourse(res, "[A]", dim_sim=k)
        assert end.values[k] == values[-1]
    assert np.all(np.isfinite(end.values))


def test_the_start_and_end_values_need_a_timecourse() -> None:
    res = run(DECAY, [RATES])
    with pytest.raises(ValueError, match="no timecourse"):
        start_values(res, "missing")
    with pytest.raises(ValueError, match="no timecourse"):
        end_values(res, "missing")


def test_a_ragged_result_of_many_simulations_is_on_the_time_points_of_a_figure() -> (
    None
):
    # the steps of the integrator of 60 rates: every simulation steps at other
    # times, their union has more than 1000 time points, the time points of a
    # figure 500 and the time of the change
    steps = Simulation(end=10, changes=[Change([5.0], {"[A]": Q(1.0, "mM")})])
    rates = Dimension("dim_k", values={"k": Q(np.linspace(0.1, 3.0, 60), "1/min")})
    res = run(steps, [rates])
    assert res.ragged
    assert np.unique(res.time_points()).size > 1000
    after = state_after_changes(res)
    time = after.ds["time"].values
    assert time.size <= 501
    assert 5.0 in time
    # the state after the change at its time, the change sets [A] to 1 mM
    assert np.allclose(after["[A]"].isel(time=time_index(after, 5.0)).values, 1.0)
    # the end, as the union of the steps gives it
    union = res.interpolate(np.unique(res.time_points()))
    assert np.allclose(
        after["[A]"].isel(time=time_index(after, 10.0)).values,
        union["[A]"].isel(time=time_index(union, 10.0)).values,
    )


def time_index(res: ScanResult, time: float) -> int:
    """The position of a time in the time points of a result."""
    return int(np.flatnonzero(res.ds["time"].values == time)[0])


def test_the_dimensions_without_the_draws() -> None:
    volumes = Dimension("dim_volume", values={"Vc": np.array([1.0, 2.0])})
    res = run(DECAY, [volumes, DRAWS])
    assert scan_dimensions(res, "draws") == ["dim_volume"]
    assert scan_dimensions(res) == ["dim_volume", "draws"]


def test_the_band_of_a_point() -> None:
    volumes = Dimension("dim_volume", values={"Vc": np.array([1.0, 2.0])})
    res = run(Simulation(end=10, steps=10), [volumes, DRAWS])
    time, lower, median, upper = point_band(res, "[A]", across="draws", dim_volume=0)
    assert time.shape == lower.shape == median.shape == upper.shape == (11,)
    assert np.all(lower <= median) and np.all(median <= upper)
    assert lower[-1] < upper[-1]
    # the median of the five draws is the draw of the median rate
    _, values = point_timecourse(res, "[A]", dim_volume=0, draws=2)
    np.testing.assert_allclose(median, values)
    with pytest.raises(ValueError, match="no dimension"):
        point_band(res, "[A]", across="draw", dim_volume=0)
    with pytest.raises(ValueError, match="reduced, not a point"):
        point_band(res, "[A]", across="draws", dim_volume=0, draws=0)


def test_the_band_of_the_steps_of_the_integrator() -> None:
    """Draws with their own time points are on the time points of a figure."""
    res = run(Simulation(end=10), [DRAWS])
    assert res.ragged
    time, lower, median, upper = point_band(res, "[A]", across="draws")
    assert np.all(np.diff(time) > 0)
    assert time[0] == 0.0 and time[-1] == 10.0
    steps = [point_timecourse(res, "[A]", draws=k)[0] for k in range(5)]
    assert time.size <= max(PLOT_POINTS, np.unique(np.concatenate(steps)).size)
    assert np.all(lower <= median) and np.all(median <= upper)
