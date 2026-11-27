"""The functions of a point of a scan result, on a decay `A -> ` with the rate `k`."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.result import ScanResult
from sbmlsim.result.scan import (
    end_values,
    nan_quantiles,
    point_band,
    point_timecourse,
    scan_dimensions,
    single_dimension,
    start_values,
    state_after_changes,
)
from sbmlsim.result.timecourse import PLOT_POINTS, without_states_before
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


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The SBML file of the decay."""
    path = tmp_path_factory.mktemp("toy") / "toy.xml"
    path.write_text(sbml(TOY), encoding="utf-8")
    return path


def run(
    toy: Path, simulation: Simulation, dimensions: Sequence[Dimension] = ()
) -> ScanResult:
    return Simulator(n_workers=1).run(toy, Scan(simulation, list(dimensions)))


def test_the_dimension_of_a_scan(toy: Path) -> None:
    assert single_dimension(run(toy, DECAY, [RATES])) == "dim_k"


def test_a_simulation_has_no_dimension(toy: Path) -> None:
    with pytest.raises(ValueError, match="one dimension"):
        single_dimension(run(toy, DECAY))


def test_the_timecourse_of_a_point_without_changes(toy: Path) -> None:
    res = run(toy, DECAY, [RATES])
    time, values = point_timecourse(res, "[A]", dim_k=1)
    assert np.array_equal(time, res.ds["time"].values)
    assert np.array_equal(values, res["[A]"].isel(dim_k=1).values)


@pytest.mark.parametrize("name", ["res", "key"])
def test_a_dimension_can_have_the_name_of_an_argument(toy: Path, name: str) -> None:
    res = run(toy, DECAY, [Dimension(name, values={"k": Q([0.1, 0.2], "1/min")})])
    time, values = point_timecourse(res, "[A]", **{name: 1})
    assert np.array_equal(time, res.ds["time"].values)
    assert np.array_equal(values, res["[A]"].isel({name: 1}).values)


def test_the_state_before_a_change_is_left_out(toy: Path) -> None:
    res = run(toy, REDOSE, [RATES])
    assert res.ds.sizes["time"] == 52
    time, values = point_timecourse(res, "[A]", dim_k=0)
    assert time.size == 51
    assert np.all(np.diff(time) > 0)
    assert values[time == 5.0] == pytest.approx([1.0])


def test_a_simulation_needs_no_position(toy: Path) -> None:
    time, _ = point_timecourse(run(toy, REDOSE), "[A]")
    assert time.size == 51


def test_a_ragged_result_loses_its_padding(toy: Path) -> None:
    lengths = Dimension(
        "dim_sim",
        simulations={
            "short": Simulation(end=5, steps=10),
            "long": Simulation(end=10, steps=20),
        },
    )
    res = run(toy, DECAY, [lengths])
    assert res.ragged
    time, values = point_timecourse(res, "[A]", dim_sim=0)
    assert time.size == values.size == 11
    assert time[-1] == 5.0


def test_every_dimension_needs_a_position(toy: Path) -> None:
    res = run(toy, DECAY, [RATES])
    with pytest.raises(ValueError, match="dim_k"):
        point_timecourse(res, "[A]")
    with pytest.raises(ValueError, match="dim_x"):
        point_timecourse(res, "[A]", dim_k=0, dim_x=0)


def test_a_variable_which_is_no_timecourse(toy: Path) -> None:
    with pytest.raises(ValueError, match="no timecourse"):
        point_timecourse(run(toy, DECAY, [RATES]), "missing", dim_k=0)


def test_the_state_after_a_change_has_every_time_once(toy: Path) -> None:
    res = run(toy, REDOSE)
    assert res.ds.sizes["time"] == 52
    after = state_after_changes(res)
    time = after.ds["time"].values
    assert time.size == 51
    assert np.all(np.diff(time) > 0)
    assert after["[A]"].values[time == 5.0] == pytest.approx([1.0])


def test_the_state_after_a_change_is_the_state_of_the_point_timecourse(
    toy: Path,
) -> None:
    res = run(toy, REDOSE, [RATES])
    after = state_after_changes(res)
    for k in range(2):
        time, values = point_timecourse(res, "[A]", dim_k=k)
        assert np.array_equal(after.ds["time"].values, time)
        assert np.allclose(after["[A]"].isel(dim_k=k).values, values)


def test_a_result_without_a_change_is_unchanged(toy: Path) -> None:
    res = run(toy, DECAY, [RATES])
    after = state_after_changes(res)
    assert np.array_equal(after.ds["time"].values, res.ds["time"].values)
    assert np.allclose(after["[A]"].values, res["[A]"].values)


def test_the_state_after_a_change_of_a_ragged_result(toy: Path) -> None:
    lengths = Dimension(
        "dim_sim",
        simulations={
            "short": Simulation(end=5, steps=10),
            "long": Simulation(
                end=10, steps=20, changes=[Change([3.0], {"[A]": Q(1.0, "mM")})]
            ),
        },
    )
    res = run(toy, DECAY, [lengths])
    assert res.ragged
    after = state_after_changes(res)
    for k in range(2):
        time, values = point_timecourse(res, "[A]", dim_sim=k)
        got = after.isel(dim_sim=k)
        finite = np.isfinite(got.ds["time"].values)
        assert np.allclose(got.ds["time"].values[finite][: time.size], time)
        assert np.allclose(got["[A]"].values[finite][: time.size], values)


def test_the_state_after_a_change_of_a_scan_of_two_dimensions(toy: Path) -> None:
    doses = Dimension("dim_dose", values={"[A]": Q([1.0, 2.0, 3.0], "mM")})
    res = run(toy, REDOSE, [RATES, doses])
    after = state_after_changes(res)
    assert after["[A]"].shape[:2] == (2, 3)
    for i in range(2):
        for j in range(3):
            time, values = point_timecourse(res, "[A]", dim_k=i, dim_dose=j)
            assert np.array_equal(after.ds["time"].values, time)
            assert np.allclose(after["[A]"].isel(dim_k=i, dim_dose=j).values, values)


def test_the_start_and_end_values_on_a_grid(toy: Path) -> None:
    res = run(toy, DECAY, [RATES])
    assert not res.ragged
    start = start_values(res, "[A]")
    end = end_values(res, "[A]")
    assert start.dims == end.dims == ("dim_k",)
    np.testing.assert_array_equal(start.values, res["[A]"].isel(time=0).values)
    np.testing.assert_array_equal(end.values, res["[A]"].isel(time=-1).values)


def test_the_start_and_end_values_of_the_steps_of_the_integrator(toy: Path) -> None:
    """A scan without steps is ragged, the end is the last time of every point."""
    res = run(toy, Simulation(end=10), [RATES])
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


def test_the_end_values_of_simulations_of_different_lengths(toy: Path) -> None:
    lengths = Dimension(
        "dim_sim",
        simulations={
            "short": Simulation(end=5, steps=10),
            "long": Simulation(end=10, steps=20),
        },
    )
    res = run(toy, DECAY, [lengths])
    end = end_values(res, "[A]")
    for k in range(2):
        _, values = point_timecourse(res, "[A]", dim_sim=k)
        assert end.values[k] == values[-1]
    assert np.all(np.isfinite(end.values))


def test_the_start_and_end_values_need_a_timecourse(toy: Path) -> None:
    res = run(toy, DECAY, [RATES])
    with pytest.raises(ValueError, match="no timecourse"):
        start_values(res, "missing")
    with pytest.raises(ValueError, match="no timecourse"):
        end_values(res, "missing")


def test_a_ragged_result_of_many_simulations_is_on_the_time_points_of_a_figure(
    toy: Path,
) -> None:
    # the steps of the integrator of 60 rates: every simulation steps at other
    # times, their union has more than 1000 time points, the time points of a
    # figure 500 and the time of the change
    steps = Simulation(end=10, changes=[Change([5.0], {"[A]": Q(1.0, "mM")})])
    rates = Dimension("dim_k", values={"k": Q(np.linspace(0.1, 3.0, 60), "1/min")})
    res = run(toy, steps, [rates])
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


def test_the_dimensions_without_the_draws(toy: Path) -> None:
    volumes = Dimension("dim_volume", values={"Vc": np.array([1.0, 2.0])})
    res = run(toy, DECAY, [volumes, DRAWS])
    assert scan_dimensions(res, "draws") == ["dim_volume"]
    assert scan_dimensions(res) == ["dim_volume", "draws"]


def test_the_band_of_a_point(toy: Path) -> None:
    volumes = Dimension("dim_volume", values={"Vc": np.array([1.0, 2.0])})
    res = run(toy, Simulation(end=10, steps=10), [volumes, DRAWS])
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


def test_the_band_of_the_steps_of_the_integrator(toy: Path) -> None:
    """Draws with their own time points are on the time points of a figure."""
    res = run(toy, Simulation(end=10), [DRAWS])
    assert res.ragged
    time, lower, median, upper = point_band(res, "[A]", across="draws")
    assert np.all(np.diff(time) > 0)
    assert time[0] == 0.0 and time[-1] == 10.0
    steps = [point_timecourse(res, "[A]", draws=k)[0] for k in range(5)]
    assert time.size <= max(PLOT_POINTS, np.unique(np.concatenate(steps)).size)
    assert np.all(lower <= median) and np.all(median <= upper)


@pytest.mark.parametrize(
    "simulation", [REDOSE, Simulation(end=10, changes=REDOSE.changes)]
)
def test_the_band_of_a_point_is_the_one_of_the_timecourses_of_its_draws(
    toy: Path, simulation: Simulation
) -> None:
    """The draws of a point are read at once, value for value as one by one.

    A band read the timecourse of every draw on its own, which was most of the
    time of the bands of the figures of a scan over draws (sbmlsim#664); the
    draws lie between two dimensions and have the states before the dose.
    """
    volumes = Dimension("dim_volume", values={"Vc": np.array([1.0, 2.0])})
    doses = Dimension("dim_a", values={"[A]": Q([1.0, 2.0], "mM")})
    res = run(toy, simulation, [volumes, DRAWS, doses])
    for point in ({"dim_volume": 1, "dim_a": 0}, {"dim_volume": 0, "dim_a": 1}):
        time, lower, median, upper = point_band(
            res, "[A]", across="draws", quantiles=(0.1, 0.9), **point
        )
        draws = [
            point_timecourse(res, "[A]", **point, draws=k)
            for k in range(res.ds.sizes["draws"])
        ]
        if not res.ragged:
            assert all(np.array_equal(t, time) for t, _ in draws)
            expected = nan_quantiles(
                np.vstack([values for _, values in draws]), [0.1, 0.5, 0.9]
            )
            np.testing.assert_array_equal(np.vstack([lower, median, upper]), expected)
        assert 5.0 in time.tolist()
        assert np.unique(time).size == time.size


def test_the_states_before_the_changes_of_rows_with_the_same_changes() -> None:
    """Rows of the same changes are checked at once, value for value as one by one."""
    rng = np.random.default_rng(1)
    time = np.tile(np.array([0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, np.nan]), (4, 1))
    time[1, -2:] = np.nan
    values = rng.random(time.shape)
    changes = np.array([2.0, 4.0])
    shared = without_states_before(time, values, [changes] * 4)
    # equal changes which are no one object
    copied = without_states_before(time, values, [changes.copy() for _ in range(4)])
    rows = [without_states_before(time[[r]], values[[r]], [changes]) for r in range(4)]
    for result in (shared, copied):
        for r, (row_time, row_values) in enumerate(rows):
            n = row_time.shape[1]
            np.testing.assert_array_equal(result[0][r, :n], row_time[0])
            np.testing.assert_array_equal(result[1][r, :n], row_values[0])
            assert np.isnan(result[0][r, n:]).all()
    # the states before the changes at 2 and 4 are left out, the padding too
    np.testing.assert_array_equal(shared[0][0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(shared[1][0], values[0, [0, 1, 3, 4, 6, 7]])
    # rows of other changes
    other = without_states_before(time[:2], values[:2], [changes, np.array([])])
    np.testing.assert_array_equal(other[0][1, :7], time[1, :7])


def test_nan_quantiles_are_the_ones_of_numpy_value_for_value() -> None:
    from sbmlsim.result.scan import nan_quantiles

    rng = np.random.default_rng(7)
    table = rng.lognormal(size=(100, 300))
    table[rng.random(table.shape) < 0.02] = np.nan
    table[:, 5] = np.nan
    qs = [0.05, 0.5, 0.95]
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        expected = np.nanquantile(table, qs, axis=0)
    np.testing.assert_array_equal(nan_quantiles(table, qs), expected)
    full = rng.lognormal(size=(50, 40))
    np.testing.assert_array_equal(
        nan_quantiles(full, qs), np.nanquantile(full, qs, axis=0)
    )
