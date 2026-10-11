"""The result of a scan wraps a dataset with the units of its variables."""

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from sbmlsim import Q
from sbmlsim.result import ScanResult, scan
from sbmlsim.result.scan import STATUS
from sbmlsim.result.timecourse import apply_weights, grid_weights, interpolate


def _grid() -> ScanResult:
    """Two points of a dimension `d` on the grid 0, 1, 2 with a coordinate `k1`."""
    ds = xr.Dataset(
        {"y": (("d", "time"), np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]))},
        coords={"d": [0, 1], "k1": ("d", [1.0, 2.0]), "time": [0.0, 1.0, 2.0]},
        attrs={"dims": ["d"], "units": {"y": "mM", "k1": "1/min", "time": "min"}},
    )
    return ScanResult(ds)


def _ragged() -> ScanResult:
    """Two points with their own time points, the second padded."""
    time = np.array([[0.0, 1.0, 3.0], [0.0, 2.0, np.nan]])
    y = np.array([[0.0, 1.0, 3.0], [0.0, 4.0, np.nan]])
    ds = xr.Dataset(
        {"time": (("d", "_point"), time), "y": (("d", "_point"), y)},
        coords={"d": [0, 1]},
        attrs={"dims": ["d"], "units": {"y": "mM", "time": "min"}},
    )
    return ScanResult(ds)


def test_interpolate_ignores_the_padding_and_the_steady_state() -> None:
    time = np.array([0.0, 1.0, np.inf, np.nan])
    values = np.array([[0.0, 1.0], [2.0, 3.0], [9.0, 9.0], [np.nan, np.nan]])
    out = interpolate(time, values, np.array([0.5, 1.0, 2.0]))
    np.testing.assert_allclose(out, [[1.0, 2.0], [2.0, 3.0], [np.nan, np.nan]])
    assert np.isnan(
        interpolate(np.array([np.nan]), np.array([np.nan]), np.array([0.0]))
    ).all()


def test_a_timecourse_without_time_points_is_interpolated_to_nan() -> None:
    grid = np.array([0.0, 1.0])
    assert np.isnan(interpolate(np.empty(0), np.empty(0), grid)).all()
    out = interpolate(np.empty(0), np.empty((0, 3)), grid)
    assert out.shape == (2, 3)
    assert np.isnan(out).all()


def test_a_ragged_result_without_time_points_is_interpolated_to_nan() -> None:
    """Every point of a ragged scan failed, so it has no time point at all."""
    ds = xr.Dataset(
        {
            "time": (("d", "_point"), np.empty((2, 0))),
            "y": (("d", "_point"), np.empty((2, 0))),
        },
        coords={"d": [0, 1]},
        attrs={"dims": ["d"], "units": {"y": "mM", "time": "min", "d": ""}},
    )
    out = ScanResult(ds).interpolate([0.0, 1.0, 2.0])
    assert out["y"].dims == ("d", "time")
    assert out["y"].shape == (2, 3)
    assert np.isnan(out["y"].values).all()


def test_a_summary_has_a_unit_of_every_variable_and_coordinate() -> None:
    ds = _grid().ds.copy()
    ds.attrs["units"] = {**ds.attrs["units"], "d": ""}
    for result in (ScanResult(ds), _ragged()):
        summary = result.summary(quantiles=[0.5])
        assert summary.units["statistic"] == ""
        missing = [str(v) for v in summary.ds.variables if v not in summary.units]
        assert not missing


def test_the_dimensions_and_the_layout() -> None:
    assert _grid().dims == ("d",)
    assert not _grid().ragged
    assert _ragged().ragged
    assert _grid().variables == ["y"]
    assert _ragged().variables == ["y"]
    assert "k1" in _grid()


def test_a_variable_is_a_quantity_with_its_unit() -> None:
    y = _grid().quantity("y")
    assert str(y.units) == "millimolar"
    assert y.magnitude.shape == (2, 3)
    k1 = _grid().quantity("k1").to("1/s").magnitude
    np.testing.assert_allclose(k1, [1 / 60, 2 / 60])


def test_an_unknown_variable_names_the_variables() -> None:
    with pytest.raises(KeyError, match="'y'"):
        _grid()["nope"]


def test_a_selection_keeps_the_units() -> None:
    one = _grid().sel(d=1)
    assert one.dims == ()
    assert one["y"].values.tolist() == [0.0, 2.0, 4.0]
    assert one.units["y"] == "mM"
    assert _grid().isel(d=0)["k1"].item() == 1.0


def test_the_time_points() -> None:
    assert _grid().time_points().tolist() == [0.0, 1.0, 2.0]
    assert _ragged().time_points().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_a_ragged_result_is_interpolated_onto_a_grid() -> None:
    grid = _ragged().interpolate([0.0, 2.0, 3.0])
    assert not grid.ragged
    assert grid["y"].dims == ("d", "time")
    np.testing.assert_allclose(grid["y"].values, [[0.0, 2.0, 3.0], [0.0, 4.0, np.nan]])
    assert grid["time"].values.tolist() == [0.0, 2.0, 3.0]
    assert grid.units == _ragged().units


def test_a_grid_is_interpolated_and_a_quantity_is_converted() -> None:
    grid = _grid().interpolate(Q([30, 90], "s"))
    np.testing.assert_allclose(grid["y"].values, [[0.5, 1.5], [1.0, 3.0]])
    assert grid["k1"].values.tolist() == [1.0, 2.0]


def test_the_summary_over_a_dimension() -> None:
    summary = _grid().summary("d", quantiles=[0.5])
    assert summary["y"].dims == ("statistic", "time")
    assert summary.ds["statistic"].values.tolist() == [
        "mean",
        "sd",
        "cv",
        "min",
        "max",
        "q0.5",
    ]
    y = summary["y"]
    np.testing.assert_allclose(y.sel(statistic="mean").values, [0.0, 1.5, 3.0])
    np.testing.assert_allclose(
        y.sel(statistic="sd").values, [0.0, np.sqrt(0.5), np.sqrt(2.0)]
    )
    np.testing.assert_allclose(y.sel(statistic="max").values, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(y.sel(statistic="q0.5").values, [0.0, 1.5, 3.0])
    assert summary.dims == ()
    assert summary.units["y"] == "mM"


def test_a_ragged_summary_is_on_the_union_of_the_time_points() -> None:
    summary = _ragged().summary(statistics=["mean"])
    assert summary["y"].dims == ("statistic", "time")
    assert summary.ds["time"].values.tolist() == [0.0, 1.0, 2.0, 3.0]
    np.testing.assert_allclose(
        summary["y"].sel(statistic="mean").values, [0.0, 1.5, 3.0, 3.0]
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dims": "x"}, "no dimensions"),
        ({"statistics": ["median"]}, "Unknown statistics"),
        ({"quantiles": [1.5]}, "quantile"),
    ],
)
def test_a_wrong_summary_is_an_error(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _grid().summary(**kwargs)


def test_the_netcdf_round_trip(tmp_path: Path) -> None:
    for result in (_grid(), _ragged()):
        path = tmp_path / "result.nc"
        result.to_netcdf(path)
        again = ScanResult.from_netcdf(path)
        xr.testing.assert_allclose(again.ds, result.ds)
        assert again.units == result.units
        assert again.dims == result.dims


def test_a_result_pickles() -> None:
    again = pickle.loads(pickle.dumps(_grid()))
    xr.testing.assert_identical(again.ds, _grid().ds)


def _full() -> ScanResult:
    """A result with a variable `[X]`, string and integer labels, a status and attrs."""
    ds = xr.Dataset(
        {
            "[X]": (("g", "n", "time"), np.arange(12.0).reshape(2, 2, 3)),
            "status": (("g", "n"), np.zeros((2, 2), dtype=np.int64)),
        },
        coords={
            "g": ["wt", "ko"],
            "n": np.array([1, 2**40], dtype=np.int64),
            "time": [0.0, 1.0, 2.0],
        },
        attrs={
            "dims": ["g", "n"],
            "units": {"[X]": "mM", "time": "min"},
            "scan": {"dimensions": [{"id": "g", "values": ["wt", "ko"]}]},
            "errors": {"0": "failed"},
        },
    )
    return ScanResult(ds)


def test_the_netcdf_round_trip_keeps_names_types_and_attrs(tmp_path: Path) -> None:
    result = _full()
    path = tmp_path / "full.nc"
    result.to_netcdf(path)
    again = ScanResult.from_netcdf(path)
    assert again.ds["status"].dtype == np.int64
    assert again.ds["n"].dtype == np.int64
    assert again.ds["n"].values.tolist() == [1, 2**40]
    assert again.ds["g"].values.tolist() == ["wt", "ko"]
    xr.testing.assert_identical(again.ds, result.ds)


def test_the_attrs_may_hold_numpy_values(tmp_path: Path) -> None:
    result = _grid()
    result.ds.attrs["n"] = np.int64(3)
    result.ds.attrs["w"] = np.array([1.0, 2.0])
    result.to_netcdf(tmp_path / "np.nc")
    again = ScanResult.from_netcdf(tmp_path / "np.nc")
    assert again.ds.attrs["n"] == 3
    assert again.ds.attrs["w"] == [1.0, 2.0]


def test_a_file_without_the_attrs_is_an_error(tmp_path: Path) -> None:
    xr.Dataset({"y": ("x", [1.0])}).to_netcdf(tmp_path / "plain.nc", engine="h5netcdf")
    with pytest.raises(ValueError, match="sbmlsim"):
        ScanResult.from_netcdf(tmp_path / "plain.nc")


def test_the_cv_is_dimensionless() -> None:
    summary = _grid().summary("d", statistics=["mean", "cv"])
    with pytest.raises(ValueError, match="select a statistic"):
        summary.quantity("y")
    cv = summary.sel(statistic="cv").quantity("y")
    assert str(cv.units) == "dimensionless"
    np.testing.assert_allclose(
        cv.magnitude, [np.nan, np.sqrt(0.5) / 1.5, np.sqrt(2.0) / 3.0]
    )
    mean = summary.sel(statistic="mean").quantity("y")
    assert str(mean.units) == "millimolar"
    both = _grid().summary("d", statistics=["mean", "min"]).quantity("y")
    assert str(both.units) == "millimolar"


def test_the_summary_values_and_nan_and_status() -> None:
    ds = _grid().ds.copy(deep=True)
    ds["status"] = ("d", np.array([0, 1]))
    ds["y"][1, 2] = np.nan
    summary = ScanResult(ds).summary("d", quantiles=[0.5])
    assert "status" not in summary.ds
    y = summary["y"]
    np.testing.assert_allclose(y.sel(statistic="min").values, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(y.sel(statistic="mean").values, [0.0, 1.5, 2.0])
    np.testing.assert_allclose(y.sel(statistic="q0.5").values, [0.0, 1.5, 2.0])
    np.testing.assert_allclose(y.sel(statistic="cv").values[1], np.sqrt(0.5) / 1.5)


def test_an_all_nan_cell_is_summarized_without_warnings() -> None:
    ds = _grid().ds.copy(deep=True)
    ds["y"][:, 1] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summary = ScanResult(ds).summary("d", quantiles=[0.5])
    assert np.isnan(summary["y"].values[:, 1]).all()
    assert not np.isnan(summary["y"].sel(statistic="mean").values[0])


def test_a_summary_does_not_share_the_attrs() -> None:
    result = _grid()
    result.summary("d").units["y"] = "changed"
    result.interpolate([0.0]).units["y"] = "changed"
    assert result.units["y"] == "mM"


def test_a_label_and_a_key_without_unit_are_no_quantities() -> None:
    with pytest.raises(ValueError, match="labels"):
        _full().quantity("g")
    with pytest.raises(ValueError, match="no unit"):
        _full().quantity("n")


def test_the_status_is_reserved() -> None:
    assert STATUS == "status"
    assert "status" not in _full().variables


def test_the_ragged_summary_has_more_than_the_mean() -> None:
    summary = _ragged().summary(statistics=["mean", "min", "max", "sd"])
    y = summary["y"]
    np.testing.assert_allclose(y.sel(statistic="min").values, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(y.sel(statistic="max").values, [0.0, 2.0, 4.0, 3.0])
    np.testing.assert_allclose(
        y.sel(statistic="sd").values[:3], [0.0, np.sqrt(0.5), np.sqrt(2.0)]
    )


def test_a_summary_takes_the_times() -> None:
    summary = _ragged().summary(statistics=["mean"], times=Q([0.0, 2.0], "min"))
    assert summary.ds["time"].values.tolist() == [0.0, 2.0]
    np.testing.assert_allclose(summary["y"].sel(statistic="mean").values, [0.0, 3.0])


def test_a_large_union_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scan, "MAX_UNION_ELEMENTS", 5)
    with pytest.raises(ValueError, match=r"times=.*interpolate\(times\).*time="):
        _ragged().summary()
    assert _ragged().summary(times=[0.0, 1.0]).ds["time"].size == 2


def test_the_interpolation_of_a_grid_matches_np_interp() -> None:
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 10, 7))
    v = rng.normal(size=(3, 4, 7))
    ds = xr.Dataset(
        {"y": (("a", "b", "time"), v)},
        coords={"a": [0, 1, 2], "b": [0, 1, 2, 3], "time": t},
        attrs={"dims": ["a", "b"], "units": {}},
    )
    grid = np.array([-1.0, t[0], 3.3, t[3], t[-1], 11.0])
    out = ScanResult(ds).interpolate(grid)["y"].values
    for i in range(3):
        for j in range(4):
            np.testing.assert_allclose(
                out[i, j], np.interp(grid, t, v[i, j], left=np.nan, right=np.nan)
            )


@pytest.mark.parametrize(
    "t",
    [
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 1.0, 1.0, 2.0],
        [0.0, 1.0, 2.0, 2.0],
        [0.0, 2.0, 2.0],
        # three changes at one time, at the start, in the middle and at the end
        [0.0, 0.0, 0.0, 1.0, 2.0],
        [0.0, 1.0, 1.0, 1.0, 2.0],
        [0.0, 1.0, 2.0, 2.0, 2.0],
    ],
)
def test_the_weights_take_the_value_after_a_change(t: list[float]) -> None:
    time = np.array(t)
    values = np.arange(1.0, time.size + 1) * 2.0
    grid = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    out = apply_weights(grid_weights(time, grid), values)
    np.testing.assert_allclose(
        out, np.interp(grid, time, values, left=np.nan, right=np.nan)
    )


def test_the_weights_take_the_value_after_an_event_of_the_model() -> None:
    """Two times closer than the tolerance are a jump, as the fit has it.

    The states before and after an event are a little apart here, at the
    event at 1 and at the end. A time at a jump sees the state after it:
    inside it, at one of its points and at most the tolerance before it,
    since CVODE locates an event a little after its time.
    """
    from sbmlsim.fit.optimization import _interpolate_after_jumps

    tolerance = 1e-9
    time = np.array([0.0, 1.0 - 1e-12, 1.0 + 1e-12, 1.5, 2.0 - 1e-12, 2.0])
    values = np.array([0.0, 1.0, 0.0, 0.5, 1.0, 0.0])
    grid = np.array(
        [0.0, 0.5, 1.0 - 1e-10, 1.0 - 1e-12, 1.0, 1.0 + 1e-12, 1.25, 2.0 - 1e-12, 2.0]
    )
    out = apply_weights(grid_weights(time, grid, tolerance), values)
    np.testing.assert_allclose(out, [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0])
    np.testing.assert_allclose(
        out, _interpolate_after_jumps(grid, time, values, tolerance)
    )
    np.testing.assert_allclose(
        interpolate(time, values[:, None], grid, tolerance)[:, 0], out
    )
    # without the tolerance the times are interpolated, the event at 1 is half
    # way between the states before and after it
    assert apply_weights(grid_weights(time, grid), values)[4] == pytest.approx(
        0.5, abs=1e-3
    )
    # a time further before the jump than the tolerance is before it
    early = np.array([1.0 - 1e-8])
    assert apply_weights(grid_weights(time, early, tolerance), values)[0] == (
        pytest.approx(1.0, abs=1e-6)
    )
    assert _interpolate_after_jumps(early, time, values, tolerance)[0] == (
        pytest.approx(1.0, abs=1e-6)
    )


def test_the_fit_keeps_the_time_of_a_change() -> None:
    """A time before a change is before it, its time sees its state after it.

    The states before and after a change at 1 at one time, the first step of
    the integrator after it closer than the tolerance; an event of the model
    at 2, a time a little before it sees the state after it.
    """
    from sbmlsim.fit.optimization import _interpolate_after_jumps

    tolerance = 1e-9
    time = np.array([0.0, 1.0, 1.0, 1.0 + 1e-12, 1.5, 2.0 + 1e-12, 2.0 + 2e-12, 3.0])
    values = np.array([0.0, 1.0, 5.0, 5.0 + 1e-12, 4.0, 3.0, 0.0, 1.0])
    x = np.array([1.0 - 1e-10, 1.0, 2.0])
    out = _interpolate_after_jumps(x, time, values, tolerance, np.array([1.0]))
    assert out[0] == pytest.approx(1.0, abs=1e-6)
    assert out[1] == 5.0
    assert out[2] == 0.0


#: a change at 1, the first step of the integrator after it closer than the
#: tolerance, and an event of the model at 2 which CVODE locates after 2
JUMPS_TIME = np.array([0.0, 1.0, 1.0, 1.0 + 1e-12, 1.5, 2.0 + 1e-13, 2.0 + 1e-13, 3.0])
JUMPS_VALUES = np.array([0.0, 1.0, 5.0, 5.0 + 1e-12, 4.0, 3.0, 0.0, 1.0])


@pytest.mark.parametrize(
    ("grid", "expected"),
    [
        # before the change toward the state before it, also within the
        # tolerance before it
        (0.5, 0.5),
        (1.0 - 1e-10, 1.0 - 1e-10),
        # the time of the change: the state after it, not the step after it
        (1.0, 5.0),
        # at most the tolerance before an event: the state after it
        (2.0, 0.0),
        (2.0 + 1e-13, 0.0),
        (2.5, 0.5),
    ],
)
def test_the_weights_at_a_change_and_an_event(grid: float, expected: float) -> None:
    """A time before a change is before it, the time of a change sees the state after it."""
    weights = grid_weights(JUMPS_TIME, np.array([grid]), 1e-9, np.array([1.0]))
    assert apply_weights(weights, JUMPS_VALUES)[0] == pytest.approx(expected, abs=1e-9)


def test_a_time_twice_in_the_grid_sees_the_states_before_and_after() -> None:
    """The first of two equal times of the grid sees the state before the jump."""
    grid = np.array([1.0, 1.0, 2.0 + 1e-13, 2.0 + 1e-13])
    out = apply_weights(
        grid_weights(JUMPS_TIME, grid, 1e-9, np.array([1.0])), JUMPS_VALUES
    )
    np.testing.assert_allclose(out, [1.0, 5.0, 3.0, 0.0])
    # a grid of the time points of a timecourse gives every row of it
    out = apply_weights(
        grid_weights(JUMPS_TIME, JUMPS_TIME, 1e-9, np.array([1.0])), JUMPS_VALUES
    )
    np.testing.assert_array_equal(out, JUMPS_VALUES)
    # a mask of the times which take the state before the jump at them
    left = np.array([True, False])
    out = apply_weights(
        grid_weights(JUMPS_TIME, np.array([1.0, 1.0]), 1e-9, np.array([1.0]), left),
        JUMPS_VALUES,
    )
    np.testing.assert_array_equal(out, [1.0, 5.0])


def test_a_time_inside_a_chain_of_jumps_sees_the_state_after_the_chain() -> None:
    """A change and an event closer than the tolerance are one chain.

    A change at `2 - 1e-14` and a dose event at 2, which CVODE locates a
    little after 2, followed by a step of the integrator: the time of the
    change and a time inside the chain see the state after the dose, not the
    step after it; a time within the tolerance before the change the state
    before it.
    """
    c = 2.0 - 1e-14
    e = 2.0 + 3e-13
    time = np.array([1.0, c, c, e, e, e + 8e-12, 3.0])
    values = np.array([1.0, 1.0, 2.0, 2.0, 12.0, 12.0 - 1e-11, 11.0])
    grid = np.array([c - 1e-10, c, 2.0, e, e + 1e-12])
    out = apply_weights(grid_weights(time, grid, 1e-9, np.array([c])), values)
    np.testing.assert_allclose(out, [1.0, 12.0, 12.0, 12.0, 12.0 - 1e-12], atol=1e-11)


@pytest.mark.parametrize("step", [False, True])
def test_a_time_within_the_tolerance_before_a_change_sees_the_state_before_it(
    step: bool,
) -> None:
    """A step of the integrator closer to a change than the tolerance is before it.

    The step started the chain of the change, a time between it and the
    change saw the state after the change (5) instead of the value toward
    the state before it (1); a chain starts at its first jump.
    """
    time = np.array([0.0, 1.0, 1.0, 2.0])
    values = np.array([0.0, 1.0, 5.0, 6.0])
    if step:
        time = np.insert(time, 1, 1.0 - 5e-10)
        values = np.insert(values, 1, 1.0 - 5e-10)
    grid = np.array([1.0 - 1e-10, 1.0, 1.5])
    out = apply_weights(grid_weights(time, grid, 1e-9, np.array([1.0])), values)
    np.testing.assert_allclose(out, [1.0 - 1e-10, 5.0, 5.5], rtol=1e-12)


def test_a_time_at_a_change_sees_the_state_after_the_event_just_after_it() -> None:
    """The time of a change in a chain with an event after it sees the state after both.

    A change at 2.5 and an event of the model which CVODE locates 1e-12
    after it; the time of the change saw the state after the change and
    before the event (2), a time between them the state after the event.
    """
    e = 2.5 + 1e-12
    time = np.array([2.0, 2.5, 2.5, e, e, 3.0])
    values = np.array([0.0, 1.0, 2.0, 2.0, 12.0, 11.0])
    grid = np.array([2.5, 2.5 + 5e-13, e])
    out = apply_weights(grid_weights(time, grid, 1e-9, np.array([2.5])), values)
    np.testing.assert_array_equal(out, [12.0, 12.0, 12.0])
    # the first of two equal times, the state before the chain
    grid = np.array([2.5, 2.5])
    out = apply_weights(grid_weights(time, grid, 1e-9, np.array([2.5])), values)
    np.testing.assert_array_equal(out, [1.0, 12.0])


def test_a_step_between_a_time_and_an_event_is_before_the_event() -> None:
    """A time before a step of the integrator before an event is before the event.

    CVODE locates an event at most a little after its time, a step of the
    integrator between a time and the event is before the event, and so is
    the time; a time after the step within the tolerance before the event is
    at it.
    """
    time = np.array([0.0, 1.0 - 5e-10, 1.0, 1.0, 2.0])
    values = np.array([0.0, 1.0 - 5e-10, 1.0, 0.0, 1.0])
    grid = np.array([1.0 - 8e-10, 1.0 - 1e-10, 1.0])
    out = apply_weights(grid_weights(time, grid, 1e-9), values)
    np.testing.assert_allclose(out, [1.0 - 8e-10, 0.0, 0.0], rtol=1e-12)
    # without the step, a time within the tolerance before the event is at it
    without = [0, 2, 3, 4]
    out = apply_weights(grid_weights(time[without], grid, 1e-9), values[without])
    np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])


def test_the_union_of_the_time_points_keeps_a_time_twice() -> None:
    """The union of a ragged result has a time twice which a simulation has twice."""
    time = np.array([[0.0, 1.0, 1.0, 3.0], [0.0, 2.0, 3.0, np.nan]])
    y = np.array([[0.0, 1.0, 5.0, 7.0], [0.0, 4.0, 6.0, np.nan]])
    res = ScanResult(
        xr.Dataset(
            {"time": (("d", "_point"), time), "y": (("d", "_point"), y)},
            coords={"d": [0, 1]},
            attrs={
                "dims": ["d"],
                "units": {"y": "mM", "time": "min"},
                "changes": [1.0],
            },
        )
    )
    np.testing.assert_array_equal(res.time_points(), [0.0, 1.0, 1.0, 2.0, 3.0])
    summary = res.summary(statistics=["max"])
    np.testing.assert_allclose(
        summary["y"].sel(statistic="max").values, [0.0, 2.0, 5.0, 6.0, 7.0]
    )


def test_a_grid_with_a_time_twice_round_trips(tmp_path: Path) -> None:
    """A result whose time is twice at a change is written and read with its changes."""
    ds = xr.Dataset(
        {"y": (("d", "time"), np.array([[0.0, 1.0, 5.0, 6.0], [0.0, 2.0, 7.0, 9.0]]))},
        coords={"d": [0, 1], "time": [0.0, 1.0, 1.0, 2.0]},
        attrs={"dims": ["d"], "units": {"y": "mM", "time": "min"}, "changes": [1.0]},
    )
    res = ScanResult(ds)
    res.to_netcdf(tmp_path / "res.nc")
    back = ScanResult.from_netcdf(tmp_path / "res.nc")
    np.testing.assert_array_equal(back["time"].values, [0.0, 1.0, 1.0, 2.0])
    np.testing.assert_array_equal(back["y"].values, res["y"].values)
    assert back.ds.attrs["changes"] == [1.0]
    np.testing.assert_allclose(
        back.interpolate([0.5, 1.0, 1.5])["y"].values,
        [[0.5, 5.0, 5.5], [1.0, 7.0, 8.0]],
    )


def test_the_timecourses_of_pkpdutils_have_the_state_after_a_change() -> None:
    """pkpdutils takes a time once: the state after a change at its time."""
    ds = xr.Dataset(
        {"y": (("d", "time"), np.array([[1.0, 0.5, 2.0, 1.0], [2.0, 1.0, 4.0, 2.0]]))},
        coords={"d": [0, 1], "time": [0.0, 1.0, 1.0, 2.0]},
        attrs={"dims": ["d"], "units": {"y": "mg/l", "time": "hr"}, "changes": [1.0]},
    )
    timecourses = ScanResult(ds).to_timecourses("y")
    np.testing.assert_array_equal(
        np.asarray(timecourses.ds["time"].values).ravel()[:3], [0.0, 1.0, 2.0]
    )
    # the kept value at the change is the state after it, not the one before
    np.testing.assert_array_equal(
        timecourses.ds["value"].values, [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0]]
    )
