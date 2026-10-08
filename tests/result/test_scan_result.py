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
