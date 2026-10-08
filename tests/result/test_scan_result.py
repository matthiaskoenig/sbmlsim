"""The result of a scan wraps a dataset with the units of its variables."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from sbmlsim import Q
from sbmlsim.result import ScanResult
from sbmlsim.result.timecourse import interpolate


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
