"""A curve is one line of broadcast arrays, without the padding of a ragged result."""

import numpy as np
import pytest
import xarray as xr

from sbmlsim.plot.padding import line_values, without_padding


def _time(values: list[float]) -> xr.DataArray:
    return xr.DataArray(
        np.array(values), dims=("time",), coords={"time": np.array(values)}
    )


def test_one_line_of_a_timecourse() -> None:
    t = _time([0.0, 1.0, 2.0])
    y = xr.DataArray([1.0, 2.0, 3.0], dims=("time",), coords={"time": t.values})
    x, values, err = line_values("c", t, y, None)
    assert x.tolist() == [0.0, 1.0, 2.0] and values.tolist() == [1.0, 2.0, 3.0]
    assert err is None


def test_a_curve_of_a_ragged_point_drops_the_padding() -> None:
    t = xr.DataArray([0.0, 1.0, np.nan], dims=("_point",))
    y = xr.DataArray([1.0, np.nan, np.nan], dims=("_point",))
    x, values = line_values("c", t, y)
    assert x.tolist() == [0.0, 1.0]
    np.testing.assert_array_equal(values, [1.0, np.nan])


def test_a_dimension_of_a_scan_raises() -> None:
    t = _time([0.0, 1.0, 2.0])
    y = xr.DataArray(
        np.ones((2, 3)),
        dims=("dose", "time"),
        coords={"time": t.values, "dose": [0, 1]},
    )
    with pytest.raises(ValueError, match=r"curve 'c'.*'dose'.*Data\(sel=\.\.\.\)"):
        line_values("c", t, y)


def test_a_value_broadcasts_against_rows() -> None:
    x = xr.DataArray([1.0, 2.0], dims=("row",))
    y = xr.DataArray(5.0)
    xs, ys = line_values("c", x, y)
    assert xs.tolist() == [1.0, 2.0] and ys.tolist() == [5.0, 5.0]


def test_different_coordinates_raise() -> None:
    with pytest.raises(ValueError, match=r"curve 'c'.*coordinates"):
        line_values("c", _time([0.0, 1.0]), _time([0.0, 2.0]))


def test_nothing_is_nothing() -> None:
    assert line_values("c", None, None) == (None, None)


def test_without_padding_keeps_a_nan_of_y() -> None:
    x, y = without_padding(np.array([0.0, np.nan, 2.0]), np.array([1.0, 2.0, np.nan]))
    assert x.tolist() == [0.0, 2.0]
    np.testing.assert_array_equal(y, [1.0, np.nan])
