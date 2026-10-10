"""The lines of a curve over the points of the dimensions of a scan."""

import warnings

import numpy as np
import pytest
import xarray as xr
from matplotlib import colormaps
from matplotlib.colors import to_hex, to_rgb

from sbmlsim import Q
from sbmlsim.plot.points import (
    LINE_STYLES,
    band_lines,
    curve_lines,
    point_colors,
    point_labels,
    point_linestyles,
    point_values,
)
from sbmlsim.simulation import Dimension

TIME = np.array([0.0, 1.0, 2.0])


def _time() -> xr.DataArray:
    return xr.DataArray(TIME, dims=("time",), coords={"time": TIME})


def _scan(*dims: tuple[str, list]) -> xr.DataArray:
    shape = [len(labels) for _, labels in dims] + [TIME.size]
    values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    coords = dict(dims) | {"time": TIME}
    return xr.DataArray(values, dims=(*[n for n, _ in dims], "time"), coords=coords)


def test_one_line_per_point_of_a_dimension() -> None:
    y = _scan(("dose", ["lo", "mid", "hi"]))
    lines = curve_lines("c", ("dose",), _time(), y)
    assert [line.point for line in lines] == [("lo",), ("mid",), ("hi",)]
    assert [line.index for line in lines] == [(0,), (1,), (2,)]
    for k, line in enumerate(lines):
        np.testing.assert_array_equal(line.x, TIME)
        np.testing.assert_array_equal(line.y, y.values[k])


def test_two_dimensions_the_last_fastest() -> None:
    y = _scan(("a", [1, 2]), ("b", ["x", "y", "z"]))
    lines = curve_lines("c", ("a", "b"), _time(), y)
    assert [line.point for line in lines] == [
        (1, "x"),
        (1, "y"),
        (1, "z"),
        (2, "x"),
        (2, "y"),
        (2, "z"),
    ]
    np.testing.assert_array_equal(lines[4].y, y.values[1, 1])


def test_the_order_of_over_decides_and_not_the_order_of_the_data() -> None:
    y = _scan(("a", [1, 2]), ("b", ["x", "y", "z"]))
    lines = curve_lines("c", ("b", "a"), _time(), y)
    assert [line.point for line in lines][:2] == [("x", 1), ("x", 2)]
    np.testing.assert_array_equal(lines[1].y, y.values[1, 0])


def test_a_value_per_simulation_over_a_dimension() -> None:
    dose = xr.DataArray(
        [10.0, 20.0, 40.0], dims=("dose",), coords={"dose": ["lo", "mid", "hi"]}
    )
    cmax = xr.DataArray(
        [1.0, 2.1, 3.9], dims=("dose",), coords={"dose": ["lo", "mid", "hi"]}
    )
    (line,) = curve_lines("c", (), dose, cmax)
    np.testing.assert_array_equal(line.x, [10.0, 20.0, 40.0])
    np.testing.assert_array_equal(line.y, [1.0, 2.1, 3.9])
    points = curve_lines("c", ("dose",), dose, cmax)
    assert len(points) == 3 and float(points[2].y) == 3.9


def test_a_dimension_neither_axis_nor_over_raises() -> None:
    y = _scan(("dose", [0, 1]))
    with pytest.raises(
        ValueError, match=r"curve 'c'.*name \['dose'\] in over=.*Data\(sel=\.\.\.\)"
    ):
        curve_lines("c", (), _time(), y)


def test_over_names_a_dimension_the_data_has_not() -> None:
    with pytest.raises(
        ValueError, match=r"curve 'c' draws a line per point of \['nope'\]"
    ):
        curve_lines("c", ("nope",), _time(), _scan(("dose", [0, 1])))


def test_the_padding_of_a_ragged_result_is_dropped_per_line() -> None:
    t = xr.DataArray([[0.0, 1.0, np.nan], [0.0, 1.0, 2.0]], dims=("dose", "_point"))
    y = xr.DataArray([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]], dims=("dose", "_point"))
    first, second = curve_lines("c", ("dose",), t, y)
    assert first.x.tolist() == [0.0, 1.0] and first.y.tolist() == [1.0, 2.0]
    assert second.x.tolist() == [0.0, 1.0, 2.0]


def test_band_quantiles_ignore_nan() -> None:
    values = np.array(
        [[1.0, 2.0, 3.0], [2.0, np.nan, 4.0], [3.0, 6.0, 5.0], [4.0, 8.0, 6.0]]
    )
    y = xr.DataArray(values, dims=("draw", "time"), coords={"time": TIME})
    (band,) = band_lines("b", (), "draw", (0.25, 0.75), _time(), y)
    np.testing.assert_allclose(band.low, np.nanquantile(values, 0.25, axis=0))
    np.testing.assert_allclose(band.high, np.nanquantile(values, 0.75, axis=0))
    np.testing.assert_allclose(band.median, np.nanmedian(values, axis=0))


def test_a_time_point_of_only_nan_draws_is_nan_without_a_warning() -> None:
    values = np.array(
        [[1.0, np.nan, 3.0], [2.0, np.nan, 4.0], [3.0, np.nan, 5.0], [4.0, np.nan, 6.0]]
    )
    y = xr.DataArray(values, dims=("draw", "time"), coords={"time": TIME})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (band,) = band_lines("b", (), "draw", (0.25, 0.75), _time(), y)
    assert np.isnan(band.low[1]) and np.isnan(band.median[1]) and np.isnan(band.high[1])
    keep = [0, 2]
    np.testing.assert_allclose(
        band.low[keep], np.nanquantile(values[:, keep], 0.25, axis=0)
    )
    np.testing.assert_allclose(
        band.high[keep], np.nanquantile(values[:, keep], 0.75, axis=0)
    )
    np.testing.assert_allclose(band.median[keep], np.nanmedian(values[:, keep], axis=0))


def test_a_band_per_point_of_another_dimension() -> None:
    values = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    y = xr.DataArray(
        values,
        dims=("dose", "draw", "time"),
        coords={"dose": ["lo", "hi"], "time": TIME},
    )
    bands = band_lines("b", ("dose",), "draw", (0.05, 0.95), _time(), y)
    assert [band.point for band in bands] == [("lo",), ("hi",)]
    np.testing.assert_allclose(bands[1].median, np.median(values[1], axis=0))


def test_a_band_of_a_ragged_result_raises() -> None:
    y = xr.DataArray(np.ones((4, 3)), dims=("draw", "_point"))
    t = xr.DataArray(np.ones((4, 3)), dims=("draw", "_point"))
    with pytest.raises(ValueError, match=r"band 'b'.*common grid"):
        band_lines("b", (), "draw", (0.05, 0.95), t, y)


def test_a_band_across_a_dimension_the_data_has_not_raises() -> None:
    with pytest.raises(ValueError, match=r"band 'b' reduces the dimension 'draw'"):
        band_lines("b", (), "draw", (0.05, 0.95), _time(), _scan(("dose", [0, 1])))


def test_point_colors() -> None:
    shades = point_colors(3, "#ff0000")
    assert shades[-1] == "#ff0000" and len(set(shades)) == 3
    assert sum(to_rgb(shades[0])) > sum(to_rgb(shades[1]))  # lighter first
    viridis = point_colors(3, None)
    assert viridis[0] == to_hex(colormaps["viridis"](0.0)) and len(set(viridis)) == 3
    assert point_colors(1, "#00ff00") == ["#00ff00"]


def test_point_linestyles() -> None:
    assert point_linestyles(3) == list(LINE_STYLES[:3])
    with pytest.raises(ValueError, match="at most 4"):
        point_linestyles(5)


def test_point_labels_name_the_values_of_a_changed_target() -> None:
    dose = Dimension("dose", values={"PODOSE": Q([50.0, 100.0], "mg")})
    assert point_labels("dose", [0, 1], dose, {}) == [
        "PODOSE = 50 mg",
        "PODOSE = 100 mg",
    ]
    target, values, unit = point_values(dose, {}) or ("", np.array([]), "")
    assert target == "PODOSE" and unit == "mg" and values.tolist() == [50.0, 100.0]
    plain = Dimension("ke", values={"ke": np.array([0.1, 0.2])})
    assert point_labels("ke", [0, 1], plain, {"ke": "1/hr"}) == [
        "ke = 0.1 1/hr",
        "ke = 0.2 1/hr",
    ]
    two = Dimension(
        "both",
        values={"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
        labels=["x", "y"],
    )
    assert point_labels("both", ["x", "y"], two, {}) == ["both = x", "both = y"]
    assert point_labels("d", ["lo", "hi"], None, {}) == ["d = lo", "d = hi"]
