"""The lines of a curve over the points of the dimensions of a scan."""

import warnings

import numpy as np
import pytest
import xarray as xr

from sbmlsim import Q
from sbmlsim.engine import import_core
from sbmlsim.plot.points import (
    LINE_STYLES,
    Grids,
    Line,
    band_label,
    band_lines,
    colorbar_arguments,
    curve_lines,
    format_value,
    nan_quantile,
    nan_quantiles,
    onto_grid,
    point_colors,
    point_labels,
    point_linestyles,
    point_scale,
    point_styles,
    point_values,
)
from sbmlsim.simulation import Dimension, Simulation


def _viridis(*fractions: float) -> list[str]:
    return list(import_core().color_scale("viridis", 0.0, 1.0, list(fractions)))


def _rgb(color: str) -> list[int]:
    return [int(color[i : i + 2], 16) for i in (1, 3, 5)]


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


@pytest.mark.parametrize(
    ("name", "quantiles", "label"),
    [
        ("C", (0.05, 0.95), "C (5-95 %)"),
        ("C, PODOSE = 50 mg", (0.0, 1.0), "C, PODOSE = 50 mg (0-100 %)"),
        ("C", (0.025, 0.975), "C (2.5-97.5 %)"),
        # a band is named after its y by default, a label without a name
        # is the quantile range alone
        (None, (0.25, 0.75), "25-75 %"),
        ("", (0.25, 0.75), "25-75 %"),
    ],
)
def test_the_label_of_a_band(
    name: str | None, quantiles: tuple[float, float], label: str
) -> None:
    assert band_label(name, quantiles) == label


def _ragged(times: list[list[float]], values: list[list[float]]) -> tuple:
    """The time and a variable of a ragged result over (draw, _point)."""
    n = max(len(t) for t in times)
    pad = [[*t, *[np.nan] * (n - len(t))] for t in times]
    vals = [[*v, *[np.nan] * (n - len(v))] for v in values]
    return (
        xr.DataArray(np.array(pad), dims=("draw", "_point")),
        xr.DataArray(np.array(vals), dims=("draw", "_point")),
    )


def test_a_band_of_a_ragged_result_is_on_the_times_of_its_draws() -> None:
    """Every draw is interpolated onto the time points of all, not mixed."""
    t, y = _ragged([[0.0, 2.0], [0.0, 1.0, 2.0]], [[0.0, 4.0], [0.0, 3.0, 6.0]])
    (band,) = band_lines("b", (), "draw", (0.0, 1.0), t, y)
    np.testing.assert_allclose(band.x, [0.0, 1.0, 2.0])
    # the first draw at 1 is interpolated between its own times, 2
    np.testing.assert_allclose(band.low, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(band.high, [0.0, 3.0, 6.0])
    np.testing.assert_allclose(band.median, [0.0, 2.5, 5.0])


def test_a_draw_is_nan_outside_of_its_times() -> None:
    t, y = _ragged([[0.0, 1.0], [0.0, 1.0, 2.0]], [[1.0, 1.0], [2.0, 2.0, 2.0]])
    x, values = onto_grid("b", t, y)
    assert values.dims == ("draw", "time")
    np.testing.assert_allclose(x.values, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(values.values[0], [1.0, 1.0, np.nan])


def test_a_change_gives_the_state_after_it() -> None:
    """The two states at the time of a change are interpolated as the one after."""
    t, y = _ragged([[0.0, 1.0, 1.0, 2.0]], [[0.0, 1.0, 5.0, 6.0]])
    x, values = onto_grid("b", t, y)
    np.testing.assert_allclose(x.values, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(values.values[0], [0.0, 5.0, 6.0])


def test_a_band_of_a_ragged_result_needs_increasing_x() -> None:
    t, y = _ragged([[0.0, 2.0, 1.0]], [[0.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match=r"band 'b'.*increase"):
        band_lines("b", (), "draw", (0.05, 0.95), t, y)


def test_a_band_across_a_dimension_the_data_has_not_raises() -> None:
    with pytest.raises(ValueError, match=r"band 'b' reduces the dimension 'draw'"):
        band_lines("b", (), "draw", (0.05, 0.95), _time(), _scan(("dose", [0, 1])))


def test_point_colors() -> None:
    shades = point_colors(3, "#ff0000")
    assert shades[-1] == "#ff0000" and len(set(shades)) == 3
    assert sum(_rgb(shades[0])) > sum(_rgb(shades[1]))  # lighter first
    viridis = point_colors(3, None)
    assert viridis == _viridis(0.0, 0.425, 0.85)
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


def _lines(dimension: Dimension, labels: list | None = None) -> list[Line]:
    """One line per label, all labels of the dimension by default."""
    chosen = dimension.labels.tolist() if labels is None else labels
    return [
        Line(point=(label,), index=(i,), x=TIME, y=TIME)
        for i, label in enumerate(chosen)
    ]


def test_point_styles_of_one_dimension() -> None:
    dose = Dimension("dose", values={"PODOSE": Q([50.0, 100.0, 200.0], "mg")})
    style = point_styles(_lines(dose), ["dose"], None, [dose], {})
    assert style.colors == _viridis(0.0, 0.85 / 3, 0.85)
    assert style.linestyles is None
    assert style.labels[0] == ["PODOSE = 50 mg", "PODOSE = 100 mg", "PODOSE = 200 mg"]
    assert style.colorbar is False
    assert style.title == "PODOSE [mg]"
    np.testing.assert_allclose(style.values, [50.0, 100.0, 200.0])


def test_point_styles_colour_bar_from_eleven_points() -> None:
    dose = Dimension("dose", values={"PODOSE": Q(np.arange(1.0, 13.0), "mg")})
    lines = [
        Line(point=(label,), index=(i,), x=TIME, y=TIME)
        for i, label in enumerate(dose.labels)
    ]
    assert point_styles(lines, ["dose"], None, [dose], {}).colorbar is True


def test_point_styles_of_two_dimensions() -> None:
    dose = Dimension("dose", values={"PODOSE": Q([50.0, 100.0], "mg")})
    rate = Dimension("rate", values={"ke": np.array([0.1, 0.3])})
    lines = [
        Line(point=(a, b), index=(i, j), x=TIME, y=TIME)
        for i, a in enumerate(dose.labels)
        for j, b in enumerate(rate.labels)
    ]
    style = point_styles(lines, ["dose", "rate"], None, [dose, rate], {"ke": "1/hr"})
    assert style.linestyles == ["-", "--"]
    assert style.labels[1] == ["ke = 0.1 1/hr", "ke = 0.3 1/hr"]


def test_point_styles_without_a_target_use_the_positions() -> None:
    sims = Dimension("s", simulations={"a": Simulation(end=1), "b": Simulation(end=2)})
    lines = [
        Line(point=(label,), index=(i,), x=TIME, y=TIME)
        for i, label in enumerate(sims.labels)
    ]
    style = point_styles(lines, ["s"], None, [sims], {})
    assert (
        style.title == "s"
        and style.scale is not None
        and style.scale.labels == ["a", "b"]
    )
    np.testing.assert_allclose(style.values, [0.0, 1.0])


def test_point_labels_and_values_of_selected_points_follow_their_labels() -> None:
    dose = Dimension("dose", values={"PODOSE": Q([50.0, 100.0, 200.0], "mg")})
    assert point_labels("dose", [2, 1], dose, {}) == [
        "PODOSE = 200 mg",
        "PODOSE = 100 mg",
    ]
    _, values, _ = point_values(dose, {}, [2, 0]) or ("", np.array([]), "")
    assert values.tolist() == [200.0, 50.0]


def test_point_styles_of_selected_points_of_two_dimensions() -> None:
    dose = Dimension("dose", values={"PODOSE": Q([50.0, 100.0, 200.0], "mg")})
    rate = Dimension("rate", values={"ke": np.linspace(0.1, 0.5, 5)})
    lines = [
        Line(point=(a, b), index=(i, j), x=TIME, y=TIME)
        for i, a in enumerate([2, 1])
        for j, b in enumerate([3, 1])
    ]
    style = point_styles(lines, ["dose", "rate"], None, [dose, rate], {"ke": "1/hr"})
    assert style.labels == [
        ["PODOSE = 200 mg", "PODOSE = 100 mg"],
        ["ke = 0.4 1/hr", "ke = 0.2 1/hr"],
    ]
    assert style.linestyles == ["-", "--"]
    np.testing.assert_allclose(style.values, [200.0, 100.0])
    assert style.colors == _viridis(0.85, 0.0)


def test_point_scale_is_logarithmic_above_a_factor_of_one_hundred() -> None:
    assert point_scale(np.array([1.0, 101.0])) == (1.0, 101.0, True)
    assert point_scale(np.array([1.0, 100.0])) == (1.0, 100.0, False)
    assert point_scale(np.array([0.0, 1000.0])) == (0.0, 1000.0, False)
    assert point_scale(np.array([-1.0, 1000.0])) == (-1.0, 1000.0, False)
    assert point_scale(np.array([5.0, 5.0])) is None
    assert point_scale(np.array([5.0, np.nan])) is None


def test_point_colours_are_the_colour_scale_at_their_values() -> None:
    values = np.geomspace(1, 1000, 12)
    geometric = Dimension("dose", values={"PODOSE": Q(values, "mg")})
    style = point_styles(_lines(geometric), ["dose"], None, [geometric], {})
    assert style.scale is not None and style.scale.log and style.colorbar
    assert (style.scale.low, style.scale.high) == (1.0, 1000.0)
    fractions = np.log10(values) / 3 * 0.85
    assert style.colors == _viridis(*fractions)
    assert style.scale.labels == []
    assert not style.scale.categorical
    assert style.middle == _viridis(0.425)[0]

    decreasing = Dimension("dose", values={"PODOSE": Q(np.linspace(120, 10, 12), "mg")})
    style = point_styles(_lines(decreasing), ["dose"], "#d62728", [decreasing], {})
    assert style.scale is not None and not style.scale.log
    assert (style.scale.low, style.scale.high) == (10.0, 120.0)
    assert style.colors[0] == "#d62728"
    assert style.colors[-1] == point_colors(2, "#d62728")[0]
    # the middle shade, between the lightest and the colour
    assert style.middle not in (style.colors[0], style.colors[-1])


def test_point_styles_without_a_target_are_discrete_with_the_labels() -> None:
    cond = Dimension(
        "cond",
        values={"a": np.array([1.0, 2.0, 3.0]), "b": np.array([3.0, 4.0, 5.0])},
        labels=[0, 10, 20],
    )
    style = point_styles(_lines(cond), ["cond"], None, [cond], {})
    assert style.title == "cond"
    assert style.scale is not None and style.scale.categorical
    assert (style.scale.low, style.scale.high) == (0.0, 2.0)
    assert style.scale.labels == ["0", "10", "20"]
    assert style.colors == point_colors(3, None)
    assert style.middle == style.colors[1]
    assert style.labels[0] == ["cond = 0", "cond = 10", "cond = 20"]


def test_format_value() -> None:
    assert format_value(25.0) == "25"
    assert format_value(0.1) == "0.1"
    assert format_value(1e-5) == "1e-5"
    assert format_value(1234567.0) == "1.23457e6"
    assert format_value(0.00012) == "0.00012"


@pytest.mark.parametrize("q", [0.0, 0.05, 0.5, 0.95, 1.0])
def test_nan_quantile_is_the_one_of_numpy(q: float) -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=(7, 5, 31))
    values[values > 1.5] = np.nan
    values[3, 2, :] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.nanquantile(values, q, axis=-1)
    np.testing.assert_allclose(nan_quantile(values, q), expected, equal_nan=True)


def test_the_quantiles_sorted_once_are_the_ones_of_nan_quantile() -> None:
    """The quantiles of a band and its median share the sort, value for value."""
    rng = np.random.default_rng(3)
    values = rng.normal(size=(6, 4, 25))
    values[values > 1.2] = np.nan
    values[2, 1, :] = np.nan
    qs = (0.05, 0.5, 0.95)
    for quantile, q in zip(nan_quantiles(values, qs), qs, strict=True):
        np.testing.assert_array_equal(quantile, nan_quantile(values, q))


def _many_steps(seed: int, draws: int = 20) -> tuple[xr.DataArray, xr.DataArray]:
    rng = np.random.default_rng(seed)
    times = [np.sort(rng.uniform(0, 10, 2000)) for _ in range(draws)]
    times = [np.concatenate([[0.0], t, [5.0, 5.0], [10.0]]) for t in times]
    times = [np.sort(t) for t in times]
    return _ragged([list(t) for t in times], [list(np.exp(-t)) for t in times])


def test_the_bands_of_one_x_share_its_grid() -> None:
    """The bands of the variables of one task find the time points once.

    A band with the grids of a figure is the one without, bit for bit; an x
    of other values gets a grid of its own.
    """
    t, y = _many_steps(4)
    grids = Grids()
    x, values = onto_grid("b", t, y, grids=grids)
    alone_x, alone = onto_grid("b", t, y)
    xr.testing.assert_identical(x, alone_x)
    xr.testing.assert_identical(values, alone)
    # a band of another variable of the same x takes its grid
    found = grids.of("b", np.asarray(t.values), 500)
    assert grids.of("b", np.asarray(t.values).copy(), 500) is found
    x2, values2 = onto_grid("b", t, 2.0 * y, grids=grids)
    xr.testing.assert_identical(x2, alone_x)
    np.testing.assert_array_equal(values2.values, onto_grid("b", t, 2.0 * y)[1].values)
    # an x of other values, or another number of time points, is another grid
    other, _ = _many_steps(5)
    assert grids.of("b", np.asarray(other.values), 500) is not found
    assert grids.of("b", np.asarray(t.values), 300) is not found
    # the grid keeps its x, which the data may change after
    changed = np.asarray(t.values).copy()
    grid = Grids().of("b", changed, 500)
    changed[0, 1] = 0.5
    assert grid.xs[0, 1] != 0.5


def test_a_band_of_x_without_the_draws_is_broadcast() -> None:
    """The time of a ragged result over the draws, x of one simulation for all."""
    t, y = _ragged(
        [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], [[0.0, 2.0, 4.0], [0.0, 3.0, 6.0]]
    )
    x, values = onto_grid("b", t.isel(draw=0, drop=True), y, grids=Grids())
    np.testing.assert_array_equal(x.values, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(values.values, y.values)


def test_a_band_of_many_steps_has_the_time_points_of_a_figure() -> None:
    """The draws are not put on the union of their steps but on a few hundred."""
    rng = np.random.default_rng(2)
    times = [np.sort(rng.uniform(0, 10, 2000)) for _ in range(20)]
    times = [np.concatenate([[0.0], t, [10.0]]) for t in times]
    t, y = _ragged([list(t) for t in times], [list(np.exp(-t)) for t in times])
    x, values = onto_grid("b", t, y, n=300)
    assert x.size <= 301
    assert values.shape == (20, x.size)
    np.testing.assert_allclose(values.values[0], np.exp(-x.values), atol=1e-3)


def test_no_matplotlib_import() -> None:
    import ast
    from pathlib import Path

    import sbmlsim.plot.points as points

    tree = ast.parse(Path(points.__file__).read_text(encoding="utf-8"))
    modules = [
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in getattr(node, "names", [])
    ]
    assert not [m for m in modules if m and m.split(".")[0] == "matplotlib"]


def test_point_styles_without_points() -> None:
    style = point_styles([], ["a"], None, [None], {})
    assert style.scale is None and style.colors == []
    assert style.middle == _viridis(0.425)[0]


def test_colorbar_arguments_span_the_values_of_the_points() -> None:
    dose = Dimension("dose", values={"PODOSE": Q(np.linspace(10.0, 120.0, 12), "mg")})
    style = point_styles(_lines(dose), ["dose"], None, [dose], {})
    assert colorbar_arguments(style) == {
        "scale": "viridis",
        "low": 10.0,
        "high": 120.0,
        "label": "PODOSE [mg]",
        "log": False,
        "end": 0.85,
    }


def test_colorbar_arguments_of_labels_are_categorical_up_to_max_bands() -> None:
    def labelled(n: int) -> dict | None:
        cond = Dimension(
            "cond", values={"a": np.arange(n), "b": np.arange(n)}, labels=list(range(n))
        )
        return colorbar_arguments(
            point_styles(_lines(cond), ["cond"], None, [cond], {})
        )

    few = labelled(3)
    assert few is not None and few["categorical"] is True
    assert few["tick_labels"] == ["0", "1", "2"]
    assert (few["low"], few["high"]) == (0.0, 2.0)
    many = labelled(300)
    assert many is not None and "categorical" not in many and many["high"] == 299.0
    assert labelled(1) is None
