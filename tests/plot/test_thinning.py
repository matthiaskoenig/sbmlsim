"""Tests of the thinning of the lines of a chart, `sbmlsim.plot.thinning`."""

import itertools
import json
from typing import Any

import numpy as np
import pytest

from sbmlsim.plot import Chart, ChartAxis
from sbmlsim.plot.thinning import (
    THIN_MIN_POINTS,
    THIN_TOLERANCE,
    kept_points,
    thin_elements,
)


def _deviation(xu: np.ndarray, yu: np.ndarray, kept: np.ndarray) -> float:
    """Get the largest distance of a point left out from the segment of its kept neighbours."""
    index = np.flatnonzero(kept)
    worst = 0.0
    for a, b in itertools.pairwise(index):
        if b - a < 2:
            continue
        p = np.arange(a + 1, b)
        dx, dy = xu[b] - xu[a], yu[b] - yu[a]
        px, py = xu[p] - xu[a], yu[p] - yu[a]
        length = dx * dx + dy * dy
        t = np.clip((px * dx + py * dy) / length if length > 0 else 0.0, 0.0, 1.0)
        worst = max(worst, float(np.hypot(px - t * dx, py - t * dy).max()))
    return worst


def _doses(n: int = 5000, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Get a timecourse of repeated doses over the uneven steps of an integrator.

    Every dose is a jump: two points at its time.
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 240, n))
    t = np.sort(np.concatenate([t, np.arange(24.0, 240.0, 24.0)]))
    y = np.zeros_like(t)
    for d in np.arange(0.0, 240.0, 24.0):
        after = t > d if d > 0 else t >= d
        y[after] += np.exp(-(t[after] - d) / 8.0)
    # the state before every dose at its time, the jump
    for d in np.arange(24.0, 240.0, 24.0):
        i = int(np.searchsorted(t, d))
        y[i] = y[i - 1]
    return t, y


def _spec(chart: Chart) -> dict[str, Any]:
    """Get the spec of a chart through its JSON, as the renderer reads it."""
    return json.loads(chart.to_json())


def _kept_of(original: np.ndarray, thinned: list[float | None]) -> np.ndarray:
    """Get the points of an increasing column which a thinned column kept."""
    return np.isin(original, np.asarray(thinned, dtype=float))


def test_a_curve_keeps_its_shape_within_the_tolerance() -> None:
    """The line of the kept points is within the tolerance of every point."""
    t, y = _doses()
    xu = (t - t.min()) / np.ptp(t)
    yu = (y - y.min()) / np.ptp(y)

    kept = kept_points(xu, yu[None], THIN_TOLERANCE)

    assert kept[0] and kept[-1]
    assert kept.sum() < t.size / 5
    assert _deviation(xu, yu, kept) <= THIN_TOLERANCE


def test_a_jump_is_kept() -> None:
    """Both points at the time of a dose are kept, the line jumps as before."""
    t, y = _doses()
    xu = (t - t.min()) / np.ptp(t)
    yu = (y - y.min()) / np.ptp(y)

    kept = kept_points(xu, yu[None], THIN_TOLERANCE)

    for d in np.arange(24.0, 240.0, 24.0):
        (at,) = np.flatnonzero(t == d)
        assert kept[at]
        assert kept[at + 1] or y[at + 1] - y[at] < THIN_TOLERANCE * np.ptp(y)


def test_the_chart_thins_a_curve_to_values_of_its_points() -> None:
    """The spec has a subset of the points of the curve, their values bitwise."""
    t, y = _doses()
    chart = Chart()
    chart.plot().curve(t, y)

    (curve,) = _spec(chart)["plots"][0]["elements"]

    assert THIN_MIN_POINTS < len(curve["x"]) < t.size / 5
    kept = _kept_of(t, curve["x"])
    assert kept.sum() == len(curve["x"])
    assert curve["y"] == y[kept].tolist()
    assert (curve["x"][0], curve["x"][-1]) == (t[0], t[-1])
    # the extent of the axes is the one of all points
    assert (min(curve["y"]), max(curve["y"])) == (y.min(), y.max())


def test_the_spec_without_thinning_keeps_every_point() -> None:
    """A chart of `thinning=None` writes every point."""
    t, y = _doses()
    chart = Chart(thinning=None)
    chart.plot().curve(t, y)

    (curve,) = _spec(chart)["plots"][0]["elements"]

    assert curve["x"] == t.tolist()
    assert curve["y"] == y.tolist()


@pytest.mark.parametrize("thinning", [-1e-4, float("nan"), float("inf")])
def test_a_thinning_is_a_share(thinning: float) -> None:
    """A tolerance which is negative or not finite is refused."""
    with pytest.raises(ValueError, match="share of the span"):
        Chart(thinning=thinning)


@pytest.mark.parametrize(
    "style",
    [
        {"marker": "circle"},
        {"line": "none", "marker": "circle"},
        {"line": "none"},
        {"yerr": 0.1},
        {"xerr": 0.1},
    ],
)
def test_data_points_are_kept(style: dict[str, Any]) -> None:
    """A curve with markers, error bars or no line is data: every point is kept."""
    t, y = _doses(500)
    chart = Chart()
    keywords = dict(style)
    for err in ("xerr", "yerr"):
        if err in keywords:
            keywords[err] = np.full(t.shape, keywords[err])
    chart.plot().curve(t, y, **keywords)

    (curve,) = _spec(chart)["plots"][0]["elements"]

    assert len(curve["x"]) == t.size


def test_a_short_curve_is_kept() -> None:
    """A curve of fewer than `THIN_MIN_POINTS` points is kept as it is."""
    t = np.linspace(0, 1, THIN_MIN_POINTS - 1)
    chart = Chart()
    chart.plot().curve(t, 2 * t)

    (curve,) = _spec(chart)["plots"][0]["elements"]

    assert len(curve["x"]) == t.size


def test_a_gap_stays_where_it_is() -> None:
    """A gap and its neighbours are kept, a straight line on both sides is thinned."""
    t = np.linspace(0, 10, 1001)
    y = 2.0 * t
    y[500] = np.nan
    chart = Chart()
    chart.plot().curve(t, y)

    (curve,) = _spec(chart)["plots"][0]["elements"]

    assert curve["x"] == [0.0, t[499], t[500], t[501], 10.0]
    assert curve["y"] == [0.0, y[499], None, y[501], 20.0]


def test_a_log_axis_is_thinned_in_decades() -> None:
    """An exponential decay is a straight line on a log axis, which keeps its ends.

    A value which is not positive is no point of a log axis and is kept with
    its neighbours, as a gap.
    """
    t = np.linspace(0, 6, 2001)
    y = 10.0**-t
    y[1000] = 0.0
    log = Chart()
    log.plot(y=ChartAxis(scale="log")).curve(t, y)
    linear = Chart()
    linear.plot().curve(t, y)

    (on_log,) = _spec(log)["plots"][0]["elements"]
    (on_linear,) = _spec(linear)["plots"][0]["elements"]

    assert on_log["x"] == [0.0, t[999], t[1000], t[1001], 6.0]
    assert len(on_linear["x"]) > 20


def test_a_log_axis_spans_at_most_its_decades() -> None:
    """The tolerance of a log axis is a share of the decades it shows, not of all."""
    t = np.linspace(0, 1, 2001)
    # a wiggle of a thousandth near the top, and a last point 30 decades below
    y = 1.0 + 1e-3 * np.sin(40 * t)
    y[-1] = 1e-30
    wide = Chart()
    wide.plot(y=ChartAxis(scale="log", max_decades=30)).curve(t, y)
    narrow = Chart()
    narrow.plot(y=ChartAxis(scale="log", max_decades=2)).curve(t, y)

    (on_wide,) = _spec(wide)["plots"][0]["elements"]
    (on_narrow,) = _spec(narrow)["plots"][0]["elements"]

    assert len(on_narrow["x"]) > len(on_wide["x"])


def test_a_band_is_thinned_on_all_its_columns() -> None:
    """A band keeps every column within the tolerance, its points shared."""
    t = np.linspace(0, 10, 3001)
    median = np.exp(-t / 3)
    low = 0.8 * median
    # a wiggle of the upper end which the others have not
    high = 1.2 * median + 0.05 * np.sin(5 * t)
    chart = Chart()
    chart.plot().band(t, low, high, median=median)
    curve_of_low = Chart()
    curve_of_low.plot().curve(t, low)

    (band,) = _spec(chart)["plots"][0]["elements"]
    (line,) = _spec(curve_of_low)["plots"][0]["elements"]

    kept = _kept_of(t, band["x"])
    assert len(band["x"]) == len(band["low"]) == len(band["high"])
    assert len(band["median"]) == len(band["x"])
    assert len(band["x"]) > len(line["x"])
    low_y = min(low.min(), high.min(), median.min())
    high_y = max(low.max(), high.max(), median.max())
    for column in (low, high, median):
        assert (
            _deviation(t / 10, (column - low_y) / (high_y - low_y), kept)
            <= THIN_TOLERANCE
        )


def test_an_area_is_thinned_on_both_ends() -> None:
    """An area is thinned as a band without median."""
    t = np.linspace(0, 10, 3001)
    chart = Chart()
    chart.plot().area(t, np.zeros_like(t), 1 - np.exp(-t))

    (area,) = _spec(chart)["plots"][0]["elements"]

    assert len(area["x"]) == len(area["from"]) == len(area["to"]) < t.size / 5
    assert set(area["from"]) == {0.0}


def test_the_ends_of_the_x_axis_keep_the_points_around_them() -> None:
    """The points around an end of the x axis are kept, and the y axis spans the window.

    A peak outside of the window gives the y axis no span: within the window
    the line keeps a wiggle which is small against the peak.
    """
    t = np.linspace(0, 10, 4001)
    y = 1.0 + 1e-3 * np.sin(20 * t)
    y[t > 8] += 1e3 * np.exp(-((t[t > 8] - 9) ** 2))
    chart = Chart()
    chart.plot(x=ChartAxis(min=1.0, max=7.0)).curve(t, y)
    whole = Chart()
    whole.plot().curve(t, y)

    (window,) = _spec(chart)["plots"][0]["elements"]
    (curve,) = _spec(whole)["plots"][0]["elements"]

    assert len(window["x"]) > 10 * len(curve["x"])
    first = int(np.searchsorted(t, 1.0, side="left"))
    last = int(np.searchsorted(t, 7.0, side="right")) - 1
    for i in (first - 1, first, last, last + 1):
        assert t[i] in window["x"]


def test_an_axis_of_categories_thins_nothing() -> None:
    """A curve over categories is kept."""
    names = [f"c{i}" for i in range(150)]
    chart = Chart()
    chart.plot(x=ChartAxis(scale="category")).curve(names, np.arange(150.0))

    (curve,) = _spec(chart)["plots"][0]["elements"]

    assert len(curve["x"]) == 150


def test_thin_elements_takes_the_lists_of_a_spec() -> None:
    """A spec which was written can be thinned, its lists keep their values."""
    t, y = _doses()
    chart = Chart(thinning=None)
    chart.plot().curve(t, y)
    plot = _spec(chart)["plots"][0]
    axes = {name: plot[name] for name in ("x", "y")}

    (curve,) = thin_elements(plot["elements"], axes)
    thinned = Chart()
    thinned.plot().curve(t, y)

    assert isinstance(curve["x"], list)
    (expected,) = _spec(thinned)["plots"][0]["elements"]
    assert (curve["x"], curve["y"]) == (expected["x"], expected["y"])


def test_the_chart_is_thinned_again_after_an_element() -> None:
    """An element added after a spec was built is thinned, and the span it adds counts."""
    t, y = _doses()
    chart = Chart()
    plot = chart.plot()
    plot.curve(t, y)
    first = chart.spec()
    plot.curve(t, 10 * y)
    second = chart.spec()

    assert chart.spec() == second
    (alone,) = first["plots"][0]["elements"]
    below, above = second["plots"][0]["elements"]
    assert len(above["x"]) < t.size / 5
    # the axis spans ten times as much, a tenth of the curve is thinned more
    assert len(below["x"]) < len(alone["x"])


def test_a_thinned_chart_renders() -> None:
    """The renderer draws the thinned spec."""
    t, y = _doses()
    chart = Chart()
    plot = chart.plot(y=ChartAxis(scale="log"))
    plot.curve(t, y)
    plot.band(t, 0.5 * y, 2 * y)

    assert chart.svg().startswith("<svg")
