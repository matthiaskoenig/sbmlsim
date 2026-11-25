"""The plots of a sensitivity result.

Every plot is a function which returns a `sbmlsim.plot.Chart`, which the
renderer of sbmlsim-core draws and `Chart.save` writes as SVG or PNG:

- `plot_heatmap`: an index of the scalar observables over the parameters,
  clustered;
- `plot_indices`: the bars of `S1` and `ST` of an observable with their
  intervals;
- `plot_morris`: `mu_star` against `sigma` of an observable.

`box_stats` are the statistics of a box plot as matplotlib computes them,
`draw_boxes` draws them into a plot of a chart and `density` is the density of
a violin.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
import xarray as xr
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde

from sbmlsim.engine import import_core
from sbmlsim.plot.chart import Chart, ChartAxis, ChartPlot
from sbmlsim.plot.colors import normalize_color
from sbmlsim.sensitivity.result import (
    DIMENSIONLESS_INDICES,
    PARAMETER,
    SensitivityResult,
    at_one_time,
)
from sbmlsim.units import unit_label

SIGNED_INDICES = ("raw", "normalized", "mu")
"""The indices which have a sign, drawn on a diverging color scale around 0."""

BOUNDED_INDICES = ("S1", "ST")
"""The indices in the interval 0 to 1."""

INDEX_NAMES = {"sobol": "Sobol index", "fast": "FAST index"}
"""The label of the axis of the indices of a method."""

DEFAULT_CUTOFF = 0.1
"""The cutoff of the heatmap of a dimensionless index, see `plot_heatmap`."""

COLOR_SCALES: Final = ("viridis", "magma", "rdbu", "coolwarm", "greys")
"""The color scales of the renderer, the values of `cmap` of `plot_heatmap`."""

COLOR_SCALE_ALIASES: Final = {"seismic": "coolwarm", "bwr": "coolwarm"}
"""The color maps of matplotlib which a heatmap draws with a scale of the renderer."""

SIGNED_SCALE: Final = "coolwarm"
"""The color scale of a signed index, centered at 0."""

BOUNDED_SCALE: Final = "viridis"
"""The color scale of an index without a sign."""

CELL_SIZE = (64.8, 32.4)
"""The width and the height of a cell of the heatmap in pixels (0.9 and 0.45 inches)."""

MIN_CELLS_HEIGHT = 129.6
"""The smallest height of the cells of a heatmap in pixels, room for its colorbar."""

MAX_HEATMAP_SIZE = (1728.0, 2880.0)
"""The largest width and height of the chart of a heatmap in pixels (24 x 40 inches)."""

COLORBAR_ROOM = 72.0
"""The pixels right of the cells of a heatmap for its colorbar, its ticks and label."""

TITLE_ROOM = 34.0
"""The pixels above the plot for the title of a chart."""

TICK_PAD = 10.0
"""The pixels between an axis and its tick labels, with the room of the ticks."""

EDGE = 8.0
"""The pixels between a text and the edge of a chart."""

TEXT_SIZE = 11.0
"""The font size of the tick labels and the texts of the renderer in pixels."""

ROTATED = 45.0
"""The rotation of the tick labels which do not fit side by side, in degrees."""

VERTICAL = 90.0
"""The rotation of the tick labels which do not fit rotated by `ROTATED`, whose
first label would reach left of the room of the axis left of the plot."""

UNDEFINED_COLOR = "#d3d3d3"
"""The color of a cell whose index is undefined (`NaN`), the one the renderer gives it."""

HIDDEN_COLOR = "#ffffff"
"""The color of a cell whose index is below the cutoff."""

S1_COLOR = "#2a78d6"
"""The color of the bars of `S1`."""

ST_COLOR = "#eb6834"
"""The color of the bars of `ST`, which contrasts with `S1` and the intervals."""

INTERVAL_COLOR = "#000000"
"""The color of the edges of the points of the Morris plot."""

NOTE_COLOR = "#696969"
"""The color of a note in a plot, e.g. the parameters a Morris plot leaves out."""

BOX_EDGE = "#000000"
"""The color of the outline and the median of a filled box."""

BOX_OPACITY = 0.6
"""The opacity of the fill of a filled box."""

WHISKERS = 1.5
"""The whiskers of a box in interquartile ranges, the default of matplotlib."""

DENSITY_POINTS = 128
"""The points of the density of a violin between the smallest and the largest value."""


def box_stats(values: Any, whis: float = WHISKERS) -> dict[str, Any] | None:
    """Get the statistics of a box of a sample as `matplotlib.cbook.boxplot_stats`.

    The box is the first to the third quartile (linear interpolation of
    `numpy.percentile`) with the median; a whisker reaches the last value
    within `whis` interquartile ranges of its quartile, and the quartile
    itself when no value is there; the values beyond the whiskers are the
    outliers.

    Args:
        values: the sample; values which are not finite are left out.
        whis: the length of the whiskers in interquartile ranges.

    Returns:
        The statistics `low_whisker`, `q1`, `median`, `q3`, `high_whisker` and
        `outliers` of `ChartPlot.box`, `None` for a sample without a finite
        value.
    """
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    q1, median, q3 = np.percentile(x, [25.0, 50.0, 75.0])
    iqr = q3 - q1
    low, high = q1 - whis * iqr, q3 + whis * iqr
    inside_high = x[x <= high]
    high_whisker = q3 if inside_high.size == 0 else max(q3, inside_high.max())
    inside_low = x[x >= low]
    low_whisker = q1 if inside_low.size == 0 else min(q1, inside_low.min())
    outliers = x[(x < low_whisker) | (x > high_whisker)]
    return {
        "low_whisker": float(low_whisker),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "high_whisker": float(high_whisker),
        "outliers": outliers.tolist(),
    }


def draw_boxes(
    plot: ChartPlot,
    data: Sequence[Any],
    positions: Sequence[float] | Sequence[str] | None = None,
    *,
    colors: Sequence[str] | None = None,
    showfliers: bool = True,
    label: str | None = None,
) -> list[str]:
    """Draw a box per sample into a plot of a chart, with the statistics of `box_stats`.

    Args:
        plot: the plot.
        data: the values of every box; a sample without a finite value has
            no box.
        positions: the positions of the boxes or the names of their
            categories, `1, ..., n` by default.
        colors: fill every box with a color, outlined in black, one element
            per box; one element of the boxes in the default color without.
        showfliers: draw the values beyond the whiskers.
        label: the label of the boxes in the legend, of one element only.

    Returns:
        The ids of the elements.

    Raises:
        ValueError: for a number of colors or positions other than the boxes.
    """
    n = len(data)
    if positions is None:
        positions = [float(k) for k in range(1, n + 1)]
    if len(positions) != n:
        raise ValueError(f"{len(positions)} positions for {n} boxes.")
    if colors is not None and len(colors) != n:
        raise ValueError(f"{len(colors)} colors for {n} boxes.")
    boxes = [
        (k, stats if showfliers else {**stats, "outliers": []})
        for k, stats in enumerate(box_stats(d) for d in data)
        if stats is not None
    ]
    if not boxes:
        return []
    if colors is None:
        return [
            plot.box(
                [positions[k] for k, _ in boxes],
                [stats for _, stats in boxes],
                label=label,
            )
        ]
    return [
        plot.box(
            [positions[k]],
            [stats],
            color=BOX_EDGE,
            width=1.0,
            fill=normalize_color(colors[k]),
            fill_opacity=BOX_OPACITY,
        )
        for k, stats in boxes
    ]


def density(sample: Any) -> dict[str, np.ndarray]:
    """Get the density of a sample for a violin, `ChartPlot.violin`.

    Args:
        sample: the finite values, at least two distinct ones.

    Returns:
        The `values` from the smallest to the largest value at
        `DENSITY_POINTS` points and the `density` of
        `scipy.stats.gaussian_kde` at them.
    """
    x = np.asarray(sample, dtype=float).ravel()
    values = np.linspace(x.min(), x.max(), DENSITY_POINTS)
    return {"values": values, "density": gaussian_kde(x)(values)}


def color_scale(cmap: str) -> str:
    """Get the color scale of the renderer of a color map.

    Args:
        cmap: a scale of `COLOR_SCALES` in any case, or a color map of
            matplotlib of `COLOR_SCALE_ALIASES`, e.g. `seismic`.

    Returns:
        The name of the scale.

    Raises:
        ValueError: for another color map, naming the scales.
    """
    name = cmap.lower()
    name = COLOR_SCALE_ALIASES.get(name, name)
    if name not in COLOR_SCALES:
        raise ValueError(
            f"The color scale '{cmap}' is none of the scales of the renderer, "
            f"{list(COLOR_SCALES)}, or of their aliases "
            f"{sorted(COLOR_SCALE_ALIASES)}."
        )
    return name


def _value_format(top: float) -> str:
    """Get the format of the values in the cells for the largest magnitude.

    Args:
        top: the largest magnitude of the values.

    Returns:
        The format: two decimals from `0.1` to `1000`, none up to `1e5` and
        scientific otherwise.
    """
    if top == 0.0:
        return "{:.0f}"
    if 0.1 <= top < 1000.0:
        return "{:.2f}"
    if 1000.0 <= top < 1e5:
        return "{:.0f}"
    return "{:.1e}"


def _cell_text(value: float, fmt: str) -> str:
    """Get the text of a cell, a zero without a sign or an exponent.

    Args:
        value: the value of the cell.
        fmt: the format of the values.

    Returns:
        The text, `0` for a value which rounds to zero, else the formatted
        value with a minus sign.
    """
    text = fmt.format(value)
    if float(text) == 0.0:
        return "0"
    return text.replace("-", "\N{MINUS SIGN}")


def _order(values: np.ndarray, cluster: bool) -> np.ndarray:
    """Get the order of the rows by a hierarchical clustering.

    Args:
        values: the values, a row per entry; `NaN` and infinite values count as 0.
        cluster: whether the rows are clustered.

    Returns:
        The positions of the rows in their order; their own order without
        clustering or with fewer than two rows.
    """
    if not cluster or values.shape[0] < 2:
        return np.arange(values.shape[0])
    finite = np.where(np.isfinite(values), values, 0.0)
    return leaves_list(linkage(pdist(finite), method="single"))


def _text_width(text: str) -> float:
    """Get the width of a text at the size of the ticks in pixels, as the renderer measures it."""
    return float(import_core().text_width(text, TEXT_SIZE))


def _widest(texts: Sequence[str]) -> float:
    """Get the width of the widest of the texts at the size of the ticks in pixels."""
    return max((_text_width(t) for t in texts), default=0.0)


def _rotation(labels: Sequence[str], room: float, left: float) -> float:
    """Get the rotation of the tick labels of an x axis in degrees.

    The labels are horizontal when they fit side by side, rotated by
    `ROTATED` (ending at their tick) when they do not, and vertical when the
    first rotated label would reach further left than the room left of the
    plot, which the renderer (sbmlsim-core 0.28) does not widen for it.

    Args:
        labels: the tick labels.
        room: the pixels of one tick label along the axis.
        left: the pixels left of the first tick, its half of the room and
            the room of the axis left of the plot.

    Returns:
        `0`, `ROTATED` or `VERTICAL`.
    """
    widest = _widest(labels)
    if widest <= room - 6.0:
        return 0.0
    if widest * math.cos(math.radians(ROTATED)) <= left:
        return ROTATED
    return VERTICAL


def _below(labels: Sequence[str], rotation: float) -> float:
    """Get the pixels below an x axis its tick labels take at a rotation."""
    if rotation == 0.0:
        return TEXT_SIZE * 1.4
    angle = math.radians(rotation)
    return _widest(labels) * math.sin(angle) + TEXT_SIZE * math.cos(angle)


def _heatmap(
    df: pd.DataFrame,
    *,
    label: str,
    cutoff: float,
    cluster_rows: bool,
    title: str | None,
    cmap: str,
    vcenter: float | None,
    vmin: float,
    vmax: float,
) -> Chart:
    """Draw a clustered heatmap of a table, a row per parameter and a column per observable.

    A row is kept if one of its values reaches the cutoff or one is undefined
    (`NaN` or infinite); a cell below the cutoff is white, an undefined cell
    light grey, and every other cell carries its value. The rows are
    clustered on the values with `NaN` as 0. The cells have a fixed size up
    to the largest chart, the parameters are on the left, the observables
    below (rotated when they do not fit side by side) and the colorbar with
    its label on the right.

    Args:
        df: the values, a row per parameter and a column per observable.
        label: the label of the colorbar, the index and its unit.
        cutoff: rows whose finite values are all below it are left out, `0`
            keeps every row.
        cluster_rows: whether the rows are clustered.
        title: the title of the chart, if any.
        cmap: the color scale, see `color_scale`.
        vcenter: the value of the middle color of a diverging scale, if any;
            a center outside of the scale is not used.
        vmin: the lower end of the color scale.
        vmax: the upper end of the color scale.

    Returns:
        The chart.

    Raises:
        ValueError: if no row reaches the cutoff and none is undefined, or
            for a color map the renderer has not.
    """
    scale = color_scale(cmap)
    values = df.to_numpy(dtype=float)
    undefined = ~np.isfinite(values)
    magnitude = np.abs(np.where(undefined, 0.0, values))
    if cutoff > 0:
        keep = (magnitude >= cutoff).any(axis=1) | undefined.any(axis=1)
        if not keep.any():
            top = float(magnitude.max()) if values.size else 0.0
            raise ValueError(
                f"No parameter reaches the cutoff {cutoff:g} of '{label}', the largest "
                f"absolute value is {top:.3g}; choose a smaller cutoff or cutoff=0."
            )
        df = df[keep]
        values, undefined, magnitude = values[keep], undefined[keep], magnitude[keep]
    order = _order(values, cluster_rows)
    df, values = df.iloc[order], values[order]
    undefined, magnitude = undefined[order], magnitude[order]
    hidden = undefined | (magnitude < cutoff)
    n_rows, n_cols = values.shape
    rows = [str(i) for i in df.index]
    columns = [str(c) for c in df.columns]

    cell_width, cell_height = CELL_SIZE
    cell_height = max(cell_height, MIN_CELLS_HEIGHT / n_rows)
    cells = (n_cols * cell_width, n_rows * cell_height)
    axis = _widest(rows) + TICK_PAD
    left = axis + EDGE
    width = min(left + cells[0] + COLORBAR_ROOM, MAX_HEATMAP_SIZE[0])
    # the cells of a chart of the largest width are narrower
    room = (width - left - COLORBAR_ROOM) / n_cols
    rotation = _rotation(columns, room, axis + room / 2)
    bottom = _below(columns, rotation) + TICK_PAD + EDGE
    top = TITLE_ROOM if title else EDGE
    chart = Chart(width, min(top + cells[1] + bottom, MAX_HEATMAP_SIZE[1]), title=title)
    plot = chart.plot(
        x=ChartAxis(tick_rotation=rotation),
        y=ChartAxis(reverse=True),
        legend=False,
    )
    fmt = _value_format(float(magnitude.max()) if values.size else 0.0)
    text = [
        [
            "" if hide else _cell_text(float(v), fmt)
            for v, hide in zip(row, mask, strict=True)
        ]
        for row, mask in zip(values, hidden, strict=True)
    ]
    center = vcenter if vcenter is not None and vmin < vcenter < vmax else None
    plot.heatmap(
        rows,
        columns,
        np.where(undefined, np.nan, values),
        text=text,
        scale=scale,
        low=vmin,
        high=vmax,
        center=center,
    )
    # a cell below the cutoff is white, an undefined one keeps the grey of no value
    rings = [
        [(j - 0.5, i - 0.5), (j + 0.5, i - 0.5), (j + 0.5, i + 0.5), (j - 0.5, i + 0.5)]
        for i, j in np.argwhere(hidden & ~undefined)
    ]
    if rings:
        plot.polygon(rings, line="none", fill=HIDDEN_COLOR, fill_opacity=1.0)
    plot.colorbar(scale, vmin, vmax, label=label, center=center)
    return chart


def _selected(
    data: xr.DataArray, selection: Mapping[str, Any], keep: set[str]
) -> xr.DataArray:
    """Select labels and check that only the dimensions of `keep` remain.

    Args:
        data: the indices.
        selection: the labels to select, by dimension; the time of a change,
            which a timecourse has twice, selects the state after it.
        keep: the dimensions which the plot draws.

    Returns:
        The selected data.

    Raises:
        ValueError: if a dimension or a label does not exist, or if a dimension
            outside `keep` has more than one label.
    """
    # the time of a change, which a timecourse has twice: the state after it
    selected, rest = at_one_time(data, dict(selection))
    for dim, label in rest.items():
        if dim not in selected.dims:
            available = [str(d) for d in data.dims if str(d) not in keep]
            raise ValueError(
                f"Unknown selection '{dim}', the dimensions to select are {available}."
            )
        try:
            selected = selected.sel({dim: label})
        except KeyError as err:
            labels = data[dim].values.tolist()
            raise ValueError(
                f"Unknown label {label!r} of '{dim}', the labels are {labels}."
            ) from err
    for dim in selected.dims:
        if str(dim) not in keep and selected.sizes[dim] > 1:
            raise ValueError(
                f"Choose one label of '{dim}' with {dim}=..., it has "
                f"{selected.sizes[dim]}."
            )
    return selected.squeeze([d for d in selected.dims if str(d) not in keep])


def _check_observable(result: SensitivityResult, observable: str) -> None:
    """Raise a `ValueError` naming the observables if `observable` is unknown.

    Args:
        result: the sensitivity result.
        observable: the observable.

    Raises:
        ValueError: if the result has no such observable.
    """
    if observable not in result.observables:
        raise ValueError(
            f"Unknown observable '{observable}', the observables are "
            f"{result.observables}."
        )


def _check_indices(
    result: SensitivityResult,
    observable: str,
    keys: Sequence[str],
    plot: str,
    analyses: str,
) -> None:
    """Raise a `ValueError` if the result lacks an index a plot draws.

    Args:
        result: the sensitivity result.
        observable: the observable.
        keys: the indices the plot draws.
        plot: the name of the plot.
        analyses: the analyses which have the indices.

    Raises:
        ValueError: if the observable lacks one of the indices.
    """
    if not all(f"{observable}.{key}" in result for key in keys):
        raise ValueError(
            f"{plot} draws {' and '.join(keys)} of {analyses} analysis, which the "
            f"result of '{result.method or 'an unknown method'}' does not have for "
            f"'{observable}'; see plot_heatmap."
        )


def plot_heatmap(
    result: SensitivityResult,
    index: str,
    *,
    observables: Sequence[str] | None = None,
    cutoff: float | None = None,
    cluster_rows: bool = True,
    title: str | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    **selection: Any,
) -> Chart:
    """Draw an index of the scalar observables over the parameters.

    The parameters whose finite values are all below the cutoff are left out;
    the cells below it are white. An undefined value (`NaN`, e.g. a normalized
    index of a zero reference) is never dropped by it: its row stays and its
    cells are light grey. The rows are clustered (an axis with a single entry
    is not), the chart grows with the number of parameters and observables
    and the colorbar is labelled with the index and its unit, e.g.
    `mu_star [mM]`. All observables share one color scale, so the observables
    with the largest values dominate an index with a unit (`raw`, `mu`,
    `mu_star`, `sigma`) when their units differ; `normalized` compares the
    observables.

    Args:
        result: the sensitivity result.
        index: the index, e.g. `ST` or `normalized`.
        observables: the observables, every scalar one with the index by default.
        cutoff: the cutoff, absolute in the unit of the index; `None` is
            `DEFAULT_CUTOFF` (0.1) for the dimensionless indices
            (`normalized`, `S1`, `ST`, ...) and no cutoff for the indices with
            a unit, `0` keeps every parameter.
        cluster_rows: whether the parameters are clustered.
        title: the title of the chart, none by default.
        cmap: the color scale, see `color_scale`; `coolwarm` centered at 0
            for the signed indices (`raw`, `normalized`, `mu`), else `viridis`.
        vmin: the lower end of the color scale: `-2` for `normalized`, the
            largest magnitude of the data (negated) for the other signed
            indices, else `0`.
        vmax: the upper end of the color scale: `2` for `normalized`, the largest
            magnitude of the data for the other signed indices, `1` for `S1` and
            `ST`, else the largest value of the data.
        **selection: the label of every other dimension, e.g. `dose=0`.

    Returns:
        The chart.

    Raises:
        ValueError: if a dimension with several labels is not chosen, an
            index, an observable, a dimension or a label does not exist, no
            parameter reaches the cutoff, or the color scale is unknown.
    """
    for observable in observables or ():
        _check_observable(result, observable)
    try:
        stacked = result.index(index, observables)
    except KeyError as err:
        raise ValueError(
            f"No scalar observable has the index '{index}', the observables are "
            f"{result.observables}."
        ) from err
    data = _selected(stacked, selection, {PARAMETER, "observable"})
    df = pd.DataFrame(
        data.transpose(PARAMETER, "observable").values,
        index=data[PARAMETER].values,
        columns=data["observable"].values,
    )
    finite = np.abs(df.to_numpy(dtype=float))
    top = float(finite[np.isfinite(finite)].max()) if np.isfinite(finite).any() else 1.0
    top = top if top > 0 else 1.0
    signed = index in SIGNED_INDICES
    if signed:
        extent = 2.0 if index == "normalized" else top
        low, high = -extent, extent
        vcenter: float | None = 0.0
    else:
        low, high = 0.0, 1.0 if index in BOUNDED_INDICES else top
        vcenter = None
    if cutoff is None:
        cutoff = DEFAULT_CUTOFF if index in DIMENSIONLESS_INDICES else 0.0
    units = {result.units.get(f"{o}.{index}", "") for o in df.columns}
    unit = next(iter(units)) if len(units) == 1 else None
    return _heatmap(
        df,
        label=unit_label(index, unit),
        cutoff=cutoff,
        cluster_rows=cluster_rows,
        title=title,
        cmap=cmap or (SIGNED_SCALE if signed else BOUNDED_SCALE),
        vcenter=vcenter,
        vmin=low if vmin is None else vmin,
        vmax=high if vmax is None else vmax,
    )


def plot_indices(
    result: SensitivityResult,
    observable: str,
    *,
    title: str | None = None,
    **selection: Any,
) -> Chart:
    """Draw the S1 and ST indices of an observable with their intervals.

    The parameter names are horizontal, rotated by 45 degrees when they do
    not fit side by side and vertical when the first of them would reach left
    of the chart; a parameter without indices is marked `n/a`.

    Args:
        result: the result of a Sobol or FAST analysis.
        observable: the observable.
        title: the title of the plot, the observable by default.
        **selection: the label of every other dimension, e.g. `dose=0`.

    Returns:
        The chart, a pair of bars per parameter.

    Raises:
        ValueError: if a dimension with several labels is not chosen, the
            observable, a dimension or a label does not exist, or the result
            has no `S1` and `ST` (not a Sobol or FAST analysis).
    """
    _check_observable(result, observable)
    _check_indices(result, observable, ("S1", "ST"), "plot_indices", "a Sobol or FAST")
    keep = {PARAMETER}
    values = {
        key: _selected(result[f"{observable}.{key}"], selection, keep)
        for key in ("S1", "ST", "S1_conf", "ST_conf")
        if f"{observable}.{key}" in result
    }
    parameters = [str(p) for p in result.parameters]
    n = len(parameters)
    x = np.arange(n, dtype=float)
    width = max(432.0, 43.2 * n + 144.0)
    room = (width - 100.0) / max(n, 1)
    rotation = _rotation(parameters, room, 60.0 + room / 2)
    chart = Chart(width, 288.0 + _below(parameters, rotation))
    plot = chart.plot(
        title=title or observable,
        x=ChartAxis(
            label="parameter",
            scale="category",
            categories=parameters,
            tick_rotation=rotation,
        ),
        y=ChartAxis(
            label=INDEX_NAMES.get(result.method or "", "sensitivity index"), grid=True
        ),
    )
    bar = 0.4
    for offset, key, color in ((-bar / 2, "S1", S1_COLOR), (bar / 2, "ST", ST_COLOR)):
        if key not in values:
            continue
        conf = values.get(f"{key}_conf")
        plot.bar(
            x + offset,
            values[key].values,
            err=None if conf is None else conf.values,
            bar_width=bar,
            color=color,
            label=key,
        )
    undefined = np.all(
        [np.isnan(v.values) for k, v in values.items() if k in ("S1", "ST")], axis=0
    )
    # the note a little above the baseline, which its text would cross
    heights = np.concatenate([np.ravel(v.values) for v in values.values()])
    top = float(np.max(heights[np.isfinite(heights)], initial=0.0))
    for position in x[undefined]:
        plot.text(
            position, 0.04 * (top or 1.0), "n/a", anchor="middle", color=NOTE_COLOR
        )
    return chart


def plot_morris(
    result: SensitivityResult,
    observable: str,
    *,
    title: str | None = None,
    **selection: Any,
) -> Chart:
    """Draw `mu_star` against `sigma` of an observable, a point per parameter.

    Both axes start at 0, so that the ratio `sigma / mu_star` reads from the
    position, and end beyond the labels of the points. A parameter whose
    effects are undefined (`NaN`) has no point; a note in the plot names it.

    Args:
        result: the result of a Morris analysis.
        observable: the observable.
        title: the title of the plot, the observable by default.
        **selection: the label of every other dimension.

    Returns:
        The chart, every point labelled with its parameter.

    Raises:
        ValueError: if a dimension with several labels is not chosen, the
            observable, a dimension or a label does not exist, or the result
            has no `mu_star` and `sigma` (not a Morris analysis).
    """
    _check_observable(result, observable)
    _check_indices(result, observable, ("mu_star", "sigma"), "plot_morris", "a Morris")
    mu_star = _selected(result[f"{observable}.mu_star"], selection, {PARAMETER})
    sigma = _selected(result[f"{observable}.sigma"], selection, {PARAMETER})
    defined = np.isfinite(mu_star.values) & np.isfinite(sigma.values)
    parameters = [str(p) for p in result.parameters]
    names = [p for p, ok in zip(parameters, defined, strict=True) if ok]
    undefined = [p for p, ok in zip(parameters, defined, strict=True) if not ok]
    xs = np.asarray(mu_star.values[defined], dtype=float)
    ys = np.asarray(sigma.values[defined], dtype=float)
    width, height = 360.0, 324.0
    # the panel without the room of the axes, which the labels of the points
    # must fit into
    panel_width, panel_height = width - 110.0, height - 100.0
    note = f"not shown, undefined (NaN): {', '.join(undefined)}" if undefined else None
    x_end = _end(xs, [_text_width(n) + 8.0 for n in names], panel_width)
    lines = 2 if note else 1
    y_end = _end(ys, [lines * TEXT_SIZE * 1.4] * len(names), panel_height)
    if note:
        y_end *= 1.0 + 1.6 * TEXT_SIZE / panel_height
    chart = Chart(width, height)
    plot = chart.plot(
        title=title or observable,
        x=ChartAxis(
            label="mu_star",
            unit=_unit(result.units.get(f"{observable}.mu_star")),
            min=0.0,
            max=x_end,
            grid=True,
        ),
        y=ChartAxis(
            label="sigma",
            unit=_unit(result.units.get(f"{observable}.sigma")),
            min=0.0,
            max=y_end,
            grid=True,
        ),
        legend=False,
    )
    if names:
        plot.curve(
            xs,
            ys,
            line="none",
            marker="circle",
            size=7.0,
            color=S1_COLOR,
            marker_edge=INTERVAL_COLOR,
        )
    dx, dy = 5.0 * x_end / panel_width, 5.0 * y_end / panel_height
    for name, x, y in zip(names, xs, ys, strict=True):
        plot.text(x + dx, y + dy, name)
    if note:
        plot.text(
            dx, y_end * (1.0 - 1.4 * TEXT_SIZE / panel_height), note, color=NOTE_COLOR
        )
    return chart


def _unit(unit: str | None) -> str | None:
    """Get the unit of an axis, `None` for no unit and a dimensionless one."""
    return unit if unit and unit != "dimensionless" else None


def _end(values: np.ndarray, extents: Sequence[float], room: float) -> float:
    """Get the upper end of an axis from 0 which holds the values and their labels.

    A label takes its extent in pixels beyond its value; the end `e` holds it
    when `value + extent * e / room <= e`, the value at a share of the axis
    which leaves the room of its label.

    Args:
        values: the values, not negative.
        extents: the pixels of the label of every value beyond it.
        room: the pixels of the axis.

    Returns:
        The end, at least 10 % beyond the largest value and 1 without a value.
    """
    finite = [v for v in values.tolist() if math.isfinite(v)]
    top = max(finite, default=0.0)
    end = 1.1 * top if top > 0 else 1.0
    for value, extent in zip(values.tolist(), extents, strict=True):
        share = 1.0 - (extent + 6.0) / room
        if share > 0.1 and math.isfinite(value):
            end = max(end, value / share)
    return end
