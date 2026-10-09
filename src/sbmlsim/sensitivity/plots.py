"""The plots of a sensitivity result.

Every plot is a function which returns a figure and saves it only with a
`path`; the figure is not held by pyplot and opens no window, a caller which
wants to display it does so itself.

- `plot_heatmap`: an index of the scalar observables over the parameters,
  clustered;
- `plot_indices`: the bars of `S1` and `ST` of an observable with their
  intervals;
- `plot_morris`: `mu_star` against `sigma` of an observable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation, Text
from matplotlib.ticker import ScalarFormatter
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from sbmlsim.sensitivity.result import (
    DIMENSIONLESS_INDICES,
    PARAMETER,
    SensitivityResult,
)

SIGNED_INDICES = ("raw", "normalized", "mu")
"""The indices which have a sign, drawn on a diverging color map around 0."""

BOUNDED_INDICES = ("S1", "ST")
"""The indices in the interval 0 to 1."""

INDEX_NAMES = {"sobol": "Sobol index", "fast": "FAST index"}
"""The label of the axis of the indices of a method."""

DEFAULT_CUTOFF = 0.1
"""The cutoff of the heatmap of a dimensionless index, see `plot_heatmap`."""

CELL_SIZE = (0.9, 0.45)
"""The width and the height of a cell of the heatmap in inches."""

MIN_CELLS_HEIGHT = 1.8
"""The smallest height of the cells of a heatmap in inches, room for its colorbar."""

MAX_HEATMAP_SIZE = (24.0, 40.0)
"""The largest width and height of the figure of a heatmap in inches."""

MAX_COLORBAR_HEIGHT = 5.0
"""The largest height of the colorbar of a heatmap in inches."""

COLORBAR_WIDTH = 0.18
"""The width of the colorbar of a heatmap in inches."""

LABEL_SIZE = 12.0
"""The font size of the labels of a heatmap."""

VALUE_SIZE = 10.0
"""The font size of the values in the cells of a heatmap."""

TITLE_SIZE = 14.0
"""The font size of the title of a heatmap."""

UNDEFINED_COLOR = "lightgrey"
"""The color of a cell whose index is undefined (`NaN`)."""

S1_COLOR = "#2a78d6"
"""The color of the bars of `S1`."""

ST_COLOR = "#eb6834"
"""The color of the bars of `ST`, which contrasts with `S1` and the intervals."""

INTERVAL_COLOR = "black"
"""The color of the intervals of the bars and of their edges."""


def _overlapping(texts: Sequence[Text], pad: float = 2.0) -> bool:
    """Check whether two neighbouring texts overlap or come closer than `pad` points.

    Args:
        texts: the texts in their order along the axis.
        pad: the smallest gap in points.

    Returns:
        Whether two neighbours overlap.
    """
    shown = [t for t in texts if t.get_text() and t.get_visible()]
    if len(shown) < 2:
        return False
    figure = shown[0].get_figure(root=True)
    gap = pad * (figure.dpi if figure is not None else 72.0) / 72.0
    boxes = [t.get_window_extent() for t in shown]
    return any(b.x0 < a.x1 + gap and a.x0 < b.x1 + gap for a, b in pairwise(boxes))


def _rotate_if_crowded(figure: Figure, labels: Sequence[Text]) -> None:
    """Rotate the tick labels of an axis by 45 degrees if they do not fit side by side.

    Args:
        figure: the figure, laid out before the labels are measured.
        labels: the tick labels along the horizontal axis.
    """
    figure.draw_without_rendering()
    if _overlapping(labels):
        for label in labels:
            label.set_rotation(45)
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")


def _centered(cmap: Colormap, vcenter: float, vmin: float, vmax: float) -> Colormap:
    """Get the part of a diverging color map which puts its center at `vcenter`.

    The colors are the ones of a scale symmetric around `vcenter` which
    reaches the farther end of `vmin` and `vmax`, so that `vcenter` has the
    middle color of the map.

    Args:
        cmap: the diverging color map.
        vcenter: the value of the middle color.
        vmin: the lower end of the scale.
        vmax: the upper end of the scale.

    Returns:
        The color map, `cmap` itself for a scale symmetric around `vcenter`.
    """
    half = max(vmax - vcenter, vcenter - vmin)
    if half <= 0:
        return cmap
    low = 0.5 + (vmin - vcenter) / (2.0 * half)
    high = 0.5 + (vmax - vcenter) / (2.0 * half)
    if np.isclose(low, 0.0) and np.isclose(high, 1.0):
        return cmap
    return ListedColormap(cmap(np.linspace(low, high, 256)), name=f"{cmap.name}_part")


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


def _text_color(rgba: tuple[float, float, float, float]) -> str:
    """Get the color of a text which reads on a cell, black or white.

    Args:
        rgba: the color of the cell.

    Returns:
        `black` on a light cell, `white` on a dark one.
    """
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "black" if luminance > 0.45 else "white"


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
    path: str | Path | None,
    dpi: int,
) -> Figure:
    """Draw a clustered heatmap of a table, a row per parameter and a column per observable.

    A row is kept if one of its values reaches the cutoff or one is undefined
    (`NaN` or infinite); a cell below the cutoff is white, an undefined cell light grey,
    and every other cell carries its value. The rows are clustered on the
    values with `NaN` as 0. The cells have a fixed size up to the largest
    figure, the parameters are on the left, the observables below (rotated by
    45 degrees when they do not fit side by side) and the colorbar with its
    label on the right; constrained layout keeps every text clear of the cells.

    Args:
        df: the values, a row per parameter and a column per observable.
        label: the label of the colorbar, the index and its unit.
        cutoff: rows whose finite values are all below it are left out, `0`
            keeps every row.
        cluster_rows: whether the rows are clustered.
        title: the title of the figure, if any.
        cmap: the name of the color map.
        vcenter: the value of the middle color of a diverging map, if any.
        vmin: the lower end of the color scale.
        vmax: the upper end of the color scale.
        path: where the figure is saved, if given.
        dpi: the resolution of the saved figure.

    Returns:
        The figure.

    Raises:
        ValueError: if no row reaches the cutoff and none is undefined.
    """
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

    cell_width, cell_height = CELL_SIZE
    cell_height = max(cell_height, MIN_CELLS_HEIGHT / n_rows)
    cells = (n_cols * cell_width, n_rows * cell_height)
    figure = Figure(figsize=(cells[0] + 3.0, cells[1] + 2.0), layout="constrained")
    ax = figure.add_subplot(1, 1, 1, label="heatmap")
    colors = colormaps[cmap]
    if vcenter is not None:
        colors = _centered(colors, vcenter, vmin, vmax)
    norm = Normalize(vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(
        np.arange(n_cols + 1),
        np.arange(n_rows + 1),
        np.ma.masked_array(values, mask=hidden),
        cmap=colors,
        norm=norm,
    )
    fmt = _value_format(float(magnitude.max()) if values.size else 0.0)
    for row, col in np.argwhere(undefined):
        ax.add_patch(
            Rectangle((col, row), 1, 1, facecolor=UNDEFINED_COLOR, edgecolor="none")
        )
    for row, col in np.argwhere(~hidden):
        value = float(values[row, col])
        ax.text(
            col + 0.5,
            row + 0.5,
            _cell_text(value, fmt),
            ha="center",
            va="center",
            size=VALUE_SIZE,
            color=_text_color(colors(norm(value))),
        )
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks(np.arange(n_cols) + 0.5, [str(c) for c in df.columns])
    ax.set_yticks(np.arange(n_rows) + 0.5, [str(i) for i in df.index])
    ax.tick_params(length=0, labelsize=LABEL_SIZE)
    for spine in ax.spines.values():
        spine.set_color(UNDEFINED_COLOR)
    height = min(cells[1], MAX_COLORBAR_HEIGHT)
    shown = values[~hidden]
    below = bool(shown.size) and float(shown.min()) < vmin
    above = bool(shown.size) and float(shown.max()) > vmax
    colorbar = figure.colorbar(
        mesh,
        ax=ax,
        shrink=height / cells[1],
        aspect=height / COLORBAR_WIDTH,
        pad=0.1 / cells[0],
        # an arrow at an end of the scale which values pass
        extend={
            (False, False): "neither",
            (True, False): "min",
            (False, True): "max",
            (True, True): "both",
        }[(below, above)],
    )
    colorbar.set_label(label, size=LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=LABEL_SIZE - 1)
    colorbar.outline.set_visible(False)
    formatter = colorbar.ax.yaxis.get_major_formatter()
    if isinstance(formatter, ScalarFormatter):
        formatter.set_useMathText(True)
    # the power of ten above the colorbar extends to the right, not over the cells
    colorbar.ax.yaxis.get_offset_text().set_horizontalalignment("left")
    heading = (
        figure.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold")
        if title
        else None
    )
    figure.draw_without_rendering()
    _fit(figure, ax, cells, heading)
    # the rotation follows the width of a cell in the final, possibly capped figure
    widest = max(t.get_window_extent().width for t in ax.get_xticklabels())
    final_width = ax.get_window_extent().width / figure.dpi / n_cols
    if widest / figure.dpi > final_width - 0.1:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_horizontalalignment("right")
            tick.set_rotation_mode("anchor")
        _fit(figure, ax, cells, heading)
    if path:
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return figure


def _fit(
    figure: Figure, ax: Axes, cells: tuple[float, float], heading: Text | None
) -> None:
    """Size a figure so that its heatmap has the size of its cells and its title fits.

    Constrained layout keeps the decorations (labels, colorbar, title) at
    their size in inches, so the difference between the size of the axes and
    the size of the cells is the change of the size of the figure. The figure
    is at most `MAX_HEATMAP_SIZE` and at least as wide as its title.

    Args:
        figure: the figure with constrained layout.
        ax: the axes of the heatmap.
        cells: the width and the height of the cells in inches.
        heading: the title of the figure, if any.
    """
    for _ in range(2):
        figure.draw_without_rendering()
        width, height = figure.get_size_inches()
        box = ax.get_position()
        width += cells[0] - box.width * width
        height += cells[1] - box.height * height
        if heading is not None:
            width = max(width, heading.get_window_extent().width / figure.dpi + 0.3)
        figure.set_size_inches(
            min(width, MAX_HEATMAP_SIZE[0]), min(height, MAX_HEATMAP_SIZE[1])
        )
    figure.draw_without_rendering()


def _selected(
    data: xr.DataArray, selection: Mapping[str, Any], keep: set[str]
) -> xr.DataArray:
    """Select labels and check that only the dimensions of `keep` remain.

    Args:
        data: the indices.
        selection: the labels to select, by dimension.
        keep: the dimensions which the plot draws.

    Returns:
        The selected data.

    Raises:
        ValueError: if a dimension or a label does not exist, or if a dimension
            outside `keep` has more than one label.
    """
    selected = data
    for dim, label in selection.items():
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


def _label(name: str, unit: str | None) -> str:
    """Get an axis label with the unit if one is known.

    Args:
        name: the name.
        unit: the unit, `None`, `""` or `dimensionless` for none.

    Returns:
        The label, e.g. `mu_star [mM]`.
    """
    return f"{name} [{unit}]" if unit and unit != "dimensionless" else name


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
    path: str | Path | None = None,
    dpi: int = 300,
    **selection: Any,
) -> Figure:
    """Draw an index of the scalar observables over the parameters.

    The parameters whose finite values are all below the cutoff are left out;
    the cells below it are white. An undefined value (`NaN`, e.g. a normalized
    index of a zero reference) is never dropped by it: its row stays and its
    cells are light grey. The rows are clustered (an axis with a single entry
    is not), the figure grows with the number of parameters and observables
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
        title: the title of the figure, none by default.
        cmap: the color map; `seismic` centered at 0 for the signed indices
            (`raw`, `normalized`, `mu`), else `Reds` from 0.
        vmin: the lower end of the color scale: `-2` for `normalized`, the
            largest magnitude of the data (negated) for the other signed
            indices, else `0`.
        vmax: the upper end of the color scale: `2` for `normalized`, the largest
            magnitude of the data for the other signed indices, `1` for `S1` and
            `ST`, else the largest value of the data.
        path: where the figure is saved, if given.
        dpi: the resolution of the saved figure.
        **selection: the label of every other dimension, e.g. `dose=0`.

    Returns:
        The figure.

    Raises:
        ValueError: if a dimension with several labels is not chosen, an
            index, an observable, a dimension or a label does not exist, or no
            parameter reaches the cutoff.
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
        label=_label(index, unit),
        cutoff=cutoff,
        cluster_rows=cluster_rows,
        title=title,
        cmap=cmap or ("seismic" if signed else "Reds"),
        vcenter=vcenter,
        vmin=low if vmin is None else vmin,
        vmax=high if vmax is None else vmax,
        path=path,
        dpi=dpi,
    )


def plot_indices(
    result: SensitivityResult,
    observable: str,
    *,
    title: str | None = None,
    path: str | Path | None = None,
    dpi: int = 300,
    **selection: Any,
) -> Figure:
    """Draw the S1 and ST indices of an observable with their intervals.

    The parameter names are horizontal and rotated by 45 degrees only when
    they do not fit side by side.

    Args:
        result: the result of a Sobol or FAST analysis.
        observable: the observable.
        title: the title of the figure, the observable by default.
        path: where the figure is saved, if given.
        dpi: the resolution of the saved figure.
        **selection: the label of every other dimension, e.g. `dose=0`.

    Returns:
        The figure, a pair of bars per parameter.

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
    parameters = result.parameters
    x = np.arange(len(parameters))
    width = 0.4
    figure = Figure(
        figsize=(max(6.0, 0.6 * len(parameters) + 2), 4), layout="constrained"
    )
    ax = figure.subplots()
    for offset, key, color in (
        (-width / 2, "S1", S1_COLOR),
        (width / 2, "ST", ST_COLOR),
    ):
        if key not in values:
            continue
        ax.bar(
            x + offset,
            values[key].values,
            width,
            label=key,
            color=color,
            edgecolor=INTERVAL_COLOR,
            linewidth=0.8,
            yerr=values[f"{key}_conf"].values if f"{key}_conf" in values else None,
            error_kw={"ecolor": INTERVAL_COLOR, "elinewidth": 1.2, "capthick": 1.2},
            capsize=4,
        )
    ax.set_xticks(x, parameters)
    ax.set_xlim(-0.7, len(parameters) - 0.3)
    undefined = np.all(
        [np.isnan(v.values) for k, v in values.items() if k in ("S1", "ST")], axis=0
    )
    for position in x[undefined]:
        ax.annotate(
            "n/a",
            (position, 0),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=INTERVAL_COLOR,
        )
    ax.set_xlabel("Parameter")
    ax.set_ylabel(INDEX_NAMES.get(result.method, "Sensitivity index"))
    ax.set_title(title or observable)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.legend()
    _rotate_if_crowded(figure, ax.get_xticklabels())
    if path:
        figure.savefig(path, dpi=dpi)
    return figure


def _keep_labels_inside(figure: Figure, ax: Axes, notes: Sequence[Annotation]) -> None:
    """Widen the upper limits until no label touches or crosses the frame.

    The lower limits are 0. A label keeps its offset in points from its
    point, so the upper limit which keeps it inside follows from the position
    of the point and the extent of the label in pixels; the layout changes
    with the limits, so it is measured again.

    Args:
        figure: the figure with constrained layout.
        ax: the axes, whose lower limits are 0.
        notes: the labels of the points.
    """
    pad = 6.0
    for _ in range(3):
        figure.draw_without_rendering()
        frame = ax.get_window_extent()
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        right, top = x1, y1
        for note in notes:
            box = note.get_window_extent()
            px, py = ax.transData.transform(note.xy)
            room_x = frame.x1 - pad - (box.x1 - px)
            room_y = frame.y1 - pad - (box.y1 - py)
            if box.x1 > frame.x1 - pad and room_x > frame.x0:
                right = max(
                    right, x0 + (note.xy[0] - x0) * frame.width / (room_x - frame.x0)
                )
            if box.y1 > frame.y1 - pad and room_y > frame.y0:
                top = max(
                    top, y0 + (note.xy[1] - y0) * frame.height / (room_y - frame.y0)
                )
        if right == x1 and top == y1:
            return
        ax.set_xlim(x0, right)
        ax.set_ylim(y0, top)


def plot_morris(
    result: SensitivityResult,
    observable: str,
    *,
    title: str | None = None,
    path: str | Path | None = None,
    dpi: int = 300,
    **selection: Any,
) -> Figure:
    """Draw `mu_star` against `sigma` of an observable, a point per parameter.

    Both axes start at 0, so that the ratio `sigma / mu_star` reads from the
    position, and use a common power of ten for small or large values. A
    parameter whose effects are undefined (`NaN`) has no point; a note below
    the axes names it.

    Args:
        result: the result of a Morris analysis.
        observable: the observable.
        title: the title of the figure, the observable by default.
        path: where the figure is saved, if given.
        dpi: the resolution of the saved figure.
        **selection: the label of every other dimension.

    Returns:
        The figure, every point labelled with its parameter.

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
    names = [p for p, ok in zip(result.parameters, defined, strict=True) if ok]
    xs, ys = mu_star.values[defined], sigma.values[defined]
    figure = Figure(figsize=(5, 4.5), layout="constrained")
    ax = figure.subplots()
    # a point on an axis at 0 is drawn whole
    ax.scatter(
        xs, ys, color=S1_COLOR, edgecolor=INTERVAL_COLOR, zorder=3, clip_on=False
    )
    notes = [
        ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points")
        for name, x, y in zip(names, xs, ys, strict=True)
    ]
    ax.margins(0.1)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.ticklabel_format(style="sci", scilimits=(-3, 4), axis="both", useMathText=True)
    ax.set_xlabel(_label("mu_star", result.units.get(f"{observable}.mu_star")))
    ax.set_ylabel(_label("sigma", result.units.get(f"{observable}.sigma")))
    ax.set_title(title or observable)
    ax.grid(True)
    ax.set_axisbelow(True)
    undefined = [p for p, ok in zip(result.parameters, defined, strict=True) if not ok]
    if undefined:
        figure.supxlabel(
            f"not shown, undefined (NaN): {', '.join(undefined)}",
            x=0.02,
            ha="left",
            fontsize=9,
            color="dimgrey",
        )
    _keep_labels_inside(figure, ax, notes)
    if path:
        figure.savefig(path, dpi=dpi)
    return figure
