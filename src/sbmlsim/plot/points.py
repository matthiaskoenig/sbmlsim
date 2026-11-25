"""The lines of a curve over the points of the dimensions of a scan.

A curve names the dimensions of its data it draws one line per point of
(`over`); after broadcasting by name its data must leave at most one more
dimension, the axis a line is drawn along: the time of a timecourse, the
points of a ragged result, the rows of a dataset or the dimension of the
values a scan sets (a value per simulation over a dimension). The lines are in
the order of the labels of the `over` dimensions, the last one fastest. Their
colours follow the first `over` dimension, their line styles the second; a
legend entry names its point by the value and unit of the target its
dimension changes, else by its label. Everything of a point is found by its
label in the dimension, so a curve of selected points (`Data(sel=...)`) names
and colours the points it draws. The colour of a point is the colour map at its
value, through the norm of the colour bar (logarithmic for positive values
spanning more than a factor of `LOG_RATIO`); a dimension which does not change
one target is coloured by position, with the labels on a discrete colour bar. A
band reduces a dimension of draws to two quantiles and the median, computed
when the figure is drawn; the draws of a ragged result, whose simulations keep
their own time points (the steps of the integrator), are interpolated onto the
time points of a figure first, a few hundred which follow the density of their
time points (`onto_grid`, `sbmlsim.result.timecourse.plot_times`), and its
legend entry is its area, `<name> (<low>-<high> %)`.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from sbmlsim.data import ROW
from sbmlsim.engine import import_core
from sbmlsim.plot.colors import normalize_color
from sbmlsim.plot.padding import without_padding
from sbmlsim.result.scan import POINT, TIME
from sbmlsim.result.timecourse import PLOT_POINTS, plot_times
from sbmlsim.simulation import Dimension
from sbmlsim.units import Quantity

#: the line styles of the points of a second `over` dimension
LINE_STYLES: tuple[str, ...] = ("-", "--", ":", "-.")

#: the number of points from which a dimension gets a colour bar instead of
#: legend entries
COLORBAR_FROM = 11

#: the colour map of the points of a curve whose style sets no colour
COLORMAP = "viridis"

#: the ratio of the largest to the smallest value of a dimension above which its
#: colours and its colour bar follow a logarithmic norm, if all are positive
LOG_RATIO = 100.0

#: the part of the colour map the points use: its light yellow end is hard to
#: see on white
_COLORMAP_END = 0.85

#: the most points of a categorical colour bar, a band per point; a dimension
#: of more points gets a colour bar of their positions
MAX_BANDS = 256

#: how much white the lightest shade of a colour has
_LIGHTEST = 0.65


@dataclass(frozen=True)
class Line:
    """One line of a curve, see `curve_lines`.

    Attributes:
        point: the label of every `over` dimension, in their order.
        index: the position of the point along every `over` dimension.
        x: the x values along the axis, without the padding.
        y: the y values.
        xerr: the x errors, `None` without.
        yerr: the y errors, `None` without.
    """

    point: tuple[Any, ...]
    index: tuple[int, ...]
    x: Any
    y: Any
    xerr: Any = None
    yerr: Any = None


@dataclass(frozen=True)
class BandLine:
    """One band, see `band_lines`.

    Attributes:
        point: the label of every `over` dimension, in their order.
        index: the position of the point along every `over` dimension.
        x: the x values along the axis.
        low: the lower quantile.
        median: the median.
        high: the upper quantile.
    """

    point: tuple[Any, ...]
    index: tuple[int, ...]
    x: Any
    low: Any
    median: Any
    high: Any


def curve_lines(
    sid: str,
    over: Sequence[str],
    x: xr.DataArray,
    y: xr.DataArray,
    xerr: xr.DataArray | None = None,
    yerr: xr.DataArray | None = None,
) -> list[Line]:
    """Split the data of a curve into one line per point of its `over` dimensions.

    Args:
        sid: the id of the curve, for the errors.
        over: the dimensions with a line per point, in the order of the lines.
        x: the x data.
        y: the y data.
        xerr: the x errors, `None` without.
        yerr: the y errors, `None` without.

    Returns:
        The lines, without the padding of a ragged result.

    Raises:
        ValueError: if two arrays have different coordinates of a dimension, an
            `over` dimension is not one of the data, or more than one other
            dimension is left.
    """
    full, dims = _broadcast(sid, "curve", [x, y, xerr, yerr])
    missing = [d for d in over if d not in dims]
    if missing:
        raise ValueError(
            f"The curve '{sid}' draws a line per point of {missing}, which its data "
            f"has not: {dims}."
        )
    axis = [d for d in dims if d not in over]
    if len(axis) > 1:
        scan = [d for d in axis if d not in (TIME, POINT, ROW)] or axis
        raise ValueError(
            f"The curve '{sid}' has the dimensions {dims}; a curve draws a line along "
            f"one of them: name {scan} in over= or select one label with "
            f"Data(sel=...)."
        )
    if not over:
        # one line: the values as they are, without the copies of `isel`
        values = [None if a is None else np.asarray(a.values) for a in full]
        lx, ly, lxerr, lyerr = without_padding(*values)
        return [Line(point=(), index=(), x=lx, y=ly, xerr=lxerr, yerr=lyerr)]
    ordered = [None if a is None else a.transpose(*over, *axis) for a in full]
    labels = _labels(ordered, over)
    # the values are read once and a line is a numpy index of them: the `isel`
    # of an array per line and array built a DataArray for every line, which
    # was most of the time of the figures of a scan over draws (sbmlsim#664)
    data = [None if a is None else np.asarray(a.values) for a in ordered]
    lines: list[Line] = []
    for index in itertools.product(*(range(len(labels[d])) for d in over)):
        position = dict(zip(over, index, strict=True))
        values = [None if a is None else a[index] for a in data]
        lx, ly, lxerr, lyerr = without_padding(*values)
        lines.append(
            Line(
                point=tuple(labels[d][k] for d, k in position.items()),
                index=tuple(index),
                x=lx,
                y=ly,
                xerr=lxerr,
                yerr=lyerr,
            )
        )
    return lines


def band_label(name: str | None, quantiles: tuple[float, float]) -> str:
    """Get the legend label of a band, its name and its quantiles in percent.

    Args:
        name: the name of the band, of a point of it with `over`; none or
            empty for a band without a name.
        quantiles: the lower and the upper quantile.

    Returns:
        `<name> (<low>-<high> %)`, e.g. `C (5-95 %)`, or `<low>-<high> %`
        without a name.
    """
    low, high = (f"{100 * q:g}" for q in quantiles)
    return f"{name} ({low}-{high} %)" if name else f"{low}-{high} %"


def nan_quantile(values: np.ndarray, q: float) -> np.ndarray:
    """Get a quantile along the last axis, ignoring NaN, vectorized.

    The linear interpolation of `numpy.nanquantile`, whose reduction of an
    array with a NaN runs in python once per position of the other axes; the
    values are sorted once (NaN last) and the quantile is taken from the
    finite values of every position.

    Args:
        values: the values, reduced along the last axis.
        q: the quantile in `[0, 1]`.

    Returns:
        The quantile of every position of the other axes, `NaN` where every
        value is `NaN`.
    """
    ordered = np.sort(np.asarray(values, dtype=float), axis=-1)
    n = np.isfinite(ordered).sum(axis=-1)
    position = q * np.maximum(n - 1, 0)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, np.maximum(n - 1, 0))
    low = np.take_along_axis(ordered, lower[..., np.newaxis], axis=-1)[..., 0]
    high = np.take_along_axis(ordered, upper[..., np.newaxis], axis=-1)[..., 0]
    result = low + (position - lower) * (high - low)
    return np.where(n > 0, result, np.nan)


def onto_grid(
    sid: str, x: xr.DataArray, y: xr.DataArray, n: int = PLOT_POINTS
) -> tuple[xr.DataArray, xr.DataArray]:
    """Interpolate the data of a band of a ragged result onto the time points of a figure.

    In the ragged layout every simulation keeps its own time points (`_point`,
    padded with `NaN`), so the quantiles of a band at a position of `_point`
    would mix times. Every simulation is interpolated linearly onto `n` time
    points which follow the density of the time points of all simulations
    (`plot_times`), with `NaN` outside of its own range; the times of the
    changes, which a simulation has twice (the states before and after it),
    are among them and give the state after the change, as
    `ScanResult.interpolate` does.

    Args:
        sid: the id of the band, for the errors.
        x: the x data, the time over `(*dims, _point)`.
        y: the y data over `(*dims, _point)`.
        n: the number of time points.

    Returns:
        The time points as x over `time` and the y data over `(*dims, time)`.

    Raises:
        ValueError: if x of a simulation does not increase.
    """
    x, y = xr.broadcast(x, y)
    dims = [str(d) for d in y.dims if d != POINT]
    xs = np.asarray(x.transpose(*dims, POINT).values, dtype=float)
    ys = np.asarray(y.transpose(*dims, POINT).values, dtype=float)
    shape = xs.shape[:-1]
    xs = xs.reshape(-1, xs.shape[-1])
    ys = ys.reshape(-1, ys.shape[-1])
    simulations = []
    changes: set[float] = set()
    for xk, yk in zip(xs, ys, strict=True):
        finite = np.isfinite(xk)
        xp, yp = xk[finite], yk[finite]
        if np.any(np.diff(xp) < 0):
            raise ValueError(
                f"The band '{sid}' interpolates a ragged result onto the time "
                f"points of a figure, its x values have to increase, e.g. the time."
            )
        # the state after a change at its time, the last of equal times
        last = np.append(xp[1:] != xp[:-1], True) if xp.size else xp.astype(bool)
        changes.update(xp[:-1][xp[1:] == xp[:-1]].tolist())
        simulations.append((xp[last], yp[last]))
    grid = plot_times(xs, n=n, keep=sorted(changes))
    values = np.full((xs.shape[0], grid.size), np.nan)
    for k, (xp, yp) in enumerate(simulations):
        if xp.size:
            values[k] = np.interp(grid, xp, yp, left=np.nan, right=np.nan)
    coords = {d: y.coords[d] for d in dims if d in y.coords}
    return (
        xr.DataArray(grid, dims=(TIME,)),
        xr.DataArray(
            values.reshape(*shape, grid.size), dims=(*dims, TIME), coords=coords
        ),
    )


def band_lines(
    sid: str,
    over: Sequence[str],
    across: str,
    quantiles: tuple[float, float],
    x: xr.DataArray,
    y: xr.DataArray,
) -> list[BandLine]:
    """Reduce the dimension `across` of the data of a band to quantiles and the median.

    Args:
        sid: the id of the band, for the errors.
        over: the dimensions with a band per point.
        across: the dimension of the draws, which is reduced.
        quantiles: the lower and the upper quantile, in `[0, 1]`.
        x: the x data.
        y: the y data.

    Returns:
        One band per point of `over`.

    Raises:
        ValueError: if the data has not the dimension `across`, or leaves more
            than one dimension besides `over` and `across`; for a ragged result
            see `onto_grid`.
    """
    if POINT in y.dims or POINT in x.dims:
        x, y = onto_grid(sid, x, y)
    if across not in y.dims:
        raise ValueError(
            f"The band '{sid}' reduces the dimension '{across}', which its data has "
            f"not: {[str(d) for d in y.dims]}."
        )
    # the quantiles over the draws, ignoring NaN; a point at which every draw
    # is NaN (all simulations failed) has no quantile and is NaN
    low, median, high = (
        xr.apply_ufunc(
            nan_quantile,
            y,
            input_core_dims=[[across]],
            kwargs={"q": q},
        )
        for q in (quantiles[0], 0.5, quantiles[1])
    )
    lows = curve_lines(sid, over, x, low)
    medians = curve_lines(sid, over, x, median)
    highs = curve_lines(sid, over, x, high)
    return [
        BandLine(
            point=lo.point, index=lo.index, x=lo.x, low=lo.y, median=me.y, high=hi.y
        )
        for lo, me, hi in zip(lows, medians, highs, strict=True)
    ]


def point_colors(n: int, color: str | None) -> list[str]:
    """Get the colours of the points of a dimension.

    Args:
        n: the number of points.
        color: the colour of the style of the curve, `None` without.

    Returns:
        Shades of the colour from light to the colour itself, or colours of the
        colour scale `COLORMAP` of sbmlsim-core, as hex strings; one point
        keeps the colour.
    """
    if color is not None and n == 1:
        return [_hex(_rgb(color))]
    return _ramp(color, np.linspace(0.0, 1.0, n))


def _ramp(color: str | None, fractions: np.ndarray) -> list[str]:
    """Get the colours at fractions of the colour range of the points.

    The range is the shades of a colour from light to the colour or the first
    `_COLORMAP_END` of the colour scale.
    """
    if color is not None:
        return [_shade(color, _LIGHTEST * (1.0 - float(f))) for f in fractions]
    if fractions.size == 0:
        return []
    return list(
        import_core().color_scale(
            COLORMAP, 0.0, 1.0, [float(f) * _COLORMAP_END for f in fractions]
        )
    )


def point_linestyles(n: int) -> list[str]:
    """Get the line styles of the points of a second `over` dimension.

    Raises:
        ValueError: for more than `len(LINE_STYLES)` points.
    """
    if n > len(LINE_STYLES):
        raise ValueError(
            f"A second dimension of a curve has at most {len(LINE_STYLES)} points, "
            f"one per line style, not {n}."
        )
    return list(LINE_STYLES[:n])


def point_values(
    dimension: Dimension | None,
    units: Mapping[str, str],
    labels: Sequence[Any] | None = None,
) -> tuple[str, np.ndarray, str] | None:
    """Get the target, the values and the unit of a dimension which changes one target.

    Args:
        dimension: the dimension of the scan, `None` if it is not known.
        units: the units of the symbols of the model, for values without a unit.
        labels: the labels of the points, in their order, every point of the
            dimension by default; a point is found by its label.

    Returns:
        The target, the values of the points and their unit, `None` for a
        dimension which changes no or several targets or has not every label.
    """
    if dimension is None or len(dimension.values) != 1:
        return None
    ((target, values),) = dimension.values.items()
    if isinstance(values, Quantity):
        magnitudes, unit = (
            np.asarray(values.magnitude, dtype=float),
            f"{values.units:~P}",
        )
    else:
        magnitudes, unit = np.asarray(values, dtype=float), units.get(target, "") or ""
    if labels is not None:
        positions = _positions(dimension, labels)
        if positions is None:
            return None
        magnitudes = magnitudes[positions]
    return target, magnitudes, unit


def point_labels(
    dim: str,
    labels: Sequence[Any],
    dimension: Dimension | None,
    units: Mapping[str, str],
) -> list[str]:
    """Get the legend label of every point of a dimension.

    Args:
        dim: the id of the dimension.
        labels: the labels of the points, in their order.
        dimension: the dimension of the scan, `None` if it is not known.
        units: the units of the symbols of the model, see `point_values`.

    Returns:
        `<target> = <value> <unit>` for a dimension which changes one target,
        else `<dim> = <label>`.
    """
    found = point_values(dimension, units, labels)
    if found is None:
        return [f"{dim} = {label_text(label)}" for label in labels]
    target, values, unit = found
    return [f"{target} = {format_value(value)} {unit}".rstrip() for value in values]


def point_scale(values: np.ndarray) -> tuple[float, float, bool] | None:
    """Get the range which maps the values of the points of a dimension onto colours.

    Returns:
        `(low, high, log)`, logarithmic when all values are positive and the
        largest is more than `LOG_RATIO` times the smallest, else linear from
        the smallest to the largest value; `None` if the values do not span a
        range (a single value, equal values or a value which is not finite).
    """
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    low, high = float(np.min(values)), float(np.max(values))
    if high <= low:
        return None
    return low, high, low > 0.0 and high / low > LOG_RATIO


def _fraction(value: float, low: float, high: float, log: bool) -> float:
    """Get the position of a value in the range, 0 at its low and 1 at its high end."""
    if log:
        return math.log(value / low) / math.log(high / low)
    return (value - low) / (high - low)


def format_value(value: float) -> str:
    """Format a value of a dimension: six significant digits, a short exponent."""
    text = f"{value:.6g}"
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e")
    return f"{mantissa}e{int(exponent)}"


def label_text(label: Any) -> str:
    """Get the text of a label of a dimension, a float formatted as a value."""
    if isinstance(label, float):
        return format_value(label)
    return str(label)


def _positions(dimension: Dimension, labels: Sequence[Any]) -> list[int] | None:
    """Get the positions of labels in a dimension, `None` if one is not in it."""
    index = {label: k for k, label in enumerate(dimension.labels.tolist())}
    try:
        return [index[label] for label in labels]
    except KeyError, TypeError:
        return None


def _broadcast(
    sid: str, what: str, arrays: Sequence[xr.DataArray | None]
) -> tuple[list[xr.DataArray | None], list[str]]:
    """Broadcast the arrays of a curve by name.

    Raises:
        ValueError: if two arrays have different coordinates of a dimension.
    """
    given = [a for a in arrays if a is not None]
    if _aligned(given):
        return list(arrays), [str(d) for d in given[0].dims]
    try:
        aligned = xr.align(*given, join="exact")
    except ValueError as err:
        raise ValueError(
            f"The data of the {what} '{sid}' has different coordinates of a "
            f"dimension: {err}"
        ) from err
    broadcast = iter(xr.broadcast(*aligned))
    full = [None if a is None else next(broadcast) for a in arrays]
    template = next(a for a in full if a is not None)
    return full, [str(d) for d in template.dims]


def _aligned(arrays: Sequence[xr.DataArray]) -> bool:
    """Get whether arrays are broadcast and aligned already.

    Arrays of the same dimensions in the same order and of the same shape,
    whose indexes of a dimension have the same labels, are what `xr.align`
    and `xr.broadcast` give back; these two are most of the cost of the
    lines of a figure, whose curves are mostly of such arrays (a timecourse,
    the column of a dataset and its errors).
    """
    first = arrays[0]
    if any(a.dims != first.dims or a.shape != first.shape for a in arrays[1:]):
        return False
    for d in first.dims:
        labels = [a.indexes[d] for a in arrays if d in a.indexes]
        if any(not labels[0].equals(other) for other in labels[1:]):
            return False
    return True


def _labels(
    arrays: Sequence[xr.DataArray | None], dims: Sequence[str]
) -> dict[str, list[Any]]:
    """Get the labels of dimensions, their positions for a dimension without labels."""
    template = next(a for a in arrays if a is not None)
    return {
        d: template[d].values.tolist()
        if d in template.coords
        else list(range(template.sizes[d]))
        for d in dims
    }


def _rgb(color: str) -> np.ndarray:
    """Parse a colour, see `sbmlsim.plot.colors`, as floats of 0 to 1."""
    hexed = normalize_color(color)
    if hexed is None:
        raise ValueError(f"the colour {color!r} is no colour")
    return np.array([int(hexed[i : i + 2], 16) for i in (1, 3, 5)]) / 255.0


def _hex(rgb: np.ndarray) -> str:
    """Write floats of 0 to 1 as a hex colour."""
    return "#" + "".join(f"{round(float(c) * 255):02x}" for c in rgb)


def _shade(color: str, white: float) -> str:
    """Mix a colour with a part of white, as a hex string."""
    rgb = _rgb(color)
    return _hex(rgb + (1.0 - rgb) * white)


@dataclass(frozen=True)
class ColorBarScale:
    """The scale of the colour bar of a dimension.

    Attributes:
        low: the value at the low end of the colour bar, the smallest value
            of the points, the position 0 of a categorical one.
        high: the value at its high end, the largest value of the points, the
            position `n - 1` of a categorical one.
        log: whether the scale is logarithmic.
        labels: the label of every point of a categorical colour bar, empty
            for a colour bar of values.
        categorical: whether it has a band per point, at the positions of the
            points `0..n-1`.
    """

    low: float
    high: float
    log: bool
    labels: list[str]
    categorical: bool


@dataclass(frozen=True)
class PointStyle:
    """How the lines of a curve over scan points look, see `point_styles`.

    Attributes:
        colors: the colour of every point of the first dimension.
        linestyles: the line style of every point of the second, `None` with one.
        labels: the legend label of every point of every dimension.
        colorbar: whether the first dimension gets a colour bar.
        values: the values of the points of the first dimension, their
            positions without a single target.
        title: the label of the colour bar, `<target> [<unit>]` or the dimension.
        scale: the scale of the colour bar, `None` for no points.
        middle: the colour of the middle of the colour range, of the legend
            entry the lines of a colour bar share.
    """

    colors: list[str]
    linestyles: list[str] | None
    labels: list[list[str]]
    colorbar: bool
    values: np.ndarray
    title: str
    scale: ColorBarScale | None
    middle: str


def colorbar_arguments(styles: PointStyle) -> dict[str, Any] | None:
    """Get the colour bar of points whose colours are of the colour scale `COLORMAP`.

    The points take the first `_COLORMAP_END` of the colour scale, since its
    light yellow end is hard to see on white, so the colour bar shows that part
    of the scale over the range of the values of the points: the colour of the
    bar at the value of a point is the colour of its line and the bar shows no
    value no line has. A dimension which does not change one target gets a
    categorical bar of a band per point, labelled by the labels of its points,
    up to `MAX_BANDS` points, a bar of their positions above.

    Args:
        styles: the style of the points, see `point_styles`, of a curve
            without a colour of its own.

    Returns:
        The keyword arguments of `ChartPlot.colorbar`, `None` for points which
        span no range.
    """
    scale = styles.scale
    if scale is None or scale.high <= scale.low:
        return None
    arguments: dict[str, Any] = {
        "scale": COLORMAP,
        "low": scale.low,
        "high": scale.high,
        "label": styles.title,
        "log": scale.log,
        "end": _COLORMAP_END,
    }
    if scale.categorical and len(scale.labels) <= MAX_BANDS:
        arguments["tick_labels"] = list(scale.labels)
        arguments["categorical"] = True
    return arguments


def point_styles(
    lines: Sequence[Line | BandLine],
    over: Sequence[str],
    color: str | None,
    dimensions: Sequence[Dimension | None],
    units: Mapping[str, str],
) -> PointStyle:
    """Get the colours, line styles, labels and colour bar of the lines of a curve.

    The points are the ones the lines draw, i.e. the labels left after the
    `sel` of the data, each found by its label in its dimension.

    Args:
        lines: the lines of the curve or band.
        over: the ids of the dimensions it is drawn over.
        color: the colour of the style of the curve, `None` without.
        dimensions: the dimension of the scan of every one of `over`, `None`
            where it is not known.
        units: the units of the symbols of the model, see `point_values`.

    Returns:
        The style shared by the serializers.

    Raises:
        ValueError: for more points of a second dimension than line styles.
    """
    labels_of = [
        list(dict.fromkeys(line.point[k] for line in lines)) for k in range(len(over))
    ]
    labels = [
        point_labels(d, labels_of[k], dimensions[k], units) for k, d in enumerate(over)
    ]
    n = len(labels_of[0])
    found = point_values(dimensions[0], units, labels_of[0])
    span = point_scale(found[1]) if found is not None else None
    scale: ColorBarScale | None
    if found is not None and span is not None:
        target, values, unit = found
        low, high, log = span
        title = f"{target} [{unit}]" if unit else target
        fractions = np.array([_fraction(float(v), low, high, log) for v in values])
        colors = _ramp(color, fractions)
        scale = ColorBarScale(low, high, log, [], False)
        (middle,) = _ramp(color, np.array([0.5]))
    else:
        values = np.arange(n, dtype=float)
        title = over[0]
        colors = point_colors(n, color)
        scale = (
            ColorBarScale(
                0.0,
                float(n - 1),
                False,
                [label_text(label) for label in labels_of[0]],
                True,
            )
            if n
            else None
        )
        middle = colors[n // 2] if n else _ramp(color, np.array([0.5]))[0]
    return PointStyle(
        colors=colors,
        linestyles=point_linestyles(len(labels_of[1])) if len(over) == 2 else None,
        labels=labels,
        colorbar=n >= COLORBAR_FROM,
        values=values,
        title=title,
        scale=scale,
        middle=middle,
    )
