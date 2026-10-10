"""The lines of a curve over the points of the dimensions of a scan.

A curve names the dimensions of its data it draws one line per point of
(`over`); after broadcasting by name its data must leave at most one more
dimension, the axis a line is drawn along: the time of a timecourse, the
points of a ragged result, the rows of a dataset or the dimension of the
values a scan sets (a value per simulation over a dimension). The lines are in
the order of the labels of the `over` dimensions, the last one fastest. Their
colours follow the first `over` dimension, their line styles the second; a
legend entry names its point by the value and unit of the target its
dimension changes, else by its label. A band reduces a dimension of draws to
two quantiles and the median, computed when the figure is drawn; it needs a
common grid of times.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from matplotlib import colormaps
from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    to_hex,
    to_rgb,
)

from sbmlsim.data import ROW
from sbmlsim.plot.padding import without_padding
from sbmlsim.result.scan import POINT, TIME
from sbmlsim.simulation import Dimension
from sbmlsim.units import Quantity

#: the line styles of the points of a second `over` dimension
LINE_STYLES: tuple[str, ...] = ("-", "--", ":", "-.")

#: the number of points from which a dimension gets a colour bar instead of
#: legend entries
COLORBAR_FROM = 11

#: the colour map of the points of a curve whose style sets no colour
COLORMAP = "viridis"

#: the part of the colour map the points use: its light yellow end is hard to
#: see on white
_COLORMAP_END = 0.85

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
    ordered = [None if a is None else a.transpose(*over, *axis) for a in full]
    labels = _labels(ordered, over)
    lines: list[Line] = []
    for index in itertools.product(*(range(len(labels[d])) for d in over)):
        position = dict(zip(over, index, strict=True))
        values = [
            None if a is None else np.asarray(a.isel(position).values) for a in ordered
        ]
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
        ValueError: if the data is ragged, has not the dimension `across`, or
            leaves more than one dimension besides `over` and `across`.
    """
    if POINT in y.dims or POINT in x.dims:
        raise ValueError(
            f"The band '{sid}' reduces '{across}' of a ragged result, whose "
            f"simulations keep their own time points; run the scan on a common grid "
            f"(a simulation with steps or times)."
        )
    if across not in y.dims:
        raise ValueError(
            f"The band '{sid}' reduces the dimension '{across}', which its data has "
            f"not: {[str(d) for d in y.dims]}."
        )
    # a point at which every draw is NaN (all simulations failed) has no quantile:
    # it is reduced on zeros, which the mask then replaces with NaN
    valid = y.notnull().any(dim=across)
    filled = y.where(valid, 0.0)
    low = (
        filled.quantile(quantiles[0], dim=across, skipna=True)
        .drop_vars("quantile")
        .where(valid)
    )
    high = (
        filled.quantile(quantiles[1], dim=across, skipna=True)
        .drop_vars("quantile")
        .where(valid)
    )
    median = filled.median(dim=across, skipna=True).where(valid)
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
        colour map `COLORMAP`, as hex strings; one point keeps the colour.
    """
    if color is not None:
        if n == 1:
            return [to_hex(color)]
        return [_shade(color, f) for f in np.linspace(_LIGHTEST, 0.0, n)]
    return [to_hex(colormaps[COLORMAP](f)) for f in np.linspace(0.0, _COLORMAP_END, n)]


def point_colormap(color: str | None) -> Colormap:
    """Get the colour map of the colour bar of a dimension, see `point_colors`."""
    if color is not None:
        return LinearSegmentedColormap.from_list(
            "shades", [_shade(color, _LIGHTEST), to_hex(color)]
        )
    return ListedColormap(colormaps[COLORMAP](np.linspace(0.0, _COLORMAP_END, 256)))


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
    dimension: Dimension | None, units: Mapping[str, str]
) -> tuple[str, np.ndarray, str] | None:
    """Get the target, the values and the unit of a dimension which changes one target.

    Args:
        dimension: the dimension of the scan, `None` if it is not known.
        units: the units of the symbols of the model, for values without a unit.

    Returns:
        The target, its values and their unit, `None` for a dimension which
        changes no or several targets.
    """
    if dimension is None or len(dimension.values) != 1:
        return None
    ((target, values),) = dimension.values.items()
    if isinstance(values, Quantity):
        return target, np.asarray(values.magnitude, dtype=float), f"{values.units:~P}"
    return target, np.asarray(values, dtype=float), units.get(target, "") or ""


def point_labels(
    dim: str,
    labels: Sequence[Any],
    dimension: Dimension | None,
    units: Mapping[str, str],
) -> list[str]:
    """Get the legend label of every point of a dimension.

    Args:
        dim: the id of the dimension.
        labels: the labels of its points.
        dimension: the dimension of the scan, `None` if it is not known.
        units: the units of the symbols of the model, see `point_values`.

    Returns:
        `<target> = <value> <unit>` for a dimension which changes one target,
        else `<dim> = <label>`.
    """
    found = point_values(dimension, units)
    if found is None:
        return [f"{dim} = {label}" for label in labels]
    target, values, unit = found
    return [f"{target} = {value:g} {unit}".rstrip() for value in values]


def _broadcast(
    sid: str, what: str, arrays: Sequence[xr.DataArray | None]
) -> tuple[list[xr.DataArray | None], list[str]]:
    """Broadcast the arrays of a curve by name.

    Raises:
        ValueError: if two arrays have different coordinates of a dimension.
    """
    given = [a for a in arrays if a is not None]
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


def _shade(color: str, white: float) -> str:
    """Mix a colour with a part of white, as a hex string."""
    rgb = np.asarray(to_rgb(color))
    return to_hex(rgb + (1.0 - rgb) * white)
