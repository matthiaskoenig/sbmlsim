"""The chart builder of the figure contract `figure/1` of sbmlsim-core.

A `Chart` is a figure of plots in a grid, which the renderer of sbmlsim-core
draws as SVG or PNG. Its plots are `ChartPlot`s with the axes `ChartAxis` and
elements of the kinds of the contract (curves, bands, bars, box plots, heat
maps, ...), added by the methods of the plot, each returning the id of its
element. The data of an element are sequences, numpy arrays, pint quantities
or xarray data arrays (`attrs["units"]` their unit): an axis with a unit
converts every quantity on it to that unit and takes plain values as values
in it, an axis without a unit takes the unit of the first quantity placed on
it, an axis without a label takes the `name` of the first data array placed
on it, and the renderer shows the axis as `<label> [<unit>]`:

    chart = Chart(title="HCTZ")
    plot = chart.plot(x=ChartAxis(label="time", unit="hr"), y=ChartAxis(label="cp"))
    plot.curve(res.q["time"], res.q["cp"], label="plasma")
    chart.save("hctz.svg")

A category axis (`scale="category"`) takes names, which are placed at the
index of their category: the categories of the axis, or, for an axis without
them, the names in the order of their first use.

`Chart.spec` is the spec as a dictionary of the snake case fields of
`figure/1`, `Chart.to_json` its JSON, in which a value which is not finite is
`null`, a gap. The renderer is imported through `sbmlsim.engine.import_core`
only when a chart is drawn.
"""

from __future__ import annotations

import copy
import json
import math
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tokenize import TokenError
from typing import Any, Final, Literal

import numpy as np
import xarray as xr
from pint.errors import DimensionalityError, PintError

from sbmlsim.atomic import atomic_path, write_text_atomic
from sbmlsim.plot.colors import color_alpha
from sbmlsim.plot.thinning import THIN_TOLERANCE, thin_elements
from sbmlsim.units import Quantity, short_unit, ureg

__all__ = ["Chart", "ChartAxis", "ChartPlot", "axis_label"]

#: The contract of the spec of a chart.
FIGURE_CONTRACT: Final = "figure/1"

#: The suffixes `Chart.save` writes.
SAVE_SUFFIXES: Final = (".svg", ".png", ".json")

#: The dpi of a PNG which is saved without one.
DEFAULT_DPI: Final = 144.0

#: The opacity the renderer fills a band, an area, a box, a violin and a
#: polygon with when their style has no fill; a fill color given without an
#: opacity keeps it.
BAND_OPACITY: Final = 0.18

#: The errors of pint parsing a unit it does not know.
_UNIT_ERRORS: Final = (PintError, ValueError, TypeError, AttributeError, TokenError)

#: The data of an element: a sequence, a numpy array, a pint quantity or an
#: xarray data array.
type Values = Any

type AxisName = Literal["x", "y", "y2"]

type LineType = Literal["solid", "dashed", "dotted", "dashdot", "none"]

#: The aspect of a panel: `equal` or the length of a unit of y in units of x.
type Aspect = Literal["equal"] | float

#: A colour map: the name of a map of matplotlib, or its colours at positions,
#: `[(0.0, color), (0.3, color), ..., (1.0, color)]`.
type ColorMap = str | Sequence[tuple[float, str]]

#: Which half of a marker its fill fills, matplotlib's `fillstyle`.
type FillStyle = Literal["full", "left", "right", "bottom", "top"]

#: Where a colour bar goes: right of its panel, above the panels of its row or
#: inside its panel.
type Placement = Literal["right", "top", "inset", "strip", "panel"]

type Coordinates = Literal["data", "axes"]

type Baseline = Literal["baseline", "center", "top", "bottom"]

type Extend = Literal["neither", "min", "max", "both"]

#: The markers of the renderer, the names of matplotlib's markers `o`, `s`,
#: `^`, `D`, `+`, `x`, `v`, `<`, `>`, `*`, `X`, `P`, `p`, `h` and `d` (and
#: `cross`, an `x` of its own).
MARKERS: Final = (
    "none",
    "circle",
    "square",
    "triangle",
    "diamond",
    "cross",
    "plus",
    "x",
    "triangle_down",
    "triangle_left",
    "triangle_right",
    "star",
    "x_filled",
    "plus_filled",
    "pentagon",
    "hexagon",
    "thin_diamond",
    "octagon",
    "tri_down",
    "tri_up",
    "tri_left",
    "tri_right",
)


def axis_label(name: str, unit: Any) -> str:
    """Get the label of an axis: its name and the short form of its unit.

    Args:
        name: the name of what the axis shows.
        unit: the unit, a text or a pint unit, `None` for none.

    Returns:
        `name [unit]` in the short form of the unit, e.g. `time [h]` for
        `hr`, the name for no unit and for a dimensionless one.
    """
    text = short_unit(None if unit is None else str(unit))
    return f"{name} [{text}]" if text else name


@dataclass(frozen=True)
class ChartAxis:
    """An axis of a plot of a chart, the `Axis` of `figure/1`.

    Attributes:
        label: the label, which the renderer shows with the unit; `None`
            takes the `name` of the first data array on the axis.
        unit: the unit, which the quantities on the axis are converted to;
            `None` takes the unit of the first quantity on the axis.
        scale: `linear`, `log` or `category`.
        categories: the names of the categories of a category axis, which
            are at the positions 0, 1, ...; `None` collects the names placed
            on the axis in the order of their first use, which depends on
            the order the elements are added in, so give the categories for
            an order which is reproducible.
        min: the lower end of the axis.
        max: the upper end of the axis.
        reverse: whether the axis is reversed.
        grid: whether the plot draws grid lines of the axis.
        label_visible: whether the label is shown.
        ticks_visible: whether the ticks are shown.
        tick_marks_visible: whether the tick marks are shown independent of
            the labels, `None` for marks which follow `ticks_visible`.
        line_visible: whether the line of the axis is shown; a plot without
            axes, e.g. a map of a lobule, hides the lines of both.
        tick_rotation: the rotation of the tick labels in degrees.
        max_decades: the decades of a log axis below its largest value,
            `None` for the default of the renderer.
        frame: the values `(low, high)` the axis must hold with its margins,
            in contrast to `min` and `max`, which are the ends themselves.
        ticks: the values of the ticks in the unit of the axis instead of the
            ones the renderer computes, matplotlib's `set_xticks`, e.g. every
            24 hours of a time axis; a tick outside of the limits is not
            drawn, the ticks do not change the limits; `None` for the ticks
            of the renderer. Not on a category axis.
        tick_labels: the labels of `ticks`, one per tick, a newline breaking
            a label into lines and `^{...}` a superscript; `None` for their
            numbers.
        tick_label_color: the color of the tick labels: one for all, or one
            per tick of `ticks` or per category (matplotlib's
            `ticklabel.set_color`); `None` for the color of the renderer.
    """

    label: str | None = None
    unit: str | None = None
    scale: Literal["linear", "log", "category"] = "linear"
    categories: Sequence[str] | None = None
    min: float | None = None
    max: float | None = None
    reverse: bool = False
    grid: bool = False
    label_visible: bool = True
    ticks_visible: bool = True
    tick_marks_visible: bool | None = None
    line_visible: bool = True
    tick_rotation: float = 0.0
    max_decades: float | None = None
    frame: tuple[float, float] | None = None
    ticks: Sequence[float] | None = None
    tick_labels: Sequence[str] | None = None
    tick_label_color: str | Sequence[str] | None = None

    def __post_init__(self) -> None:
        """Keep the categories, the frame and the ticks as tuples.

        Raises:
            ValueError: for ticks of a category axis, for tick labels without
                ticks and for as many labels as there are not ticks.
        """
        if self.categories is not None:
            object.__setattr__(self, "categories", tuple(self.categories))
        if self.frame is not None:
            low, high = self.frame
            object.__setattr__(self, "frame", (float(low), float(high)))
        if self.ticks is not None:
            if self.scale == "category":
                raise ValueError(
                    "A category axis has its categories as ticks and takes no ticks."
                )
            object.__setattr__(self, "ticks", tuple(float(t) for t in self.ticks))
        if self.tick_labels is not None:
            if self.ticks is None:
                raise ValueError("The tick labels of an axis need its ticks.")
            labels = tuple(str(t) for t in self.tick_labels)
            if len(labels) != len(self.ticks):
                raise ValueError(
                    f"An axis of {len(self.ticks)} ticks takes as many tick labels, "
                    f"not {len(labels)}."
                )
            object.__setattr__(self, "tick_labels", labels)
        if self.tick_label_color is not None and not isinstance(
            self.tick_label_color, str
        ):
            colors = tuple(_hex(c, "tick label") for c in self.tick_label_color)
            n = len(self.ticks if self.ticks is not None else self.categories or ())
            if len(colors) != n:
                raise ValueError(
                    f"An axis of {n} ticks or categories takes as many tick label "
                    f"colors, not {len(colors)}."
                )
            object.__setattr__(self, "tick_label_color", colors)


def _hex(color: str, what: str) -> str:
    """Get a color as `#rrggbb`.

    Raises:
        ValueError: for a color which is none.
    """
    hexed, _ = color_alpha(str(color))
    if hexed is None:
        raise ValueError(f"The {what} color {color!r} is no color.")
    return hexed


@dataclass
class _AxisState:
    """An axis of a plot with the unit, label and categories it collected."""

    name: AxisName
    plot: str
    axis: ChartAxis
    unit: Any = None
    label: str | None = None
    categories: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Parse the unit of the axis; one pint does not parse is shown as given."""
        self.label = self.axis.label
        if self.axis.categories is not None:
            self.categories = list(self.axis.categories)
        if self.axis.unit:
            try:
                self.unit = ureg.Unit(self.axis.unit)
            except _UNIT_ERRORS:
                self.unit = None

    @property
    def where(self) -> str:
        """Get the axis in the words of a message."""
        return f"the axis {self.name} of the plot '{self.plot}'"

    def index(self, names: list[str]) -> np.ndarray:
        """Get the positions of names on the category axis.

        Raises:
            ValueError: for an axis which is no category axis and for a name
                which is no category of an axis with given categories.
        """
        if self.axis.scale != "category":
            raise ValueError(
                f"{self.where} is no category axis and takes no names: {names[:3]}."
            )
        positions: list[float] = []
        for name in names:
            if name not in self.categories:
                if self.axis.categories is not None:
                    raise ValueError(
                        f"'{name}' is no category of {self.where}, "
                        f"which has {list(self.axis.categories)}."
                    )
                self.categories.append(name)
            positions.append(float(self.categories.index(name)))
        return np.asarray(positions, dtype=float)

    def spec(self) -> dict[str, Any]:
        """Get the axis as a dictionary of `figure/1`."""
        a = self.axis
        unit = short_unit(str(self.unit)) if self.unit is not None else a.unit
        spec: dict[str, Any] = {}
        if self.label is not None:
            spec["label"] = self.label
        if unit:
            spec["unit"] = unit
        spec["scale"] = a.scale
        if a.scale == "category" or a.categories is not None:
            spec["categories"] = list(self.categories)
        if a.min is not None:
            spec["min"] = float(a.min)
        if a.max is not None:
            spec["max"] = float(a.max)
        if a.frame is not None:
            spec["frame"] = {"low": a.frame[0], "high": a.frame[1]}
        if a.max_decades is not None:
            spec["max_decades"] = float(a.max_decades)
        spec.update(
            reverse=bool(a.reverse),
            grid=bool(a.grid),
            label_visible=bool(a.label_visible),
            ticks_visible=bool(a.ticks_visible),
            tick_rotation=float(a.tick_rotation),
        )
        if a.tick_marks_visible is not None:
            spec["tick_marks_visible"] = bool(a.tick_marks_visible)
        if not a.line_visible:
            spec["line_visible"] = False
        if a.ticks is not None:
            spec["ticks"] = list(a.ticks)
        if a.tick_labels is not None:
            spec["tick_labels"] = list(a.tick_labels)
        if isinstance(a.tick_label_color, str):
            spec["tick_label_color"] = _hex(a.tick_label_color, "tick label")
        elif a.tick_label_color is not None:
            spec["tick_label_color"] = list(a.tick_label_color)
        return spec


#: the locations of a legend inside its panel, the `loc` of matplotlib: `best`
#: takes the one of the least data, the others are fixed
LEGEND_LOCS: Final = (
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
)


def legend_loc_spec(loc: str) -> str:
    """Get a location of a legend for the spec, its words joined by `_`.

    Raises:
        ValueError: for a location which is none of `LEGEND_LOCS`.
    """
    name = str(loc).strip().lower().replace("_", " ")
    if name not in LEGEND_LOCS:
        raise ValueError(
            f"The legend location {loc!r} is none of {', '.join(LEGEND_LOCS)}."
        )
    return name.replace(" ", "_")


def _aspect(aspect: Aspect | None) -> str | float | None:
    """Get the aspect of a plot for the spec.

    Raises:
        ValueError: for an aspect which is neither `equal` nor a positive
            number.
    """
    if aspect is None or aspect == "equal":
        return aspect
    if (
        isinstance(aspect, str)
        or not math.isfinite(value := float(aspect))
        or value <= 0
    ):
        raise ValueError(
            f"The aspect of a plot is 'equal' or a positive number, not {aspect!r}."
        )
    return value


def _error_cap(error_cap: float) -> float:
    """Get the length of the caps of error bars in pixels.

    Raises:
        ValueError: for a length which is negative or not finite.
    """
    value = float(error_cap)
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"The caps of error bars are a length in pixels, not {error_cap!r}."
        )
    return value


def _legend_columns(columns: int | None) -> int | None:
    """Get the columns of a legend, `None` for the default.

    Raises:
        ValueError: for fewer than one column.
    """
    if columns is None:
        return None
    if int(columns) < 1:
        raise ValueError(f"A legend has at least 1 column, not {columns!r}.")
    return int(columns)


def _error_width(error_width: float | None) -> float | None:
    """Get the width of the lines of error bars in pixels, `None` for the default.

    Raises:
        ValueError: for a width which is negative or not finite.
    """
    if error_width is None:
        return None
    value = float(error_width)
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"The lines of error bars have a width in pixels, not {error_width!r}."
        )
    return value


def _numbers(values: Values) -> list[float | None]:
    """Get the numbers of an array as a list, a value which is not finite `None`."""
    flat = np.asarray(values, dtype=float).ravel()
    numbers: list[float | None] = list(flat.tolist())
    for i in np.flatnonzero(~np.isfinite(flat)).tolist():
        numbers[i] = None
    return numbers


def _column(values: np.ndarray) -> np.ndarray:
    """Get a column of the line of an element, a copy of its numbers.

    The columns of curves, bands and areas are kept as arrays until the spec
    is built, which thins them first (`ChartPlot._thinned_elements`) and
    then writes them as numbers (`_listed`).
    """
    return np.array(values, dtype=float).ravel()


def _listed(element: dict[str, Any]) -> dict[str, Any]:
    """Get an element of the spec with its columns as numbers, see `_column`."""
    for key, value in element.items():
        if isinstance(value, np.ndarray):
            element[key] = _numbers(value)
    return element


def _finite(value: Values, name: str) -> float:
    """Get a number which must be finite.

    Raises:
        ValueError: for a value which is not finite, naming the field.
    """
    number = float(np.asarray(value, dtype=float).ravel()[0])
    if not math.isfinite(number):
        raise ValueError(f"The {name} must be a finite number, not {number}.")
    return number


def _names(data: Values) -> list[str] | None:
    """Get data as names of categories, `None` for data which are no names."""
    if isinstance(data, str):
        return [data]
    if isinstance(data, Quantity):
        return None
    values = data.values if isinstance(data, xr.DataArray) else data
    try:
        flat = np.asarray(values, dtype=object).ravel().tolist()
    except ValueError:
        return None
    if flat and all(isinstance(v, str) for v in flat):
        return flat
    return None


def _magnitude(data: Values) -> np.ndarray:
    """Get the numbers of data in its own unit, for values on no axis."""
    if isinstance(data, Quantity):
        return np.asarray(data.magnitude, dtype=float)
    if isinstance(data, xr.DataArray):
        return np.asarray(data.values, dtype=float)
    return np.asarray(data, dtype=float)


def _marker_color(color: str) -> str:
    """Get the color of a marker as `#rrggbb`, `none` for a hollow marker.

    A fill `none` (or a color without opacity) draws the marker as its edge
    alone. The markers have no opacity, the alpha of a color is dropped.
    """
    normalized, alpha = color_alpha(color)
    return normalized if normalized is not None and alpha > 0.0 else "none"


@dataclass(frozen=True)
class _Style:
    """The style keywords of an element; `None` leaves a part to the renderer."""

    color: str | None = None
    opacity: float = 1.0
    line: LineType | None = None
    width: float | None = None
    marker: str | None = None
    size: float | None = None
    marker_fill: str | None = None
    marker_edge: str | None = None
    marker_edge_width: float | None = None
    marker_fill_style: FillStyle = "full"
    marker_fill_alt: str | None = None
    fill: str | None = None
    fill_opacity: float | None = None
    default_fill_opacity: float = 1.0

    def spec(self) -> dict[str, Any]:
        """Get the style as a dictionary of `figure/1`."""
        style: dict[str, Any] = {}
        opacity = float(self.opacity)
        if self.color is not None:
            color, alpha = color_alpha(self.color)
            if color is not None:
                style["color"] = color
            opacity *= alpha
        style["opacity"] = opacity
        if self.line is not None or self.width is not None:
            style["line"] = {
                "type": self.line if self.line is not None else "solid",
                "width": float(self.width) if self.width is not None else 2.0,
            }
        if self.marker is not None:
            marker: dict[str, Any] = {"type": self.marker}
            if self.size is not None:
                marker["size"] = float(self.size)
            if self.marker_fill is not None:
                marker["fill"] = _marker_color(self.marker_fill)
            if self.marker_edge is not None:
                marker["edge"] = _marker_color(self.marker_edge)
            if self.marker_edge_width is not None:
                marker["edge_width"] = float(self.marker_edge_width)
            if self.marker_fill_style != "full":
                marker["fill_style"] = self.marker_fill_style
            if self.marker_fill_alt is not None:
                marker["fill_alt"] = _marker_color(self.marker_fill_alt)
            style["marker"] = marker
        if self.fill is not None or self.fill_opacity is not None:
            fill: dict[str, Any] = {}
            alpha = 1.0
            if self.fill is not None:
                color, alpha = color_alpha(self.fill)
                if color is not None:
                    fill["color"] = color
            fill["opacity"] = alpha * float(
                self.fill_opacity
                if self.fill_opacity is not None
                else self.default_fill_opacity
            )
            style["fill"] = fill
        return style


#: the style keywords of a legend swatch, see `Chart.plot`
_SWATCH_KEYWORDS: Final = frozenset(
    {"color", "opacity", "line", "width", "marker", "size", "marker_fill"}
    | {"marker_edge", "marker_edge_width", "fill", "fill_opacity"}
    | {"marker_fill_style", "marker_fill_alt"}
)


def _swatch_keywords(group: str, keywords: Mapping[str, Any]) -> dict[str, Any]:
    """Get the style keywords of the swatch of a legend group.

    Raises:
        ValueError: for a keyword which is no style keyword of a swatch.
    """
    unknown = sorted(set(keywords) - _SWATCH_KEYWORDS)
    if unknown:
        raise ValueError(
            f"The swatch of the legend group '{group}' has the keywords {unknown}, "
            f"which are no style keywords: {sorted(_SWATCH_KEYWORDS)}."
        )
    return dict(keywords)


class ChartPlot:
    """A plot of a chart, which its methods add elements to.

    A plot is created by `Chart.plot`; every method adds one element and
    returns its id. The style keywords of a method are the ones the renderer
    reads for its kind: `line` and `width` the line (of a curve, a rule, the
    median of a band and the outline of a box, a violin and a polygon),
    `fill` and `fill_opacity` the fill (of a band, an area, a bar, a box, a
    violin, a histogram and a polygon; a band, an area, a box, a violin and a
    polygon are filled with their color at `BAND_OPACITY` without them) and
    `marker`, `size`, `marker_fill`, `marker_edge`, `marker_edge_width`,
    `marker_fill_style` and `marker_fill_alt` the markers of a curve and of
    the outliers of a box, so `fill` is the fill of a shape and never the one
    of a marker. The markers are `MARKERS`, matplotlib's shapes: a half
    filled marker (`marker_fill_style` `left`, `right`, `bottom` or `top`,
    matplotlib's `fillstyle`) fills that half with `marker_fill` and the
    other with `marker_fill_alt` (none by default, matplotlib's
    `markerfacecoloralt`).
    """

    def __init__(
        self,
        chart: Chart,
        id: str,
        row: int,
        column: int,
        *,
        row_span: int,
        column_span: int,
        title: str | None,
        x: ChartAxis,
        y: ChartAxis,
        y2: ChartAxis | None,
        legend: bool,
        legend_order: Sequence[str] | None,
        declared_legends: Sequence[Sequence[str]] | None,
        legend_swatches: Mapping[str, Mapping[str, Any]] | None = None,
        aspect: Aspect | None = None,
        legend_loc: str | None = None,
        legend_title: str | None = None,
        legend_columns: int | None = None,
        x2: ChartAxis | None = None,
        frame: dict[str, Any] | None = None,
        inset: dict[str, Any] | None = None,
        legend_anchor: tuple[float, float] | None = None,
        legends: Sequence[Mapping[str, Any]] | None = None,
        legend_tuples: Sequence[str] | None = None,
    ) -> None:
        """Create a plot, see `Chart.plot`."""
        self.chart = chart
        self.id = id
        self.row = int(row)
        self.column = int(column)
        self.row_span = int(row_span)
        self.column_span = int(column_span)
        self.title = title
        self.legend = bool(legend)
        self.legend_loc = None if legend_loc is None else legend_loc_spec(legend_loc)
        self.legend_title = legend_title
        self.legend_columns = _legend_columns(legend_columns)
        self.aspect = _aspect(aspect)
        self.legend_order = None if legend_order is None else list(legend_order)
        self.declared_legends = (
            None
            if declared_legends is None
            else [list(entries) for entries in declared_legends]
        )
        self._axes: dict[str, _AxisState] = {
            "x": _AxisState("x", id, x),
            "y": _AxisState("y", id, y),
        }
        if y2 is not None:
            self._axes["y2"] = _AxisState("y2", id, y2)
        self.x2 = x2
        if x2 is not None and x2.scale == "category":
            raise ValueError("A top axis is linear or log, its labels its tick labels.")
        self.frame = frame
        self.inset = inset
        self.legend_anchor = (
            None if legend_anchor is None else [float(v) for v in legend_anchor]
        )
        self.legends = None if legends is None else [_plot_legend(lg) for lg in legends]
        self.legend_tuples = None if legend_tuples is None else list(legend_tuples)
        self._elements: list[dict[str, Any]] = []
        # the thinned elements by the tolerance and the number of elements
        self._thinned: tuple[float, int, list[dict[str, Any]]] | None = None
        self._colorbars: list[dict[str, Any]] = []
        self.legend_swatches = (
            None
            if legend_swatches is None
            else {
                group: _Style(**_swatch_keywords(group, keywords)).spec()
                for group, keywords in legend_swatches.items()
            }
        )

    def _state(self, axis: AxisName) -> _AxisState:
        """Get an axis of the plot."""
        if axis not in self._axes:
            raise ValueError(f"The plot '{self.id}' has no axis '{axis}'.")
        return self._axes[axis]

    def _on(self, axis: AxisName, data: Values) -> np.ndarray:
        """Get data as numbers in the unit of an axis, which may take its unit.

        Names on a category axis are the indices of their categories; the
        first data array with a name on an axis without a label names it.

        Raises:
            pint.DimensionalityError: for a quantity which the unit of the
                axis cannot take.
            ValueError: for names on an axis which takes none and for units
                which cannot be converted to the unit of the axis.
        """
        state = self._state(axis)
        names = _names(data)
        if names is not None:
            return state.index(names)
        if isinstance(data, xr.DataArray):
            if state.label is None and isinstance(data.name, str):
                state.label = data.name
            units = data.attrs.get("units")
            values = np.asarray(data.values, dtype=float)
            if not units:
                return values
            try:
                data = ureg.Quantity(values, units)
            except _UNIT_ERRORS as error:
                raise ValueError(
                    f"The units '{units}' of a data array on {state.where} are "
                    f"no units of pint: {error}"
                ) from error
        if not isinstance(data, Quantity):
            return np.asarray(data, dtype=float)
        if state.unit is None:
            if state.axis.unit:
                raise ValueError(
                    f"{state.where} has the unit '{state.axis.unit}', which pint "
                    "does not know, so a quantity cannot be converted to it."
                )
            try:
                state.unit = ureg.Unit(str(data.units))
            except _UNIT_ERRORS as error:
                raise ValueError(
                    f"The unit '{data.units}' of a quantity on {state.where} is no "
                    f"unit of the registry of sbmlsim: {error}"
                ) from error
            return np.asarray(data.magnitude, dtype=float)
        try:
            return np.asarray(data.to(state.unit).magnitude, dtype=float)
        except DimensionalityError as error:
            raise DimensionalityError(
                error.units1,
                error.units2,
                error.dim1,
                error.dim2,
                extra_msg=f" on {state.where}",
            ) from error
        except _UNIT_ERRORS as error:
            raise ValueError(
                f"A quantity of '{data.units}' cannot be converted to the unit "
                f"'{state.unit}' of {state.where}: {error}"
            ) from error

    def _position(self, axis: AxisName, value: Values, name: str) -> float:
        """Get one finite value in the unit of an axis."""
        return _finite(self._on(axis, value), name)

    def _placed(
        self, axis: AxisName, values: Values, coordinates: Coordinates
    ) -> np.ndarray:
        """Get values in the unit of an axis, or fractions of the panel as they are."""
        if coordinates == "axes":
            return _magnitude(values).ravel()
        return self._on(axis, values)

    def _errors(self, axis: AxisName, err: Values) -> Any:
        """Get errors for the spec: one per value, or the sides apart.

        Errors of two sides are a mapping of `low` and `high` or a tuple
        `(low, high)` (matplotlib's errors of the shape `(2, n)`), each side
        not negative, a value which is not finite a side which is not drawn;
        a list or an array is one error per value.
        """
        if isinstance(err, Mapping):
            unknown = sorted(set(err) - {"low", "high"})
            if unknown or "low" not in err or "high" not in err:
                raise ValueError(
                    "Errors of two sides are a mapping of 'low' and 'high', "
                    f"not of {sorted(err)}."
                )
            low, high = err["low"], err["high"]
        elif isinstance(err, tuple) and len(err) == 2:
            low, high = err
        else:
            return _column(self._on(axis, err))
        return {
            "low": _numbers(self._on(axis, low)),
            "high": _numbers(self._on(axis, high)),
        }

    def _add(
        self,
        kind: str,
        data: dict[str, Any],
        style: _Style,
        *,
        label: str | None,
        legend: str | None,
        id: str | None,
        axis: AxisName,
        coordinates: Coordinates = "data",
        clip: bool = True,
    ) -> str:
        """Add an element of a kind with its data and return its id."""
        self._state(axis)
        if coordinates not in ("data", "axes"):
            raise ValueError(
                f"The coordinates of an element are 'data' or 'axes', not {coordinates!r}."
            )
        element_id = self.chart._element_id(id)
        element: dict[str, Any] = {"id": element_id, "kind": kind}
        if label is not None:
            element["label"] = label
        if legend is not None:
            element["legend"] = legend
        element["axis"] = axis
        element["style"] = style.spec()
        if coordinates != "data":
            element["coordinates"] = coordinates
        if not clip:
            element["clip"] = False
        element.update(data)
        self._elements.append(element)
        return element_id

    def curve(
        self,
        x: Values,
        y: Values,
        *,
        xerr: Values | None = None,
        yerr: Values | None = None,
        error_cap: float = 0.0,
        error_width: float | None = None,
        line: LineType = "solid",
        width: float = 2.0,
        marker: str = "none",
        size: float = 6.0,
        marker_fill: str | None = None,
        marker_edge: str | None = None,
        marker_edge_width: float = 1.0,
        marker_fill_style: FillStyle = "full",
        marker_fill_alt: str | None = None,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
        coordinates: Coordinates = "data",
        clip: bool = True,
    ) -> str:
        """Add a curve, a line and/or markers with error bars.

        Args:
            x: the x values.
            y: the y values.
            xerr: the errors of x: one per value, or the errors below and
                above every value apart, a mapping of `low` and `high` or a
                tuple `(low, high)` (matplotlib's errors of the shape
                `(2, n)`), a value which is not finite a side which is not
                drawn (a one-sided error bar).
            yerr: the errors of y, as `xerr`.
            error_cap: the length of the caps across the ends of the error
                bars in pixels, matplotlib's `capsize`; 0 for none.
            error_width: the width of the lines of the error bars and their
                caps in pixels, matplotlib's `elinewidth`; `None` for the 1
                pixel of the renderer.
            line: the type of the line, `solid`, `dashed`, `dotted`,
                `dashdot` or `none`.
            width: the width of the line in pixels.
            marker: the type of the markers, one of `MARKERS`: `none`,
                `circle`, `square`, `triangle`, `diamond`, `cross`, `plus`,
                `x`, `triangle_down`, `triangle_left`, `triangle_right`,
                `star`, `x_filled`, `plus_filled`, `pentagon`, `hexagon`,
                `thin_diamond`, `octagon`, `tri_down`, `tri_up`, `tri_left` or
                `tri_right`.
            size: the size of the markers in pixels.
            marker_fill: the fill color of the markers, the color by default.
            marker_edge: the edge color of the markers, the color by default.
            marker_edge_width: the width of the edge of the markers in pixels.
            marker_fill_style: the half of the markers `marker_fill` fills,
                `full`, `left`, `right`, `bottom` or `top` (matplotlib's
                `fillstyle`).
            marker_fill_alt: the fill color of the other half of a half
                filled marker, none by default (matplotlib's
                `markerfacecoloralt`).
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color, a name of matplotlib or hex (see
                `sbmlsim.plot.colors`), the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.
            coordinates: `data`, values of the axes, or `axes`, fractions of
                the panel (matplotlib's `transform=ax.transAxes`), which put
                nothing on the axes.
            clip: whether the curve is clipped to the panel; `False` draws it
                over the whole figure (matplotlib's `clip_on=False`), e.g. the
                slashes of a broken axis.

        Returns:
            The id of the element.
        """
        data: dict[str, Any] = {
            "x": _column(self._placed("x", x, coordinates)),
            "y": _column(self._placed(axis, y, coordinates)),
        }
        if xerr is not None:
            data["xerr"] = self._errors("x", xerr)
        if yerr is not None:
            data["yerr"] = self._errors(axis, yerr)
        if cap := _error_cap(error_cap):
            data["error_cap"] = cap
        if (lines := _error_width(error_width)) is not None:
            data["error_width"] = lines
        style = _Style(
            color=color,
            opacity=opacity,
            line=line,
            width=width,
            marker=marker,
            size=size,
            marker_fill=marker_fill,
            marker_edge=marker_edge,
            marker_edge_width=marker_edge_width,
            marker_fill_style=marker_fill_style,
            marker_fill_alt=marker_fill_alt,
        )
        return self._add(
            "curve",
            data,
            style,
            label=label,
            legend=legend,
            id=id,
            axis=axis,
            coordinates=coordinates,
            clip=clip,
        )

    def band(
        self,
        x: Values,
        low: Values,
        high: Values,
        *,
        median: Values | None = None,
        line: LineType = "solid",
        width: float = 2.0,
        fill: str | None = None,
        fill_opacity: float | None = None,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add a band between two curves with an optional median line.

        Args:
            x: the x values.
            low: the lower edge.
            high: the upper edge.
            median: the median line.
            line: the type of the median line.
            width: the width of the median line in pixels.
            fill: the fill color, the color by default.
            fill_opacity: the opacity of the fill, `BAND_OPACITY` by default.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color, a name of matplotlib or hex (see
                `sbmlsim.plot.colors`), the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.
        """
        data: dict[str, Any] = {
            "x": _column(self._on("x", x)),
            "low": _column(self._on(axis, low)),
            "high": _column(self._on(axis, high)),
        }
        if median is not None:
            data["median"] = _column(self._on(axis, median))
        style = _Style(
            color=color,
            opacity=opacity,
            line=line,
            width=width,
            fill=fill,
            fill_opacity=fill_opacity,
            default_fill_opacity=BAND_OPACITY,
        )
        return self._add(
            "band", data, style, label=label, legend=legend, id=id, axis=axis
        )

    def area(
        self,
        x: Values,
        low: Values,
        high: Values,
        *,
        fill: str | None = None,
        fill_opacity: float | None = None,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add a filled area between two curves.

        Args:
            x: the x values.
            low: the start of the area, `from` of the contract.
            high: the end of the area, `to` of the contract.
            fill: the fill color, the color by default.
            fill_opacity: the opacity of the fill, `BAND_OPACITY` by default.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color, a name of matplotlib or hex (see
                `sbmlsim.plot.colors`), the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.
        """
        data = {
            "x": _column(self._on("x", x)),
            "from": _column(self._on(axis, low)),
            "to": _column(self._on(axis, high)),
        }
        style = _Style(
            color=color,
            opacity=opacity,
            fill=fill,
            fill_opacity=fill_opacity,
            default_fill_opacity=BAND_OPACITY,
        )
        return self._add(
            "area", data, style, label=label, legend=legend, id=id, axis=axis
        )

    def bar(
        self,
        x: Values,
        height: Values,
        *,
        err: Values | None = None,
        error_cap: float = 0.0,
        error_width: float | None = None,
        horizontal: bool = False,
        bar_width: float | None = None,
        base: Values | None = None,
        line: LineType | None = None,
        width: float | None = None,
        fill: str | None = None,
        fill_opacity: float | None = None,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add bars at positions or at categories.

        Args:
            x: the positions or the names of their categories, on the y axis
                for horizontal bars.
            height: the heights, on the x axis for horizontal bars.
            err: the errors of the heights, one per bar or the sides apart,
                see `curve`.
            error_cap: the length of the caps across the ends of the error
                bars in pixels; 0 for none.
            error_width: the width of the lines of the error bars in pixels,
                `None` for the 1 pixel of the renderer.
            horizontal: whether the bars are horizontal.
            bar_width: the width of a bar in units of the axis of the
                positions, the default of the renderer for `None`.
            base: where every bar starts, matplotlib's `bottom` (`left` of
                horizontal bars), so bars stack; 0 for `None`.
            line: the type of the outline, in the color of the bars; a bar
                has no outline without `line` and `width`.
            width: the width of the outline in pixels.
            fill: the fill color, the color by default.
            fill_opacity: the opacity of the fill, 1 by default.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color, a name of matplotlib or hex (see
                `sbmlsim.plot.colors`), the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.
        """
        positions, heights = (axis, "x") if horizontal else ("x", axis)
        data: dict[str, Any] = {
            "x": _numbers(self._on(positions, x)),
            "height": _numbers(self._on(heights, height)),
        }
        if err is not None:
            errors = self._errors(heights, err)
            data["err"] = errors if isinstance(errors, dict) else _numbers(errors)
        if cap := _error_cap(error_cap):
            data["error_cap"] = cap
        if (lines := _error_width(error_width)) is not None:
            data["error_width"] = lines
        data["horizontal"] = bool(horizontal)
        if bar_width is not None:
            data["width"] = float(bar_width)
        if base is not None:
            data["base"] = _numbers(self._on(heights, base))
        style = _Style(
            color=color,
            opacity=opacity,
            line=line,
            width=width,
            fill=fill,
            fill_opacity=fill_opacity,
        )
        return self._add(
            "bar", data, style, label=label, legend=legend, id=id, axis=axis
        )

    def box(
        self,
        positions: Values,
        stats: Sequence[Mapping[str, Any]],
        *,
        box_width: float | None = None,
        median_color: str | None = None,
        whisker_opacity: float | None = None,
        line: LineType = "solid",
        width: float = 2.0,
        fill: str | None = None,
        fill_opacity: float | None = None,
        marker: str = "none",
        size: float = 6.0,
        marker_fill: str | None = None,
        marker_edge: str | None = None,
        marker_edge_width: float = 1.0,
        marker_fill_style: FillStyle = "full",
        marker_fill_alt: str | None = None,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add box plots, one per position.

        Args:
            positions: the positions on the x axis or the names of their
                categories.
            stats: the statistics of every box: `low_whisker`, `q1`,
                `median`, `q3`, `high_whisker` and optional `outliers`,
                values on the y axis; an outlier which is not finite is left
                out.
            box_width: the width of a box in units of the x axis,
                matplotlib's `widths`, a share of the slot of a category on a
                category axis (two boxes side by side at ±0.2 of a category
                with 0.35); `None` for half of the spacing of the positions.
            median_color: the color of the median line (matplotlib's
                `medianprops`), the color of the box for `None`.
            whisker_opacity: the opacity of the whiskers and their caps, the
                opacity of the box for `None`.
            line: the type of the outline of the boxes.
            width: the width of the outline in pixels.
            fill: the fill color, the color by default.
            fill_opacity: the opacity of the fill, `BAND_OPACITY` by default.
            marker: the marker of the outliers, `none` for the open circles
                of the renderer.
            size: the size of the markers of the outliers in pixels.
            marker_fill: the fill color of the markers of the outliers,
                none by default.
            marker_edge: the edge color of the markers of the outliers, the
                color by default.
            marker_edge_width: the width of the edge of the markers in pixels.
            marker_fill_style: the half of the markers of the outliers
                `marker_fill` fills, see `curve`.
            marker_fill_alt: the fill color of the other half, see `curve`.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color, a name of matplotlib or hex (see
                `sbmlsim.plot.colors`), the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.

        Raises:
            ValueError: for a statistic which is not finite and for a width
                of a box which is not positive.
        """
        boxes: list[dict[str, Any]] = []
        for k, stat in enumerate(stats):
            box: dict[str, Any] = {
                key: _finite(self._on(axis, stat[key]), f"stats[{k}].{key}")
                for key in ("low_whisker", "q1", "median", "q3", "high_whisker")
            }
            outliers = _numbers(self._on(axis, stat.get("outliers", [])))
            box["outliers"] = [v for v in outliers if v is not None]
            boxes.append(box)
        data: dict[str, Any] = {
            "positions": _numbers(self._on("x", positions)),
            "stats": boxes,
        }
        if box_width is not None:
            value = float(box_width)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"The width of a box is a positive number, not {box_width!r}."
                )
            data["width"] = value
        if median_color is not None:
            data["median_color"] = _hex(median_color, "median")
        if whisker_opacity is not None:
            data["whisker_opacity"] = float(whisker_opacity)
        style = _Style(
            color=color,
            opacity=opacity,
            line=line,
            width=width,
            fill=fill,
            fill_opacity=fill_opacity,
            default_fill_opacity=BAND_OPACITY,
            marker=marker,
            size=size,
            marker_fill=marker_fill,
            marker_edge=marker_edge,
            marker_edge_width=marker_edge_width,
            marker_fill_style=marker_fill_style,
            marker_fill_alt=marker_fill_alt,
        )
        return self._add(
            "box", data, style, label=label, legend=legend, id=id, axis=axis
        )

    def violin(
        self,
        positions: Values,
        densities: Sequence[Mapping[str, Any]],
        *,
        line: LineType = "solid",
        width: float = 2.0,
        fill: str | None = None,
        fill_opacity: float | None = None,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add violins, one per position.

        Args:
            positions: the positions on the x axis or the names of their
                categories.
            densities: the density of every violin, `values` on the y axis
                and their `density`, of one length.
            line: the type of the outline of the violins.
            width: the width of the outline in pixels.
            fill: the fill color, the color by default.
            fill_opacity: the opacity of the fill, `BAND_OPACITY` by default.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color, a name of matplotlib or hex (see
                `sbmlsim.plot.colors`), the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.
        """
        data = {
            "positions": _numbers(self._on("x", positions)),
            "densities": [
                {
                    "values": _numbers(self._on(axis, d["values"])),
                    "density": _numbers(_magnitude(d["density"])),
                }
                for d in densities
            ],
        }
        style = _Style(
            color=color,
            opacity=opacity,
            line=line,
            width=width,
            fill=fill,
            fill_opacity=fill_opacity,
            default_fill_opacity=BAND_OPACITY,
        )
        return self._add(
            "violin", data, style, label=label, legend=legend, id=id, axis=axis
        )

    def histogram(
        self,
        edges: Values,
        counts: Values,
        *,
        base: Values | None = None,
        line: LineType | None = None,
        width: float | None = None,
        fill: str | None = None,
        fill_opacity: float | None = None,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add a histogram.

        Args:
            edges: the edges of the bins on the x axis, one more than counts.
            counts: the counts of the bins.
            base: where the bar of every bin starts, matplotlib's `bottom`,
                so histograms stack (`hist(stacked=True)`); 0 for `None`.
            line: the type of the edges of the bars (matplotlib's
                `edgecolor`), in the color of the histogram; the bars of
                neighbouring bins are one outline without edges without `line`
                and `width`.
            width: the width of the edges in pixels.
            fill: the fill color, the color by default.
            fill_opacity: the opacity of the fill, 1 by default.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color, a name of matplotlib or hex (see
                `sbmlsim.plot.colors`), the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.
        """
        data: dict[str, Any] = {
            "edges": _numbers(self._on("x", edges)),
            "counts": _numbers(self._on(axis, counts)),
        }
        if base is not None:
            data["base"] = _numbers(self._on(axis, base))
        style = _Style(
            color=color,
            opacity=opacity,
            line=line,
            width=width,
            fill=fill,
            fill_opacity=fill_opacity,
        )
        return self._add(
            "histogram", data, style, label=label, legend=legend, id=id, axis=axis
        )

    def heatmap(
        self,
        rows: Sequence[str] | None,
        columns: Sequence[str] | None,
        values: Values,
        *,
        x_edges: Values | None = None,
        y_edges: Values | None = None,
        text: Sequence[Sequence[str]] | None = None,
        scale: ColorMap = "viridis",
        low: float | None = None,
        high: float | None = None,
        center: float | None = None,
        log: bool = False,
        gamma: float | None = None,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add a heat map.

        Without edges its cells lie on the categories of the axes, its
        columns along x and its rows along y. With `x_edges` its columns span
        the edges on a numeric x axis instead, which stays numeric, so the
        heat map shares the axis with another plot, e.g. the time of a
        space-time heat map above a curve of that time; `y_edges` likewise
        for its rows.

        Args:
            rows: the names of the rows, `None` with `y_edges`.
            columns: the names of the columns, `None` with `x_edges`.
            values: the values, a row per row of the length of the columns.
            x_edges: the edges of the columns on the x axis, one more than
                columns, strictly increasing or decreasing; values or
                quantities of the axis.
            y_edges: the edges of the rows on the y axis, one more than rows.
            text: the text of the cells, in the shape of `values`.
            scale: the color map: the name of a sequential or diverging
                colour map of matplotlib in any case (`viridis`, `Blues`,
                `PuRd`, `RdBu`, ...) or its reversal `<name>_r`, see
                `sbmlsim_core.COLOR_SCALES`, or the colours of a map at
                positions, `[(0.0, color), (0.3, color), ..., (1.0, color)]`
                as matplotlib's `LinearSegmentedColormap.from_list`, the
                first at 0 and the last at 1, the positions not decreasing.
            low: the value of the lower end of the scale, the smallest
                finite value (positive on a log scale) by default.
            high: the value of the upper end of the scale, the largest
                finite value by default.
            center: the value at the middle of a diverging scale.
            log: whether the scale is logarithmic.
            gamma: the exponent of a power scale, matplotlib's `PowerNorm`.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color, a name of matplotlib or hex (see
                `sbmlsim.plot.colors`), the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.

        Raises:
            ValueError: for a heat map without the names or the edges of its
                rows or of its columns.
        """
        for names, edges, which in [
            (rows, y_edges, "rows"),
            (columns, x_edges, "columns"),
        ]:
            if names is None and edges is None:
                raise ValueError(
                    f"A heat map needs the names or the edges of its {which}."
                )
        cells = [_numbers(_magnitude(row)) for row in _rows(values)]
        data: dict[str, Any] = {}
        if rows is not None:
            data["rows"] = [str(r) for r in rows]
        if columns is not None:
            data["columns"] = [str(c) for c in columns]
        data["values"] = cells
        if x_edges is not None:
            data["x_edges"] = _numbers(self._on("x", x_edges))
        if y_edges is not None:
            data["y_edges"] = _numbers(self._on(axis, y_edges))
        if text is not None:
            data["text"] = [[str(t) for t in row] for row in text]
        data["scale"] = _color_scale(scale, low, high, center, log, cells, gamma)
        return self._add(
            "heatmap",
            data,
            _Style(color=color, opacity=opacity),
            label=label,
            legend=legend,
            id=id,
            axis=axis,
        )

    def text(
        self,
        x: Values | None,
        y: Values | None,
        text: str,
        *,
        anchor: Literal["start", "middle", "end"] = "start",
        size: float | None = None,
        weight: Literal["normal", "bold"] = "normal",
        family: Literal["sans", "monospace"] = "sans",
        rotation: float = 0.0,
        baseline: Baseline | None = None,
        box: Mapping[str, Any] | None = None,
        loc: str | None = None,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
        coordinates: Coordinates = "data",
        clip: bool = True,
    ) -> str:
        """Add a text at a position or at a location of the panel.

        Args:
            x: the x position, or the name of its category; `None` with
                `loc`.
            y: the y position; `None` with `loc`.
            text: the text, whose runs of spaces are kept, `_{...}` a
                subscript and `^{...}` a superscript.
            anchor: the anchor of the text at the position, `start`,
                `middle` or `end`.
            size: the font size, the `tick` size of the fonts of the chart
                for `None`.
            weight: the font weight, `normal` or `bold`.
            family: the font, `sans` or `monospace`, whose characters are all
                0.6 of the size wide, so the columns of a table of lines
                padded with spaces align (`sbmlsim_core.text_width(text,
                size, family="monospace")` measures it).
            rotation: the rotation in degrees counterclockwise, the anchors
                aligning the box around the rotated text (matplotlib's
                `rotation`).
            baseline: the vertical anchor, `baseline`, `center`, `top` or
                `bottom` (matplotlib's `va`); `None` centres the text.
            box: a box behind the text, matplotlib's `bbox`: a mapping of
                `fill` (white; `none`), `opacity`, `edge` (none), `edge_width`,
                `padding` (pixels, 0.3 font sizes) and `style` (`square` or
                `round`).
            loc: a location of `LEGEND_LOCS` but `best` instead of a position,
                matplotlib's `AnchoredText`: the text in a white box with a
                black edge (or its `box`) in that corner of the panel.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color, a name of matplotlib or hex (see
                `sbmlsim.plot.colors`), the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.
            coordinates: `data` or `axes`, the position in fractions of the
                panel, see `curve`.
            clip: whether the text is clipped to the panel, see `curve`.

        Returns:
            The id of the element.

        Raises:
            ValueError: for a position which is not finite, for a position
                with a location and none without, and for an unknown family.
        """
        data: dict[str, Any] = {}
        if loc is not None:
            spot = legend_loc_spec(loc)
            if spot == "best" or x is not None or y is not None:
                raise ValueError(
                    f"A text at the location {loc!r} has no position and a fixed "
                    "location, not 'best'."
                )
        elif x is None or y is None:
            raise ValueError("A text needs a position x and y, or a location.")
        elif coordinates == "axes":
            data["x"] = _finite(_magnitude(x), "x of a text")
            data["y"] = _finite(_magnitude(y), "y of a text")
        else:
            data["x"] = self._position("x", x, "x of a text")
            data["y"] = self._position(axis, y, "y of a text")
        data["text"] = str(text)
        data["anchor"] = anchor
        if size is not None:
            data["size"] = float(size)
        if weight != "normal":
            data["weight"] = weight
        if family not in ("sans", "monospace"):
            raise ValueError(f"A text is 'sans' or 'monospace', not {family!r}.")
        if family != "sans":
            data["family"] = family
        if rotation:
            data["rotation"] = float(rotation)
        if baseline is not None:
            data["baseline"] = baseline
        if box is not None:
            data["box"] = _text_box(box)
        if loc is not None:
            data["loc"] = legend_loc_spec(loc)
        return self._add(
            "text",
            data,
            _Style(color=color, opacity=opacity),
            label=label,
            legend=legend,
            id=id,
            axis=axis,
            coordinates=coordinates,
            clip=clip,
        )

    def arrow(
        self,
        x: Values,
        y: Values,
        x_from: Values,
        y_from: Values,
        *,
        head: Literal["open", "filled", "none"] = "filled",
        head_length: float | None = None,
        head_width: float | None = None,
        body: float | None = None,
        shrink: float = 0.0,
        text: str | None = None,
        size: float | None = None,
        weight: Literal["normal", "bold"] = "normal",
        anchor: Literal["start", "middle", "end"] = "start",
        baseline: Baseline | None = None,
        line: LineType = "solid",
        width: float = 2.0,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
        coordinates: Coordinates = "data",
        clip: bool = True,
    ) -> str:
        """Add an arrow from a tail to a head with an optional text at its tail.

        matplotlib's `annotate(text, xy=(x, y), xytext=(x_from, y_from),
        arrowprops=...)`: an arrow of a line with an `open` (`->`), a
        `filled` (`-|>`) or no head (`-`), or with a `body` a filled arrow of
        that width (`arrowprops` of `width`, `headwidth` and `headlength`).

        Args:
            x: the x of the head.
            y: the y of the head.
            x_from: the x of the tail.
            y_from: the y of the tail.
            head: the head of an arrow of a line.
            head_length: the length of the head in pixels, `None` for the
                default of the renderer.
            head_width: the width of the head across the arrow in pixels.
            body: the width of the body of a filled arrow in pixels; `None`
                for an arrow of a line.
            shrink: the share of the length left out at each end,
                matplotlib's `shrink`.
            text: a text at the tail, the arrow starting at its edge.
            size: the font size of the text.
            weight: the font weight of the text.
            anchor: the horizontal anchor of the text.
            baseline: the vertical anchor of the text, its baseline for
                `None`.
            line: the type of the line of an arrow of a line.
            width: the width of the line in pixels.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color, the text color by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.
            coordinates: `data` or `axes`, see `curve`.
            clip: whether the arrow is clipped to the panel, see `curve`.

        Returns:
            The id of the element.
        """

        def at(name: AxisName, value: Values, what: str) -> float:
            if coordinates == "axes":
                return _finite(_magnitude(value), what)
            return self._position(name, value, what)

        data: dict[str, Any] = {
            "x": at("x", x, "x of an arrow"),
            "y": at(axis, y, "y of an arrow"),
            "x_from": at("x", x_from, "x_from of an arrow"),
            "y_from": at(axis, y_from, "y_from of an arrow"),
        }
        if head != "filled":
            data["head"] = head
        for key, value in (
            ("head_length", head_length),
            ("head_width", head_width),
            ("body", body),
            ("size", size),
        ):
            if value is not None:
                data[key] = float(value)
        if shrink:
            data["shrink"] = float(shrink)
        if text is not None:
            data["text"] = str(text)
        if weight != "normal":
            data["weight"] = weight
        if anchor != "start":
            data["anchor"] = anchor
        if baseline is not None:
            data["baseline"] = baseline
        style = _Style(color=color, opacity=opacity, line=line, width=width)
        return self._add(
            "arrow",
            data,
            style,
            label=label,
            legend=legend,
            id=id,
            axis=axis,
            coordinates=coordinates,
            clip=clip,
        )

    def contour(
        self,
        x: Values,
        y: Values,
        values: Values,
        levels: Sequence[float],
        *,
        filled: bool = False,
        scale: ColorMap | None = None,
        low: float | None = None,
        high: float | None = None,
        center: float | None = None,
        log: bool = False,
        gamma: float | None = None,
        labels: bool | Sequence[str] = False,
        label_size: float | None = None,
        line: LineType = "solid",
        width: float = 1.5,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add contours of a grid of values: lines at levels or filled bands.

        matplotlib's `contour` with `clabel(inline=True)`, or with `filled`
        its `contourf`; the axes stick to the ends of the grid.

        Args:
            x: the coordinates of the columns of the grid, strictly monotonic.
            y: the coordinates of the rows of the grid, strictly monotonic.
            values: the values, per row (the shape `(len(y), len(x))`), a
                value which is not finite no value.
            levels: the levels, strictly increasing (matplotlib's `levels`,
                which `MaxNLocator` gives for a number).
            filled: whether the bands between the levels are filled.
            scale: the color map of the levels, a name or stops, see
                `heatmap`; `None` draws the lines in the color of the element.
            low: the value of the lower end of the scale, the first level for
                `None` (matplotlib's norm of a `ContourSet`).
            high: the value of the upper end, the last level for `None`.
            center: the value at the middle of a diverging scale.
            log: whether the scale is logarithmic.
            gamma: the exponent of a power scale, matplotlib's `PowerNorm`.
            labels: the labels of the lines inline, one per level (an empty
                one for none), or `True` for the levels written `%g`.
            label_size: the font size of the labels, the `tick` size for
                `None`.
            line: the type of the lines.
            width: the width of the lines in pixels (matplotlib's 1.5).
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color of lines without a scale, the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.

        Raises:
            ValueError: for labels of filled bands.
        """
        levels_list = [float(v) for v in levels]
        data: dict[str, Any] = {
            "x": _numbers(self._on("x", x)),
            "y": _numbers(self._on(axis, y)),
            "values": [_numbers(_magnitude(row)) for row in _rows(values)],
            "levels": levels_list,
        }
        if filled:
            data["filled"] = True
        if scale is not None:
            data["scale"] = _color_scale(
                scale,
                min(levels_list) if low is None else low,
                max(levels_list) if high is None else high,
                center,
                log,
                [],
                gamma,
            )
        if labels is True:
            data["labels"] = [f"{v:g}" for v in levels_list]
        elif labels is not False:
            data["labels"] = [str(t) for t in labels]
        if filled and "labels" in data:
            raise ValueError("Filled bands of contours have no labels.")
        if label_size is not None:
            data["label_size"] = float(label_size)
        style = _Style(color=color, opacity=opacity, line=line, width=width)
        return self._add(
            "contour", data, style, label=label, legend=legend, id=id, axis=axis
        )

    def rule(
        self,
        value: Values,
        *,
        orientation: Literal["horizontal", "vertical"] = "horizontal",
        line: LineType = "solid",
        width: float = 2.0,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add a line across the plot.

        Args:
            value: the position on the other axis, a value of the y axis
                for a horizontal rule and of the x axis for a vertical one.
            orientation: `horizontal` or `vertical`.
            line: the type of the line.
            width: the width of the line in pixels.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color, a name of matplotlib or hex (see
                `sbmlsim.plot.colors`), the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.

        Raises:
            ValueError: for a value which is not finite.
        """
        on: AxisName = axis if orientation == "horizontal" else "x"
        data = {
            "orientation": orientation,
            "value": self._position(on, value, "value of a rule"),
        }
        style = _Style(color=color, opacity=opacity, line=line, width=width)
        return self._add(
            "rule", data, style, label=label, legend=legend, id=id, axis=axis
        )

    def polygon(
        self,
        paths: Sequence[Values],
        *,
        line: LineType = "solid",
        width: float = 2.0,
        fill: str | None = None,
        fill_opacity: float | None = None,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
        coordinates: Coordinates = "data",
        clip: bool = True,
    ) -> str:
        """Add filled polygons.

        Args:
            paths: the rings, each of at least 3 points: a sequence of
                `(x, y)` points, an array of the shape `(n, 2)` or a quantity
                of that shape, whose one unit both coordinates have; the x
                coordinates are values of the x axis and the y coordinates of
                the y axis.
            line: the type of the outline, `none` for none.
            width: the width of the outline in pixels.
            fill: the fill color, the color by default.
            fill_opacity: the opacity of the fill, `BAND_OPACITY` by default.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color, a name of matplotlib or hex (see
                `sbmlsim.plot.colors`), the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.
            coordinates: `data` or `axes`, the points in fractions of the
                panel, see `curve`.
            clip: whether the polygons are clipped to the panel, see `curve`.

        Returns:
            The id of the element.
        """
        rings: list[list[list[float | None]]] = []
        for k, path in enumerate(paths):
            xs, ys = _columns(path, f"paths[{k}]")
            rings.append(
                [
                    [px, py]
                    for px, py in zip(
                        _numbers(self._placed("x", xs, coordinates)),
                        _numbers(self._placed(axis, ys, coordinates)),
                        strict=True,
                    )
                ]
            )
        style = _Style(
            color=color,
            opacity=opacity,
            line=line,
            width=width,
            fill=fill,
            fill_opacity=fill_opacity,
            default_fill_opacity=BAND_OPACITY,
        )
        return self._add(
            "polygon",
            {"paths": rings},
            style,
            label=label,
            legend=legend,
            id=id,
            axis=axis,
            coordinates=coordinates,
            clip=clip,
        )

    def colorbar(
        self,
        scale: ColorMap,
        low: float,
        high: float,
        *,
        label: str | None = None,
        center: float | None = None,
        log: bool = False,
        start: float = 0.0,
        end: float = 1.0,
        ticks: Sequence[float] | None = None,
        tick_labels: Sequence[str] | None = None,
        categorical: bool = False,
        marks: Sequence[float | Mapping[str, Any]] | None = None,
        placement: Placement = "right",
        loc: str | None = None,
        length: float | None = None,
        thickness: float | None = None,
        extend: Extend = "neither",
        rows: int | None = None,
        gamma: float | None = None,
    ) -> None:
        """Add a color bar to the plot, a field of the plot and no element.

        A plot of several color bars shows them stacked right of its panel in
        the order they were added. A bar may go above the panels of the row of
        its plot instead (`placement="top"`, matplotlib's
        `fig.colorbar(..., location="top")` over a row, one bar per row) or
        into its panel (`placement="inset"`, a horizontal bar in matplotlib's
        `ax.inset_axes`, which a legend at `best` goes around):

            plot.colorbar(
                [(0.0, "#006d77"), (0.3, "#2a9d4b"), (1.0, "#d8f3dc")],
                0.1, 2.0, log=True, label="renal function [-]",
                ticks=[0.1, 1, 2], tick_labels=["0.1", "1", "2"],
                marks=[1.0], placement="top", length=0.4,
            )

        Args:
            scale: the color map, a name or stops, see `heatmap`.
            low: the value of the lower end of the scale.
            high: the value of the upper end of the scale.
            label: the label of the color bar.
            center: the value at the middle of a diverging scale.
            log: whether the scale is logarithmic.
            start: the part of the color map `low` takes, from 0 to 1.
            end: the part of the color map `high` takes, above `start`, so the
                values `low..high` show the part `start..end` of the map.
            ticks: the values of the ticks, chosen by the renderer without.
            tick_labels: the label of every tick, the numbers of the ticks
                without; a categorical bar takes one per band without `ticks`.
            categorical: whether the bar is one band of one color per integer
                from `low` to `high`, e.g. of the lines over named values.
            marks: lines across the bar at values of the scale (matplotlib's
                `cbar.ax.axhline`, e.g. at the default of a scan): a value,
                black and 2 pixels wide, or a mapping of `value`, `color` and
                `width`.
            placement: `right` of the panel (vertical), `top`, above the
                panels of the row centred over them, `inset`, inside the
                panel at `loc` (both horizontal), `strip`, the colors along
                the top of the panel without ticks and outline, or `panel`,
                vertical in place of the panel, which draws nothing else (a
                figure of a color bar alone).
            loc: the location of an inset bar, a location of `LEGEND_LOCS`
                but `best`; `None` for the upper right.
            length: the length of the bar as a share of what it runs along,
                matplotlib's `shrink`: the height of the panel (1), the span
                of the panels of the row (1, 0.4 for a bar of 40 % of it) or
                the width of the panel of an inset (0.5); `None` for these.
            thickness: the thickness of the bar in pixels, `None` for the 12
                pixels of the renderer (0.03 of the panel of a strip, the
                width of the panel of a bar in it).
            extend: the ends of the bar with a triangle, `neither`, `min`,
                `max` or `both` (matplotlib's `extend`).
            rows: the rows of the grid a bar right of its panel spans from the
                row of its plot (matplotlib's `fig.colorbar(...,
                ax=axes[:, -1])`); `None` for its row.
            gamma: the exponent of a power scale, matplotlib's `PowerNorm`.

        Raises:
            ValueError: for a mark which is no number or mapping, for an
                unknown placement, for a location of a bar which is no inset
                or `best`, and for a length which is not above 0 and at most
                1.
        """
        color_scale = _color_scale(scale, low, high, center, log, [], gamma)
        if start != 0.0:
            color_scale["start"] = float(start)
        if end != 1.0:
            color_scale["end"] = float(end)
        colorbar: dict[str, Any] = {"scale": color_scale}
        if label is not None:
            colorbar["label"] = label
        if ticks is not None:
            colorbar["ticks"] = [float(t) for t in ticks]
        if tick_labels is not None:
            colorbar["tick_labels"] = [str(t) for t in tick_labels]
        if categorical:
            colorbar["categorical"] = True
        if marks is not None:
            colorbar["marks"] = [_mark(m) for m in marks]
        if placement not in ("right", "top", "inset", "strip", "panel"):
            raise ValueError(
                "A color bar goes 'right', 'top', 'inset', 'strip' or 'panel', "
                f"not {placement!r}."
            )
        if placement != "right":
            colorbar["placement"] = placement
        if loc is not None:
            spot = legend_loc_spec(loc)
            if placement != "inset" or spot == "best":
                raise ValueError(
                    f"The location {loc!r} of a color bar is one of an inset bar "
                    "and no 'best'."
                )
            colorbar["loc"] = spot
        if length is not None:
            share = float(length)
            if not 0.0 < share <= 1.0:
                raise ValueError(
                    f"The length of a color bar is a share above 0 and at most 1, "
                    f"not {length!r}."
                )
            colorbar["length"] = share
        if thickness is not None:
            colorbar["thickness"] = float(thickness)
        if extend not in ("neither", "min", "max", "both"):
            raise ValueError(
                f"A color bar extends 'neither', 'min', 'max' or 'both', not {extend!r}."
            )
        if extend != "neither":
            colorbar["extend"] = extend
        if rows is not None:
            colorbar["rows"] = int(rows)
        self._colorbars.append(colorbar)

    def spec(self) -> dict[str, Any]:
        """Get the plot as a dictionary of `figure/1`."""
        spec: dict[str, Any] = {
            "id": self.id,
            "row": self.row,
            "column": self.column,
            "row_span": self.row_span,
            "column_span": self.column_span,
        }
        if self.title is not None:
            spec["title"] = self.title
        spec["x"] = self._axes["x"].spec()
        spec["y"] = self._axes["y"].spec()
        if "y2" in self._axes:
            spec["y2"] = self._axes["y2"].spec()
        if self.x2 is not None:
            spec["x2"] = _AxisState("x", self.id, self.x2).spec()
        if self.frame is not None:
            spec["frame"] = self.frame
        if self.inset is not None:
            spec["inset"] = self.inset
        spec["legend"] = self.legend
        if self.legend_loc is not None:
            spec["legend_loc"] = self.legend_loc
        if self.legend_title is not None:
            spec["legend_title"] = self.legend_title
        if self.legend_columns is not None:
            spec["legend_columns"] = self.legend_columns
        if self.legend_anchor is not None:
            spec["legend_anchor"] = self.legend_anchor
        if self.legends is not None:
            spec["legends"] = self.legends
        if self.legend_tuples is not None:
            spec["legend_tuples"] = self.legend_tuples
        if len(self._colorbars) == 1:
            spec["colorbar"] = self._colorbars[0]
        elif self._colorbars:
            spec["colorbars"] = self._colorbars
        if self.legend_swatches is not None:
            spec["legend_swatches"] = self.legend_swatches
        if self.declared_legends is not None:
            spec["declared_legends"] = self.declared_legends
        if self.legend_order is not None:
            spec["legend_order"] = self.legend_order
        if self.aspect is not None:
            spec["aspect"] = self.aspect
        spec["elements"] = self._thinned_elements(spec)
        spec = copy.deepcopy(spec)
        spec["elements"] = [_listed(e) for e in spec["elements"]]
        return spec

    def _thinned_elements(self, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Get the elements with their lines thinned, see `Chart`.

        The thinned elements are kept until an element is added, so the
        formats of a chart which is saved several times are thinned once.
        """
        tolerance = self.chart.thinning
        if tolerance is None:
            return self._elements
        cached = self._thinned
        if cached is not None and cached[:2] == (tolerance, len(self._elements)):
            return cached[2]
        axes = {name: spec[name] for name in ("x", "y", "y2") if name in spec}
        elements = thin_elements(self._elements, axes, tolerance)
        self._thinned = (tolerance, len(self._elements), elements)
        return elements


def _rows(values: Values) -> list[Any]:
    """Get the rows of the values of a heat map, which may differ in length."""
    if isinstance(values, Quantity | xr.DataArray | np.ndarray):
        return list(np.atleast_2d(_magnitude(values)))
    return list(values)


def _columns(path: Values, name: str) -> tuple[Values, Values]:
    """Get the x and y coordinates of a ring of a polygon, its points of shape `(n, 2)`.

    Raises:
        ValueError: for points which are no pairs.
    """
    values = path.magnitude if isinstance(path, Quantity) else path
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"The {name} of a polygon are points of the shape (n, 2), "
            f"not of the shape {points.shape}."
        )
    if isinstance(path, Quantity):
        return points[:, 0] * path.units, points[:, 1] * path.units
    return points[:, 0], points[:, 1]


def _mark(mark: float | Mapping[str, Any]) -> dict[str, Any]:
    """Get a mark of a color bar for the spec.

    Raises:
        ValueError: for a mark which is neither a number nor a mapping of
            `value`, `color` and `width`.
    """
    if isinstance(mark, Mapping):
        unknown = sorted(set(mark) - {"value", "color", "width"})
        if "value" not in mark or unknown:
            raise ValueError(
                "A mark of a color bar is a value or a mapping of 'value', 'color' "
                f"and 'width', not {dict(mark)!r}."
            )
        spec: dict[str, Any] = {"value": float(mark["value"])}
        if mark.get("color") is not None:
            color, _ = color_alpha(str(mark["color"]))
            if color is not None:
                spec["color"] = color
        if mark.get("width") is not None:
            spec["width"] = float(mark["width"])
        return spec
    if isinstance(mark, str):
        raise ValueError(f"A mark of a color bar is a value, not {mark!r}.")
    return {"value": float(mark)}


def _color_map(name: ColorMap) -> dict[str, Any]:
    """Get the colour map of a scale: its name in lower case or its stops.

    The name is that of the spec, a colour map of matplotlib in lower case:
    matplotlib's `Blues` is `blues`, its `PuRd_r` is `purd_r`. Stops are
    pairs of a position and a colour, a name of matplotlib or hex, written as
    `[position, "#rrggbb"]`.

    Raises:
        ValueError: for stops which are no pairs or of a colour which is none.
    """
    if isinstance(name, str):
        return {"name": name.lower()}
    stops: list[list[Any]] = []
    for stop in name:
        try:
            position, value = stop
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"A stop of a colour map is a pair (position, color), not {stop!r}."
            ) from error
        color, _ = color_alpha(str(value))
        if color is None:
            raise ValueError(f"The colour {value!r} of a stop is no colour.")
        stops.append([float(position), color])
    return {"stops": stops}


def _text_box(box: Mapping[str, Any]) -> dict[str, Any]:
    """Get the box of a text for the spec.

    Raises:
        ValueError: for an unknown key and a color which is none.
    """
    keys = {"fill", "opacity", "edge", "edge_width", "padding", "style"}
    unknown = sorted(set(box) - keys)
    if unknown:
        raise ValueError(f"A box of a text has the keys {sorted(keys)}, not {unknown}.")
    spec: dict[str, Any] = {}
    for key in ("fill", "edge"):
        value = box.get(key)
        if value is not None:
            spec[key] = "none" if str(value) == "none" else _hex(value, f"box {key}")
    for key in ("opacity", "edge_width", "padding"):
        if box.get(key) is not None:
            spec[key] = float(box[key])
    if box.get("style") is not None:
        spec["style"] = str(box["style"])
    return spec


def _plot_legend(legend: Mapping[str, Any]) -> dict[str, Any]:
    """Get a further legend of a plot or a legend of the figure for the spec.

    Raises:
        ValueError: for an unknown key or a legend without groups.
    """
    keys = {"groups", "title", "loc", "anchor", "columns", "family", "position"}
    unknown = sorted(set(legend) - keys)
    if unknown or "groups" not in legend:
        raise ValueError(
            f"A legend has 'groups' and the keys {sorted(keys - {'groups'})}, "
            f"not {sorted(legend)}."
        )
    spec: dict[str, Any] = {"groups": [str(g) for g in legend["groups"]]}
    if legend.get("position") is not None:
        spec["position"] = str(legend["position"])
    if legend.get("title") is not None:
        spec["title"] = str(legend["title"])
    if legend.get("loc") is not None:
        spec["loc"] = legend_loc_spec(legend["loc"])
    if legend.get("anchor") is not None:
        spec["anchor"] = [float(v) for v in legend["anchor"]]
    if legend.get("columns") is not None:
        spec["columns"] = _legend_columns(legend["columns"])
    if legend.get("family") is not None:
        spec["family"] = str(legend["family"])
    return spec


def _color_scale(
    name: ColorMap,
    low: float | None,
    high: float | None,
    center: float | None,
    log: bool,
    cells: Sequence[Sequence[float | None]],
    gamma: float | None = None,
) -> dict[str, Any]:
    """Get a color scale, its ends by default the range of the finite values.

    The map is a name or stops, see `_color_map`.
    """
    finite = [v for row in cells for v in row if v is not None and (v > 0 or not log)]
    if low is None:
        low = min(finite) if finite else (1.0 if log else 0.0)
    if high is None:
        high = max(finite) if finite else low * 10.0 if log else low + 1.0
    scale: dict[str, Any] = {
        **_color_map(name),
        "low": float(low),
        "high": float(high),
    }
    if center is not None:
        scale["center"] = float(center)
    scale["log"] = bool(log)
    if gamma is not None:
        scale["gamma"] = float(gamma)
    return scale


#: the fields of the fonts of a chart which are weights, the others sizes
_FONT_WEIGHTS: Final = frozenset(
    {"title_weight", "label_weight", "tick_weight", "legend_weight"}
    | {"plot_title_weight", "legend_family"}
)


def _fonts(fonts: Mapping[str, float | str]) -> dict[str, float | str]:
    """Get the fonts of a chart, the sizes as numbers and the weights as texts."""
    return {k: str(v) if k in _FONT_WEIGHTS else float(v) for k, v in fonts.items()}


def _with_resolution(png: bytes, dpi: float) -> bytes:
    """Get a PNG with its resolution, one `pHYs` chunk after its header.

    The renderer writes no resolution; a reader such as LaTeX sizes an image
    by it, and takes 72 dpi without it. A `pHYs` chunk the PNG has is
    replaced, a PNG has at most one.

    Args:
        png: the PNG, its first chunk the header `IHDR`.
        dpi: the resolution in dots per inch.

    Returns:
        The PNG with a `pHYs` chunk of the resolution in pixels per meter.
    """
    chunks: list[bytes] = []
    position = 8
    while position < len(png):
        (length,) = struct.unpack(">I", png[position : position + 4])
        end = position + 12 + length
        if png[position + 4 : position + 8] != b"pHYs":
            chunks.append(png[position:end])
        position = end
    per_meter = round(float(dpi) / 0.0254)
    data = b"pHYs" + struct.pack(">IIB", per_meter, per_meter, 1)
    phys = struct.pack(">I", 9) + data + struct.pack(">I", zlib.crc32(data))
    return png[:8] + chunks[0] + phys + b"".join(chunks[1:])


def _thinning(tolerance: float) -> float:
    """Get the tolerance of the thinning of a chart.

    Raises:
        ValueError: for a tolerance which is negative or not finite.
    """
    value = float(tolerance)
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"The thinning of a chart is a share of the span of its axes, not "
            f"{tolerance!r}."
        )
    return value


class Chart:
    """A figure of plots in a grid, the spec `figure/1` of sbmlsim-core.

    Plots are added with `plot`, elements with the methods of the plots;
    `svg`, `png` and `save` draw it with the renderer of sbmlsim-core.

    The spec carries the lines of the curves, bands and areas thinned to the
    points which are visibly apart (`thinning`, `sbmlsim.plot.thinning`): a
    curve of a simulation has every step of the integrator, thousands of
    points no one can tell apart in a line of a few hundred pixels. The
    points of a curve with markers or error bars are data and kept.
    """

    def __init__(
        self,
        width: float = 504,
        height: float = 360,
        *,
        title: str | None = None,
        rows: int = 1,
        columns: int = 1,
        wspace: float = 0.3,
        hspace: float = 0.3,
        fonts: Mapping[str, float | str] | None = None,
        legend_scope: str = "plot",
        legend_position: str = "inside",
        legend_loc: str = "best",
        legend_columns: int | None = None,
        legend_title: str | None = None,
        legends: Sequence[Mapping[str, Any]] | None = None,
        row_titles: Sequence[str] | None = None,
        dpi: float | None = None,
        thinning: float | None = THIN_TOLERANCE,
    ) -> None:
        """Create a chart.

        Args:
            width: the width in pixels, which legends outside the plots grow.
            height: the height in pixels.
            title: the title above all plots.
            rows: the rows of the grid.
            columns: the columns of the grid.
            wspace: the space between the columns as a fraction of a plot.
            hspace: the space between the rows as a fraction of a plot.
            fonts: the font sizes in points by `title`, `label`, `tick`,
                `tick_y` (the tick labels of the y axes and the color bars,
                `tick` without), `legend` and `plot_title` (the titles of the
                plots, `label` without), `colorbar_label` (the labels of the
                color bars, `label` without), the weights `normal` or
                `bold` by `title_weight`, `plot_title_weight` (the plot
                titles, `title_weight` without), `label_weight`, `tick_weight`
                and `legend_weight`, and `legend_family`, `sans` or `monospace`
                (every legend); the defaults of the renderer for the others.
            legend_scope: `plot`, a legend per plot, or `row`, one per row.
            legend_position: `inside`, `outside`, right of the plot, or
                `bottom`, one legend of the entries of all plots below the
                grid, centred (matplotlib's `fig.legend(loc="lower center")`).
            legend_loc: where a legend inside goes, a location of matplotlib
                (`LEGEND_LOCS`): `best`, the location of the least data (a
                legend stays inside however much data it covers and goes
                outside only when it does not fit into its panel), or a fixed
                one, e.g. `upper left`; a plot may set its own.
            legend_columns: the columns of the legend `bottom`, matplotlib's
                `ncols`, filled top down and then left to right; `None` for
                as many as fit into the width in one row.
            legend_title: the title of the legend `bottom`.
            legends: legends of the figure of the groups they list, each a
                mapping of `groups`, `position` (`right`, right of the grid at
                its top, matplotlib's `"outside right upper"`, or `bottom`,
                below the grid, `"outside lower center"`), and optional
                `title`, `columns` and `family`; their groups leave the
                legends of the plots.
            row_titles: the titles of the rows of the grid, one per row (an
                empty one for none), centred above the panels of each row
                (matplotlib's `subfig.suptitle`).
            dpi: the resolution of its PNG, `DEFAULT_DPI` without; no part of
                the spec.
            thinning: the tolerance the lines of its curves, bands and areas
                are thinned with in the spec, the SVG and the PNG, a share of
                the span of the axes (`sbmlsim.plot.thinning`); `None` keeps
                every point.
        """
        self.width = float(width)
        self.height = float(height)
        self.title = title
        self.rows = int(rows)
        self.columns = int(columns)
        self.wspace = float(wspace)
        self.hspace = float(hspace)
        self.fonts = None if fonts is None else _fonts(fonts)
        self.legend_scope = legend_scope
        self.legend_position = legend_position
        self.legend_loc = legend_loc_spec(legend_loc)
        self.legend_columns = _legend_columns(legend_columns)
        self.legend_title = legend_title
        self.legends = None if legends is None else [_plot_legend(lg) for lg in legends]
        self.row_titles = None if row_titles is None else [str(t) for t in row_titles]
        self.dpi = None if dpi is None else float(dpi)
        self.thinning = None if thinning is None else _thinning(thinning)
        self.plots: list[ChartPlot] = []
        self._element_ids: set[str] = set()
        self._created = 0

    def plot(
        self,
        row: int = 0,
        column: int = 0,
        *,
        row_span: int = 1,
        column_span: int = 1,
        title: str | None = None,
        x: ChartAxis | None = None,
        y: ChartAxis | None = None,
        y2: ChartAxis | None = None,
        legend: bool = True,
        legend_order: Sequence[str] | None = None,
        declared_legends: Sequence[Sequence[str]] | None = None,
        legend_swatches: Mapping[str, Mapping[str, Any]] | None = None,
        aspect: Aspect | None = None,
        legend_loc: str | None = None,
        legend_title: str | None = None,
        legend_columns: int | None = None,
        x2: ChartAxis | None = None,
        frame_color: str | None = None,
        frame_width: float | None = None,
        inset: tuple[ChartPlot, Sequence[float]] | None = None,
        legend_anchor: tuple[float, float] | None = None,
        legends: Sequence[Mapping[str, Any]] | None = None,
        legend_tuples: Sequence[str] | None = None,
    ) -> ChartPlot:
        """Add a plot at a cell of the grid, or as an inset of another plot.

        Args:
            row: the row of its top left cell, from 0.
            column: the column of its top left cell, from 0.
            row_span: the rows it spans.
            column_span: the columns it spans.
            title: the title of the plot.
            x: the x axis.
            y: the left y axis.
            y2: the right y axis, which elements select with `axis="y2"`.
                A category axis is best given its `categories`, so their
                order is reproducible; without them the categories follow
                the order of their first use in the data.
            legend: whether the plot has a legend.
            legend_order: the legend groups listed first, in this order.
            declared_legends: the legends the plot may show in some state of
                a page, each a list of entry labels, whose largest takes the
                room of the legend.
            legend_swatches: the swatch of a legend group by group, the style
                keywords of `ChartPlot.curve` (`color`, `opacity`, `line`,
                `width`, `marker`, `size`, `marker_fill`, `marker_edge`,
                `marker_edge_width`, `marker_fill_style`, `marker_fill_alt`,
                `fill` and `fill_opacity`) it is drawn
                with instead of the style of its element, e.g. the lines of a
                scan in one group shown in the middle colour of their map; a
                swatch without `color` takes the colour of its entry.
            aspect: `equal`, one unit of y as long as one unit of x, or a
                positive number, one unit of y that many times as long as one
                of x (matplotlib's `set_aspect` with `adjustable="datalim"`):
                the renderer keeps the panel and widens the limits of one
                axis, so e.g. a hexagon is regular; `None` for axes of their
                own limits.
            legend_loc: where the legend of the plot goes inside its panel, a
                location of `LEGEND_LOCS` (matplotlib's `loc`, e.g. `upper
                left`); `None` for the `legend_loc` of the chart. A legend of a
                row takes the one of its rightmost plot with a legend.
            legend_title: the title of the legend of the plot, matplotlib's
                `ax.legend(title=...)`; a legend of a row takes the first of
                its plots.
            legend_columns: the columns of the legend of the plot,
                matplotlib's `ncol`, filled top down and then left to right,
                inside or outside; a legend of a row takes the first of its
                plots; `None` for one.
            x2: a second x axis on top of the panel with ticks and labels of
                its own (matplotlib's `twiny`), its limits its `min` and `max`
                or those of the x axis; no element is drawn against it.
            frame_color: the color of a frame around all four sides of the
                panel instead of the axis lines (matplotlib's spines); a
                frame is drawn with `frame_color` or `frame_width`.
            frame_width: the width of the frame in pixels, 1 without.
            inset: the plot as an inset of a host plot added before it,
                `(host, (x, y, width, height))` in fractions of the panel of
                the host from its lower left corner (matplotlib's
                `ax.inset_axes`); its row and column are those of the host.
            legend_anchor: a point in fractions of the panel the legend of
                the plot is anchored at by its `legend_loc` (matplotlib's
                `bbox_to_anchor` with `loc`).
            legends: further legends of the plot in its panel, each a mapping
                of `groups` and optional `title`, `loc`, `anchor`, `columns`
                and `family` (matplotlib's second legend of an axes); their
                groups leave the legend of the plot.
            legend_tuples: the legend groups whose swatch draws the markers
                of all their elements side by side (matplotlib's
                `HandlerTuple`).

        Returns:
            The plot, which its methods add elements to.

        Raises:
            ValueError: for an inset of a host of another chart.
        """
        frame: dict[str, Any] | None = None
        if frame_color is not None or frame_width is not None:
            frame = {}
            if frame_color is not None:
                frame["color"] = _hex(frame_color, "frame")
            if frame_width is not None:
                frame["width"] = float(frame_width)
        inset_spec: dict[str, Any] | None = None
        if inset is not None:
            host, rect = inset
            if host not in self.plots:
                raise ValueError("The host of an inset is a plot of the chart.")
            x0, y0, w, h = (float(v) for v in rect)
            inset_spec = {"plot": host.id, "x": x0, "y": y0, "width": w, "height": h}
            row, column, row_span, column_span = host.row, host.column, 1, 1
        plot = ChartPlot(
            self,
            f"p{len(self.plots)}",
            row,
            column,
            row_span=row_span,
            column_span=column_span,
            title=title,
            x=x if x is not None else ChartAxis(),
            y=y if y is not None else ChartAxis(),
            y2=y2,
            legend=legend,
            legend_order=legend_order,
            declared_legends=declared_legends,
            legend_swatches=legend_swatches,
            aspect=aspect,
            legend_loc=legend_loc,
            legend_title=legend_title,
            legend_columns=legend_columns,
            x2=x2,
            frame=frame,
            inset=inset_spec,
            legend_anchor=legend_anchor,
            legends=legends,
            legend_tuples=legend_tuples,
        )
        self.plots.append(plot)
        return plot

    def _element_id(self, id: str | None) -> str:
        """Get the id of a new element, `e<n>` by default, and reserve it.

        Raises:
            ValueError: for an id the chart has already.
        """
        if id is None:
            n = self._created
            while f"e{n}" in self._element_ids:
                n += 1
            id = f"e{n}"
        elif id in self._element_ids:
            raise ValueError(f"The chart has an element of the id '{id}' already.")
        self._element_ids.add(id)
        self._created += 1
        return id

    def spec(self) -> dict[str, Any]:
        """Get the spec of the chart as a dictionary of `figure/1`.

        Returns:
            A copy of the spec with the snake case fields of the contract, an
            optional field which is `None` left out and a gap `None`.
        """
        spec: dict[str, Any] = {
            "contract": FIGURE_CONTRACT,
            "width": self.width,
            "height": self.height,
        }
        if self.title is not None:
            spec["title"] = self.title
        if self.fonts is not None:
            spec["fonts"] = dict(self.fonts)
        spec["grid"] = {
            "rows": self.rows,
            "columns": self.columns,
            "wspace": self.wspace,
            "hspace": self.hspace,
        }
        legend: dict[str, Any] = {
            "scope": self.legend_scope,
            "position": self.legend_position,
        }
        spec["legend"] = legend
        if self.legend_loc != "best":
            spec["legend"]["loc"] = self.legend_loc
        if self.legend_columns is not None:
            spec["legend"]["columns"] = self.legend_columns
        if self.legend_title is not None:
            spec["legend"]["title"] = self.legend_title
        if self.legends is not None:
            spec["legends"] = self.legends
        if self.row_titles is not None:
            spec["row_titles"] = self.row_titles
        spec["plots"] = [plot.spec() for plot in self.plots]
        return spec

    def to_json(self) -> str:
        """Get the spec as the JSON text the renderer reads.

        Returns:
            The JSON, in which a gap is `null` and never `NaN`.
        """
        return json.dumps(self.spec(), allow_nan=False)

    def svg(self) -> str:
        """Draw the chart as SVG.

        Returns:
            The SVG text.

        Raises:
            ValueError: for a spec the renderer refuses, with its path.
        """
        from sbmlsim.engine import import_core

        return str(import_core().render_svg(self.to_json()))

    def png(self, dpi: float | None = None) -> bytes:
        """Draw the chart as PNG.

        Args:
            dpi: the resolution, 72 is one pixel per pixel of the chart; the
                `dpi` of the chart without, else `DEFAULT_DPI`.

        Returns:
            The bytes of the PNG.

        Raises:
            ValueError: for a spec the renderer refuses, with its path.
        """
        from sbmlsim.engine import import_core

        if dpi is None:
            dpi = DEFAULT_DPI if self.dpi is None else self.dpi
        return _with_resolution(
            bytes(import_core().render_png(self.to_json(), dpi=float(dpi))), dpi
        )

    def save(self, path: Path | str, dpi: float | None = None) -> Path:
        """Save the chart as SVG, PNG or JSON by the suffix of the path.

        The file is written atomically.

        Args:
            path: the file, its suffix `.svg`, `.png` or `.json`.
            dpi: the resolution of a PNG, see `png`.

        Returns:
            The path of the file.

        Raises:
            ValueError: for another suffix.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".svg":
            write_text_atomic(path, self.svg())
        elif suffix == ".png":
            content = self.png(dpi)
            with atomic_path(path) as temporary:
                temporary.write_bytes(content)
        elif suffix == ".json":
            write_text_atomic(path, json.dumps(self.spec(), allow_nan=False, indent=2))
        else:
            raise ValueError(
                f"A chart is saved as {', '.join(SAVE_SUFFIXES)}, not as "
                f"'{path.suffix}': '{path}'."
            )
        return path

    def _repr_svg_(self) -> str:
        """Get the SVG of the chart, which a notebook shows."""
        return self.svg()
