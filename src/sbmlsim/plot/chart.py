"""The chart builder of the figure contract `figure/1` of sbmlsim-core.

A `Chart` is a figure of plots in a grid, which the renderer of sbmlsim-core
draws as SVG or PNG. Its plots are `ChartPlot`s with the axes `ChartAxis` and
elements of the kinds of the contract (curves, bands, bars, box plots, heat
maps, ...), added by the methods of the plot, each returning the id of its
element. The data of an element are sequences, numpy arrays, pint quantities
or xarray data arrays (`attrs["units"]` their unit): an axis with a unit
converts every quantity on it to that unit and takes plain values as values
in it, an axis without a unit takes the unit of the first quantity placed on
it, and the renderer shows the axis as `<label> [<unit>]`:

    chart = Chart(title="HCTZ")
    plot = chart.plot(x=ChartAxis(label="time", unit="hr"), y=ChartAxis(label="cp"))
    plot.curve(res.q["time"], res.q["cp"], label="plasma")
    chart.save("hctz.svg")

`Chart.spec` is the spec as a dictionary of the snake case fields of
`figure/1`, `Chart.to_json` its JSON, in which a value which is not finite is
`null`, a gap. The renderer is imported through `sbmlsim.engine.import_core`
only when a chart is drawn.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tokenize import TokenError
from typing import Any, Final, Literal

import numpy as np
import xarray as xr
from pint.errors import DimensionalityError, PintError

from sbmlsim.atomic import atomic_path, write_text_atomic
from sbmlsim.units import Quantity, short_unit, ureg

__all__ = ["Chart", "ChartAxis", "ChartPlot", "axis_label"]

#: The contract of the spec of a chart.
FIGURE_CONTRACT: Final = "figure/1"

#: The suffixes `Chart.save` writes.
SAVE_SUFFIXES: Final = (".svg", ".png", ".json")

#: The dpi of a PNG which is saved without one.
DEFAULT_DPI: Final = 144.0

#: The data of an element: a sequence, a numpy array, a pint quantity or an
#: xarray data array.
type Values = Any

type AxisName = Literal["x", "y", "y2"]


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
        label: the label, which the renderer shows with the unit.
        unit: the unit, which the quantities on the axis are converted to;
            `None` takes the unit of the first quantity on the axis.
        scale: `linear`, `log` or `category`.
        categories: the names of the categories of a category axis, which
            are at the positions 0, 1, ...
        min: the lower end of the axis.
        max: the upper end of the axis.
        reverse: whether the axis is reversed.
        grid: whether the plot draws grid lines of the axis.
        label_visible: whether the label is shown.
        ticks_visible: whether the ticks are shown.
        tick_rotation: the rotation of the tick labels in degrees.
        max_decades: the decades of a log axis below its largest value,
            `None` for the default of the renderer.
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
    tick_rotation: float = 0.0
    max_decades: float | None = None

    def __post_init__(self) -> None:
        """Keep the categories as a tuple."""
        if self.categories is not None:
            object.__setattr__(self, "categories", tuple(self.categories))


@dataclass
class _AxisState:
    """An axis of a plot with the unit its values are in."""

    name: AxisName
    axis: ChartAxis
    unit: Any = None

    def __post_init__(self) -> None:
        """Parse the unit of the axis; one pint does not parse is shown as given."""
        if self.axis.unit:
            try:
                self.unit = ureg.Unit(self.axis.unit)
            except PintError, ValueError, TypeError, AttributeError, TokenError:
                self.unit = None

    def spec(self) -> dict[str, Any]:
        """Get the axis as a dictionary of `figure/1`."""
        a = self.axis
        unit = short_unit(str(self.unit)) if self.unit is not None else a.unit
        spec: dict[str, Any] = {}
        if a.label is not None:
            spec["label"] = a.label
        if unit:
            spec["unit"] = unit
        spec["scale"] = a.scale
        if a.categories is not None:
            spec["categories"] = list(a.categories)
        if a.min is not None:
            spec["min"] = float(a.min)
        if a.max is not None:
            spec["max"] = float(a.max)
        if a.max_decades is not None:
            spec["max_decades"] = float(a.max_decades)
        spec.update(
            reverse=a.reverse,
            grid=a.grid,
            label_visible=a.label_visible,
            ticks_visible=a.ticks_visible,
            tick_rotation=float(a.tick_rotation),
        )
        return spec


def _numbers(values: np.ndarray) -> list[float | None]:
    """Get the numbers of an array as a list, a value which is not finite `None`."""
    flat = np.asarray(values, dtype=float).ravel()
    return [float(v) if math.isfinite(v) else None for v in flat.tolist()]


def _as_quantity(data: Values) -> Any:
    """Get data as a pint quantity, or as an array without a unit."""
    if isinstance(data, Quantity):
        return data
    if isinstance(data, xr.DataArray):
        units = data.attrs.get("units")
        values = np.asarray(data.values, dtype=float)
        return ureg.Quantity(values, units) if units else values
    return np.asarray(data, dtype=float)


def _magnitude(data: Values) -> np.ndarray:
    """Get the numbers of data in its own unit, for values on no axis."""
    data = _as_quantity(data)
    if isinstance(data, Quantity):
        return np.asarray(data.magnitude, dtype=float)
    return data


class ChartPlot:
    """A plot of a chart, which its methods add elements to.

    A plot is created by `Chart.plot`; every method adds one element and
    returns its id.
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
    ) -> None:
        """Create a plot, see `Chart.plot`."""
        self.chart = chart
        self.id = id
        self.row = row
        self.column = column
        self.row_span = row_span
        self.column_span = column_span
        self.title = title
        self.legend = legend
        self.legend_order = None if legend_order is None else list(legend_order)
        self._axes: dict[str, _AxisState] = {
            "x": _AxisState("x", x),
            "y": _AxisState("y", y),
        }
        if y2 is not None:
            self._axes["y2"] = _AxisState("y2", y2)
        self._elements: list[dict[str, Any]] = []
        self._colorbar: dict[str, Any] | None = None

    def _state(self, axis: AxisName) -> _AxisState:
        """Get an axis of the plot."""
        if axis not in self._axes:
            raise ValueError(f"The plot '{self.id}' has no axis '{axis}'.")
        return self._axes[axis]

    def _on(self, axis: AxisName, data: Values) -> np.ndarray:
        """Get data as numbers in the unit of an axis, which may take its unit.

        Raises:
            pint.DimensionalityError: for a quantity which the unit of the
                axis cannot take.
        """
        state = self._state(axis)
        data = _as_quantity(data)
        if not isinstance(data, Quantity):
            return data
        if state.unit is None:
            if state.axis.unit:
                raise ValueError(
                    f"The axis {axis} of the plot '{self.id}' has the unit "
                    f"'{state.axis.unit}', which pint does not know, so a quantity "
                    "cannot be converted to it."
                )
            state.unit = data.units
            return np.asarray(data.magnitude, dtype=float)
        try:
            return np.asarray(data.to(state.unit).magnitude, dtype=float)
        except DimensionalityError as error:
            raise DimensionalityError(
                error.units1,
                error.units2,
                error.dim1,
                error.dim2,
                extra_msg=f" on the axis {axis} of the plot '{self.id}'",
            ) from error

    def _scalar(self, axis: AxisName, value: Values) -> float:
        """Get one value in the unit of an axis."""
        return float(np.asarray(self._on(axis, value), dtype=float).ravel()[0])

    def _add(
        self,
        kind: str,
        data: dict[str, Any],
        *,
        label: str | None,
        legend: str | None,
        id: str | None,
        axis: AxisName,
        style: dict[str, Any],
    ) -> str:
        """Add an element of a kind with its data and return its id."""
        if axis != "y":
            self._state(axis)
        element_id = self.chart._element_id(id)
        element: dict[str, Any] = {"id": element_id, "kind": kind}
        if label is not None:
            element["label"] = label
        if legend is not None:
            element["legend"] = legend
        element["axis"] = axis
        element["style"] = style
        element.update(data)
        self._elements.append(element)
        return element_id

    @staticmethod
    def _style(color: str | None, opacity: float) -> dict[str, Any]:
        """Get the style of a color and an opacity."""
        style: dict[str, Any] = {}
        if color is not None:
            style["color"] = color
        style["opacity"] = float(opacity)
        return style

    def curve(
        self,
        x: Values,
        y: Values,
        *,
        xerr: Values | None = None,
        yerr: Values | None = None,
        line: str = "solid",
        width: float = 2.0,
        marker: str = "none",
        size: float = 6.0,
        fill: str | None = None,
        edge: str | None = None,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add a curve, a line and/or markers with error bars.

        Args:
            x: the x values.
            y: the y values.
            xerr: the errors of x.
            yerr: the errors of y.
            line: the type of the line, `solid`, `dashed`, `dotted`,
                `dashdot` or `none`.
            width: the width of the line in pixels.
            marker: the type of the markers, `none`, `circle`, `square`,
                `triangle`, `diamond`, `cross`, `plus` or `x`.
            size: the size of the markers in pixels.
            fill: the fill color of the markers.
            edge: the edge color of the markers.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color `#rrggbb`, the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.
        """
        data: dict[str, Any] = {
            "x": _numbers(self._on("x", x)),
            "y": _numbers(self._on(axis, y)),
        }
        if xerr is not None:
            data["xerr"] = _numbers(self._on("x", xerr))
        if yerr is not None:
            data["yerr"] = _numbers(self._on(axis, yerr))
        style = self._style(color, opacity)
        style["line"] = {"type": line, "width": float(width)}
        markers: dict[str, Any] = {"type": marker, "size": float(size)}
        if fill is not None:
            markers["fill"] = fill
        if edge is not None:
            markers["edge"] = edge
        style["marker"] = markers
        return self._add(
            "curve", data, label=label, legend=legend, id=id, axis=axis, style=style
        )

    def band(
        self,
        x: Values,
        low: Values,
        high: Values,
        *,
        median: Values | None = None,
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
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color `#rrggbb`, the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.
        """
        data: dict[str, Any] = {
            "x": _numbers(self._on("x", x)),
            "low": _numbers(self._on(axis, low)),
            "high": _numbers(self._on(axis, high)),
        }
        if median is not None:
            data["median"] = _numbers(self._on(axis, median))
        return self._add(
            "band",
            data,
            label=label,
            legend=legend,
            id=id,
            axis=axis,
            style=self._style(color, opacity),
        )

    def area(
        self,
        x: Values,
        low: Values,
        high: Values,
        *,
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
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color `#rrggbb`, the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.
        """
        data = {
            "x": _numbers(self._on("x", x)),
            "from": _numbers(self._on(axis, low)),
            "to": _numbers(self._on(axis, high)),
        }
        return self._add(
            "area",
            data,
            label=label,
            legend=legend,
            id=id,
            axis=axis,
            style=self._style(color, opacity),
        )

    def bar(
        self,
        x: Values,
        height: Values,
        *,
        err: Values | None = None,
        horizontal: bool = False,
        bar_width: float | None = None,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add bars at positions or at the indices of categories.

        Args:
            x: the positions, on the y axis for horizontal bars.
            height: the heights, on the x axis for horizontal bars.
            err: the errors of the heights.
            horizontal: whether the bars are horizontal.
            bar_width: the width of a bar in units of the axis of the
                positions, the default of the renderer for `None`.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color `#rrggbb`, the cycle by default.
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
            data["err"] = _numbers(self._on(heights, err))
        data["horizontal"] = horizontal
        if bar_width is not None:
            data["width"] = float(bar_width)
        return self._add(
            "bar",
            data,
            label=label,
            legend=legend,
            id=id,
            axis=axis,
            style=self._style(color, opacity),
        )

    def box(
        self,
        positions: Values,
        stats: Sequence[Mapping[str, Any]],
        *,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add box plots, one per position.

        Args:
            positions: the positions on the x axis.
            stats: the statistics of every box: `low_whisker`, `q1`,
                `median`, `q3`, `high_whisker` and optional `outliers`,
                values on the y axis.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color `#rrggbb`, the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.
        """
        boxes: list[dict[str, Any]] = []
        for stat in stats:
            box: dict[str, Any] = {
                key: self._scalar(axis, stat[key])
                for key in ("low_whisker", "q1", "median", "q3", "high_whisker")
            }
            box["outliers"] = _numbers(self._on(axis, stat.get("outliers", [])))
            boxes.append(box)
        data = {"positions": _numbers(self._on("x", positions)), "stats": boxes}
        return self._add(
            "box",
            data,
            label=label,
            legend=legend,
            id=id,
            axis=axis,
            style=self._style(color, opacity),
        )

    def violin(
        self,
        positions: Values,
        densities: Sequence[Mapping[str, Any]],
        *,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add violins, one per position.

        Args:
            positions: the positions on the x axis.
            densities: the density of every violin, `values` on the y axis
                and their `density`, of one length.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color `#rrggbb`, the cycle by default.
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
        return self._add(
            "violin",
            data,
            label=label,
            legend=legend,
            id=id,
            axis=axis,
            style=self._style(color, opacity),
        )

    def histogram(
        self,
        edges: Values,
        counts: Values,
        *,
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
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color `#rrggbb`, the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.
        """
        data = {
            "edges": _numbers(self._on("x", edges)),
            "counts": _numbers(self._on(axis, counts)),
        }
        return self._add(
            "histogram",
            data,
            label=label,
            legend=legend,
            id=id,
            axis=axis,
            style=self._style(color, opacity),
        )

    def heatmap(
        self,
        rows: Sequence[str],
        columns: Sequence[str],
        values: Values,
        *,
        text: Sequence[Sequence[str]] | None = None,
        scale: str = "viridis",
        low: float | None = None,
        high: float | None = None,
        center: float | None = None,
        log: bool = False,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add a heat map.

        Args:
            rows: the names of the rows.
            columns: the names of the columns.
            values: the values, a row per name of `rows` of the length of
                `columns`.
            text: the text of the cells, in the shape of `values`.
            scale: the name of the color scale, `viridis`, `magma`, `rdbu`,
                `coolwarm` or `greys`.
            low: the value of the lower end of the scale, the smallest
                finite value (positive on a log scale) by default.
            high: the value of the upper end of the scale, the largest
                finite value by default.
            center: the value at the middle of a diverging scale.
            log: whether the scale is logarithmic.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color `#rrggbb`, the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.
        """
        cells = [_numbers(_magnitude(row)) for row in _magnitude_rows(values)]
        data: dict[str, Any] = {
            "rows": [str(r) for r in rows],
            "columns": [str(c) for c in columns],
            "values": cells,
        }
        if text is not None:
            data["text"] = [[str(t) for t in row] for row in text]
        data["scale"] = _color_scale(scale, low, high, center, log, cells)
        return self._add(
            "heatmap",
            data,
            label=label,
            legend=legend,
            id=id,
            axis=axis,
            style=self._style(color, opacity),
        )

    def text(
        self,
        x: Values,
        y: Values,
        text: str,
        *,
        anchor: Literal["start", "middle", "end"] = "start",
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add a text at a position.

        Args:
            x: the x position.
            y: the y position.
            text: the text.
            anchor: the anchor of the text at the position, `start`,
                `middle` or `end`.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color `#rrggbb`, the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.
        """
        data = {
            "x": self._scalar("x", x),
            "y": self._scalar(axis, y),
            "text": text,
            "anchor": anchor,
        }
        return self._add(
            "text",
            data,
            label=label,
            legend=legend,
            id=id,
            axis=axis,
            style=self._style(color, opacity),
        )

    def rule(
        self,
        value: Values,
        *,
        orientation: Literal["horizontal", "vertical"] = "horizontal",
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
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color `#rrggbb`, the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.
        """
        on: AxisName = axis if orientation == "horizontal" else "x"
        data = {"orientation": orientation, "value": self._scalar(on, value)}
        return self._add(
            "rule",
            data,
            label=label,
            legend=legend,
            id=id,
            axis=axis,
            style=self._style(color, opacity),
        )

    def polygon(
        self,
        paths: Sequence[Values],
        *,
        label: str | None = None,
        legend: str | None = None,
        id: str | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        axis: Literal["y", "y2"] = "y",
    ) -> str:
        """Add filled polygons.

        Args:
            paths: the rings, each of `[x, y]` pairs with at least 3 points.
            label: the label in the legend.
            legend: the legend group.
            id: the id, `e<n>` by default.
            color: the color `#rrggbb`, the cycle by default.
            opacity: the opacity.
            axis: the y axis, `y` or `y2`.

        Returns:
            The id of the element.
        """
        rings: list[list[list[float | None]]] = []
        for path in paths:
            points = np.asarray(path, dtype=float).reshape(-1, 2)
            xs = _numbers(points[:, 0])
            ys = _numbers(points[:, 1])
            rings.append([[px, py] for px, py in zip(xs, ys, strict=True)])
        return self._add(
            "polygon",
            {"paths": rings},
            label=label,
            legend=legend,
            id=id,
            axis=axis,
            style=self._style(color, opacity),
        )

    def colorbar(
        self,
        scale: str,
        low: float,
        high: float,
        *,
        label: str | None = None,
        center: float | None = None,
        log: bool = False,
    ) -> None:
        """Give the plot a color bar, which is no element.

        Args:
            scale: the name of the color scale.
            low: the value of the lower end of the scale.
            high: the value of the upper end of the scale.
            label: the label of the color bar.
            center: the value at the middle of a diverging scale.
            log: whether the scale is logarithmic.
        """
        colorbar: dict[str, Any] = {
            "scale": _color_scale(scale, low, high, center, log, [])
        }
        if label is not None:
            colorbar["label"] = label
        self._colorbar = colorbar

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
        spec["legend"] = self.legend
        if self._colorbar is not None:
            spec["colorbar"] = self._colorbar
        if self.legend_order is not None:
            spec["legend_order"] = list(self.legend_order)
        spec["elements"] = [dict(e) for e in self._elements]
        return spec


def _magnitude_rows(values: Values) -> list[Any]:
    """Get the rows of the values of a heat map, which may differ in length."""
    if isinstance(values, Quantity | xr.DataArray | np.ndarray):
        return list(np.atleast_2d(_magnitude(values)))
    return list(values)


def _color_scale(
    name: str,
    low: float | None,
    high: float | None,
    center: float | None,
    log: bool,
    cells: Sequence[Sequence[float | None]],
) -> dict[str, Any]:
    """Get a color scale, its ends by default the range of the finite values."""
    finite = [v for row in cells for v in row if v is not None and (v > 0 or not log)]
    if low is None:
        low = min(finite) if finite else (1.0 if log else 0.0)
    if high is None:
        high = max(finite) if finite else low * 10.0 if log else low + 1.0
    scale: dict[str, Any] = {"name": name, "low": float(low), "high": float(high)}
    if center is not None:
        scale["center"] = float(center)
    scale["log"] = log
    return scale


class Chart:
    """A figure of plots in a grid, the spec `figure/1` of sbmlsim-core.

    Plots are added with `plot`, elements with the methods of the plots;
    `svg`, `png` and `save` draw it with the renderer of sbmlsim-core.
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
        fonts: Mapping[str, float] | None = None,
        legend_scope: str = "plot",
        legend_position: str = "inside",
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
            fonts: the font sizes in points by `title`, `label`, `tick` and
                `legend`, the defaults of the renderer for the others.
            legend_scope: `plot`, a legend per plot, or `row`, one per row.
            legend_position: `inside` or `outside`, right of the plot.
        """
        self.width = width
        self.height = height
        self.title = title
        self.rows = rows
        self.columns = columns
        self.wspace = wspace
        self.hspace = hspace
        self.fonts = None if fonts is None else dict(fonts)
        self.legend_scope = legend_scope
        self.legend_position = legend_position
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
    ) -> ChartPlot:
        """Add a plot at a cell of the grid.

        Args:
            row: the row of its top left cell, from 0.
            column: the column of its top left cell, from 0.
            row_span: the rows it spans.
            column_span: the columns it spans.
            title: the title of the plot.
            x: the x axis.
            y: the left y axis.
            y2: the right y axis, which elements select with `axis="y2"`.
            legend: whether the plot has a legend.
            legend_order: the legend groups listed first, in this order.

        Returns:
            The plot, which its methods add elements to.
        """
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
            The spec with the snake case fields of the contract, an optional
            field which is `None` left out and a gap `None`.
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
        spec["legend"] = {
            "scope": self.legend_scope,
            "position": self.legend_position,
        }
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

    def png(self, dpi: float = DEFAULT_DPI) -> bytes:
        """Draw the chart as PNG.

        Args:
            dpi: the resolution, 72 is one pixel per pixel of the chart.

        Returns:
            The bytes of the PNG.

        Raises:
            ValueError: for a spec the renderer refuses, with its path.
        """
        from sbmlsim.engine import import_core

        return bytes(import_core().render_png(self.to_json(), dpi=float(dpi)))

    def save(self, path: Path | str, dpi: float | None = None) -> Path:
        """Save the chart as SVG, PNG or JSON by the suffix of the path.

        The file is written atomically.

        Args:
            path: the file, its suffix `.svg`, `.png` or `.json`.
            dpi: the resolution of a PNG, 144 by default.

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
            content = self.png(DEFAULT_DPI if dpi is None else dpi)
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
