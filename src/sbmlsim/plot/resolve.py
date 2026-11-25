"""The figure model of an experiment resolved into a chart of the renderer.

`figure_chart` reads the data of every curve, band and shaded area of a
`Figure` from its experiment and writes them as the elements of a `Chart`,
the spec `figure/1` which the renderer of sbmlsim-core draws:

- a `SubPlot` is a plot of the chart at its cell and spans, its `Axis`es the
  axes of the plot in the unit the data is converted to (a unit id of a
  model is resolved to its unit, shown in its short form), the right y axis
  the axis `y2`;
- a curve of points is a curve with its error bars, a curve over the points
  of a scan a curve per point in the colours, line styles and legend labels
  of `sbmlsim.plot.points.point_styles`, from `COLORBAR_FROM` points with a
  colour bar of its first dimension; a bar curve is a bar element, stacked
  bars are refused;
- a `Band` is a band of its quantiles with its median, a `ShadedArea` an area;
- the `FigureSettings` are the size (`PX_PER_INCH` pixels per inch), the
  resolution of the PNG (`fig_dpi`), the fonts (a point is a pixel), the
  spacing of the grid and the place and scope of the legends.

A legend entry is a legend group of the chart: elements of one label and one
look share their group, so the renderer shows them once, also in the legend
of a row (`legend_scope="row"`).

The renderer has no font weights, no colour of a panel or of the figure, no
style of an axis line, no location of a legend inside a plot and no line
breaks in labels; these settings of the figure model are not drawn. It has
one size of the tick labels, the larger of `xtick_labelsize` and
`ytick_labelsize`, and one switch of the ticks of an axis, so an axis whose
`ticks_visible` is false shows neither the labels nor the marks of its ticks
(matplotlib kept the marks). The grid of a plot is the one of its x axis and
of its left y axis, the right y axis draws none, as in matplotlib. A plot
has one colour bar, of its first curve or band which has one. The lines over
`COLORBAR_FROM` points or more share one legend entry, whose swatch the
renderer draws in its neutral colour, since its lines differ in colour; the
entry still toggles all of them.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from tokenize import TokenError
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from pint.errors import PintError

from sbmlsim.plot.chart import Chart, ChartAxis, ChartPlot
from sbmlsim.plot.colors import color_alpha, normalize_color
from sbmlsim.plot.padding import line_values
from sbmlsim.plot.plotting import (
    AbstractCurve,
    Axis,
    AxisScale,
    Band,
    Curve,
    CurveType,
    Figure,
    FigureSettings,
    LineType,
    MarkerType,
    ShadedArea,
    Style,
    YAxisPosition,
)
from sbmlsim.plot.points import (
    COLORMAP,
    BandLine,
    Line,
    PointStyle,
    band_label,
    band_lines,
    colorbar_range,
    curve_lines,
    point_styles,
)

if TYPE_CHECKING:
    from sbmlsim.experiment import SimulationExperiment

__all__ = ["PX_PER_INCH", "figure_chart"]

logger = logging.getLogger(__name__)

#: the pixels of a chart per inch of the figure model, so a point of a font
#: or a line is a pixel
PX_PER_INCH: Final = 72.0

#: the line types of the chart by the line types of the figure model
LINE_TYPES: Final[Mapping[LineType, str]] = {
    LineType.SOLID: "solid",
    LineType.DASH: "dashed",
    LineType.DOT: "dotted",
    LineType.DASHDOT: "dashdot",
    LineType.DASHDOTDOT: "dashdot",
    LineType.NONE: "none",
}

#: the marker types of the chart by the marker types of the figure model
MARKER_TYPES: Final[Mapping[MarkerType, str]] = {
    MarkerType.NONE: "none",
    MarkerType.CIRCLE: "circle",
    MarkerType.SQUARE: "square",
    MarkerType.DIAMOND: "diamond",
    MarkerType.TRIANGLEUP: "triangle",
    MarkerType.TRIANGLEDOWN: "triangle",
    MarkerType.TRIANGLELEFT: "triangle",
    MarkerType.TRIANGLERIGHT: "triangle",
    MarkerType.PLUS: "plus",
    MarkerType.XCROSS: "x",
    MarkerType.STAR: "cross",
    MarkerType.HDASH: "plus",
    MarkerType.VDASH: "plus",
}

#: the line types of the chart by the line styles of the points of a second
#: `over` dimension, `sbmlsim.plot.points.LINE_STYLES`
POINT_LINE_TYPES: Final[Mapping[str, str]] = {
    "-": "solid",
    "--": "dashed",
    ":": "dotted",
    "-.": "dashdot",
}

#: the colour of the legend entries of the line styles of a second dimension
NEUTRAL: Final = "#666666"

#: the colour of a marker without fill: the renderer fills every marker, the
#: background of a plot is white
HOLLOW: Final = "#ffffff"

#: the curve types the renderer draws as bars, with whether they are horizontal
BAR_TYPES: Final[Mapping[CurveType, bool]] = {
    CurveType.BAR: False,
    CurveType.HORIZONTALBAR: True,
}

#: the errors of `SimulationExperiment.Q` for a unit it cannot resolve: pint
#: for a unit it does not know or cannot parse (`UndefinedUnitError` and the
#: syntax errors are `PintError`s, the tokenizer of a text which is no
#: expression a `TokenError`), a `ValueError` for a unit id which the models
#: of the experiment define differently
_UNIT_ERRORS: Final = (PintError, ValueError, TokenError)

type AxisKey = Literal["x", "y", "y2"]


def figure_chart(
    experiment: SimulationExperiment,
    figure: Figure,
    settings: FigureSettings | Mapping[str, Any] | None = None,
) -> Chart:
    """Resolve a figure of the figure model into a chart.

    Args:
        experiment: the experiment which ran, its data and figure settings.
        figure: the figure.
        settings: the settings of the run, a `FigureSettings` or overrides of
            the defaults, which the settings of the experiment and of the
            figure override, see `Figure.resolved_settings`.

    Returns:
        The chart.

    Raises:
        ValueError: for a stacked bar curve, a curve on the right y axis of a
            plot without one, a subplot without its cell, a category axis
            which is logarithmic and the errors of `curve_lines`,
            `band_lines` and `Data.get_data`.
    """
    resolved = figure.resolved_settings(experiment, run=settings)
    width, height = figure.size(resolved)
    chart = Chart(
        width * PX_PER_INCH,
        height * PX_PER_INCH,
        title=figure.name or None,
        rows=figure.num_rows,
        columns=figure.num_cols,
        wspace=resolved.fig_subplots_wspace,
        hspace=resolved.fig_subplots_hspace,
        fonts={
            "title": resolved.fig_titlesize,
            "label": resolved.axes_labelsize,
            # figure/1 has one size of the tick labels of every axis
            "tick": max(resolved.xtick_labelsize, resolved.ytick_labelsize),
            "legend": resolved.legend_fontsize,
        },
        legend_scope=resolved.legend_scope,
        legend_position=resolved.legend_position,
        dpi=resolved.fig_dpi,
    )
    groups = _Groups()
    for subplot in figure.subplots:
        plot = subplot.plot
        if subplot.row is None or subplot.col is None:
            raise ValueError(
                f"The plot '{plot.sid}' has no row and column in the figure "
                f"'{figure.sid}'."
            )
        builder = _PlotBuilder(experiment, plot.sid or "", groups)
        y2 = builder.resolve(plot.xaxis, plot.yaxis, plot.yaxis_right, plot)
        chart_plot = chart.plot(
            subplot.row - 1,
            subplot.col - 1,
            row_span=subplot.row_span,
            column_span=subplot.col_span,
            title=plot.name if plot.name and plot.title_visible else None,
            x=builder.axis("x", plot.xaxis),
            y=builder.axis("y", plot.yaxis),
            y2=builder.axis("y2", y2) if y2 is not None else None,
            legend=plot.legend,
        )
        for mark in builder.marks:
            getattr(chart_plot, mark.method)(**mark.arguments)
        if builder.colorbar is not None:
            _colorbar(chart_plot, builder.colorbar)
    return chart


def _colorbar(plot: ChartPlot, styles: PointStyle) -> None:
    """Give a plot the colour bar of the points of a curve or band.

    The colour bar of `figure/1` is a named colour scale between two values,
    whose ticks the renderer chooses; the style of the points carries the
    ramp of their colours and the ticks and labels of their bar as well
    (`PointStyle.ramp`, `PointStyle.scale`), which a colour bar of colour
    stops and tick labels will take, see `colorbar_range`.
    """
    span = colorbar_range(styles)
    if span is None:
        return
    low, high, log = span
    plot.colorbar(COLORMAP, low, high, label=styles.title, log=log)


def _one_line(text: str | None) -> str | None:
    """Get a text as one line, `None` for no text: the renderer breaks no lines."""
    if text is None:
        return None
    return " ".join(text.split()) or None


@dataclass(frozen=True)
class _Unit:
    """The unit of an axis: the one the data is converted to and the one shown.

    Attributes:
        convert: the unit the data is converted to, `None` for its own unit.
        shown: the short form of the unit, `-` for a dimensionless one.
    """

    convert: str | None = None
    shown: str | None = None


def _unit(experiment: SimulationExperiment, unit: str | None) -> _Unit:
    """Resolve the unit of an axis, also a unit id of a model of the experiment.

    A unit pint and the models do not know is taken as it is; the conversion
    of the data reports it.
    """
    if not unit:
        return _Unit()
    try:
        units = experiment.Q(1.0, unit).units
    except _UNIT_ERRORS:
        return _Unit(unit, unit)
    return _Unit(str(units), f"{units:~P}" or "-")


@dataclass(frozen=True)
class _Mark:
    """An element of a plot: the method of `ChartPlot` and its arguments."""

    method: str
    arguments: dict[str, Any]


class _Groups:
    """The legend groups of a chart: one per label and look of its elements."""

    def __init__(self) -> None:
        """Create the groups of a chart without any."""
        self._groups: dict[tuple[str, Hashable], str] = {}
        self._names: set[str] = set()

    def of(self, label: str, look: Hashable) -> str:
        """Get the group of the elements of a label and a look.

        Args:
            label: the label of the legend entry.
            look: what the entry looks like, hashable.

        Returns:
            The group, the label for its first look and `<label> (<n>)` for
            the n-th one.
        """
        key = (label, look)
        if key not in self._groups:
            name, n = label, 1
            while name in self._names:
                n += 1
                name = f"{label} ({n})"
            self._names.add(name)
            self._groups[key] = name
        return self._groups[key]


def _style_color(color: Any) -> tuple[str | None, float]:
    """Get a colour of the figure model as `#rrggbb` and its opacity."""
    if color is None:
        return None, 1.0
    return color_alpha(color.color)


def _curve_style(style: Style | None) -> dict[str, Any]:
    """Get the style keywords of `ChartPlot.curve` of the style of a curve.

    A part the style leaves unset is left to the renderer.
    """
    if style is None:
        return {}
    resolved = style.resolve_style()
    kwargs: dict[str, Any] = {}
    if resolved.line is not None:
        color, opacity = _style_color(resolved.line.color)
        if color is not None:
            kwargs["color"] = color
            kwargs["opacity"] = opacity
        if resolved.line.type is not None:
            kwargs["line"] = LINE_TYPES[resolved.line.type]
        if resolved.line.thickness:
            kwargs["width"] = float(resolved.line.thickness)
    marker = resolved.marker
    if marker is not None:
        kwargs["marker"] = MARKER_TYPES[marker.type]
        if marker.size:
            kwargs["size"] = float(marker.size)
        if marker.fill is not None:
            fill, alpha = _style_color(marker.fill)
            kwargs["marker_fill"] = HOLLOW if fill is None or alpha == 0.0 else fill
        if marker.line_color is not None:
            edge, _ = _style_color(marker.line_color)
            if edge is not None:
                kwargs["marker_edge"] = edge
        if marker.line_thickness:
            kwargs["marker_edge_width"] = float(marker.line_thickness)
    return kwargs


def _fill_style(style: Style | None) -> dict[str, Any]:
    """Get the color and fill keywords of a bar or an area of the style of its curve.

    The color is the colour of the line, else of the fill; the fill keeps the
    opacity of its colour, which is the opacity of the element only without a
    fill, so an alpha is applied once.
    """
    if style is None:
        return {}
    resolved = style.resolve_style()
    kwargs: dict[str, Any] = {}
    line, opacity = _style_color(resolved.line.color) if resolved.line else (None, 1.0)
    fill, fill_opacity = (
        _style_color(resolved.fill.color) if resolved.fill else (None, 1.0)
    )
    if line is not None:
        kwargs["color"] = line
        if fill is None:
            kwargs["opacity"] = opacity
    elif fill is not None:
        kwargs["color"] = fill
    if fill is not None:
        kwargs["fill"] = fill
        kwargs["fill_opacity"] = fill_opacity
    return kwargs


#: the arguments of an element which are its data and not its look
_DATA_ARGUMENTS: Final = frozenset(
    {"x", "y", "low", "high", "median", "height", "xerr", "yerr", "err"}
    | {"label", "legend", "axis"}
)


def _look(arguments: Mapping[str, Any], without: tuple[str, ...] = ()) -> Hashable:
    """Get the style of the arguments of an element, hashable."""
    return tuple(
        sorted(
            (k, v)
            for k, v in arguments.items()
            if k not in _DATA_ARGUMENTS and k not in without
        )
    )


def _names(values: Any) -> bool:
    """Get whether values are names of categories, i.e. texts."""
    array = np.asarray(values)
    return (
        array.dtype.kind in "OUS"
        and array.size > 0
        and all(isinstance(v, str) for v in array.ravel().tolist())
    )


@dataclass
class _PlotBuilder:
    """The elements of a plot of the figure model, resolved before its axes.

    Attributes:
        experiment: the experiment the data is read from.
        sid: the id of the plot, for the messages.
        groups: the legend groups of the chart.
        units: the unit of every axis.
        marks: the elements, in the order they are drawn.
        names: the axes which get names of categories.
        colorbar: the style of the points whose colour bar the plot shows,
            `None` without.
    """

    experiment: SimulationExperiment
    sid: str
    groups: _Groups
    units: dict[str, _Unit] = field(default_factory=dict)
    marks: list[_Mark] = field(default_factory=list)
    names: set[str] = field(default_factory=set)
    colorbar: PointStyle | None = None

    def resolve(
        self, xaxis: Axis | None, yaxis: Axis | None, right: Axis | None, plot: Any
    ) -> Axis | None:
        """Resolve the units and the elements of a plot.

        Args:
            xaxis: the x axis.
            yaxis: the left y axis.
            right: the right y axis.
            plot: the plot of the figure model.

        Returns:
            The right y axis, `None` for a plot which draws nothing on it.

        Raises:
            ValueError: for a curve on the right y axis of a plot without one.
        """
        curves: list[AbstractCurve] = sorted(
            [*plot.curves, *plot.areas, *plot.bands],
            key=lambda c: c.order if c.order is not None else 0,
        )
        on_right = any(c.yaxis_position == YAxisPosition.RIGHT for c in curves)
        if on_right and right is None:
            raise ValueError(
                f"A curve of the plot '{self.sid}' is on the right y axis, which "
                f"the plot has not: set it with `Plot.set_yaxis_right`."
            )
        if right is not None and not on_right:
            logger.warning(
                "The plot '%s' has a right y axis, but nothing is drawn on it.",
                self.sid,
            )
            right = None
        for key, axis in (("x", xaxis), ("y", yaxis), ("y2", right)):
            self.units[key] = _unit(self.experiment, axis.unit if axis else None)
        for curve in curves:
            axis: AxisKey = "y2" if curve.yaxis_position == YAxisPosition.RIGHT else "y"
            if isinstance(curve, Curve):
                self._curve(curve, axis)
            elif isinstance(curve, Band):
                self._band(curve, axis)
            elif isinstance(curve, ShadedArea):
                self._area(curve, axis)
        return right

    def axis(self, key: AxisKey, axis: Axis | None) -> ChartAxis:
        """Get an axis of the chart of an axis of the figure model.

        A plot without the axis hides its label and ticks.

        Raises:
            ValueError: for an axis of names of categories which is logarithmic.
        """
        if axis is None:
            return ChartAxis(label_visible=False, ticks_visible=False)
        override = axis.name_override
        if override is not None:
            label, unit = _one_line(override), None
        else:
            label, unit = _one_line(axis.label), self.units[key].shown
        log = axis.scale == AxisScale.LOG10
        scale: Literal["linear", "log", "category"] = "log" if log else "linear"
        if key in self.names:
            if log:
                raise ValueError(
                    f"The axis {key} of the plot '{self.sid}' is logarithmic and "
                    f"gets names of categories."
                )
            scale = "category"
        low, high, reverse = axis.min, axis.max, axis.reverse
        if log:
            if low is not None and low <= 0.0:
                logger.warning(
                    "The log axis %s of the plot '%s' has the lower bound %s, which "
                    "is not positive and is left out.",
                    key,
                    self.sid,
                    low,
                )
                low = None
            if high is not None and high <= 0.0:
                logger.warning(
                    "The log axis %s of the plot '%s' has the upper bound %s, which "
                    "is not positive and is left out.",
                    key,
                    self.sid,
                    high,
                )
                high = None
        if low is not None and high is not None and low > high:
            # the bounds of matplotlib in the other direction draw it reversed
            low, high, reverse = high, low, not reverse
        return ChartAxis(
            label=label,
            unit=unit,
            scale=scale,
            min=low,
            max=high,
            reverse=reverse,
            # the grid of a plot is the one of its left y axis, as matplotlib
            # drew it
            grid=axis.grid and key != "y2",
            label_visible=axis.label_visible,
            ticks_visible=axis.ticks_visible,
        )

    def _data(self, curve: AbstractCurve, data: Any, axis: AxisKey) -> Any:
        """Get the values of data of a curve in the unit of an axis, `None` without data."""
        if data is None:
            return None
        return data.get_data(
            experiment=self.experiment, to_units=self.units[axis].convert
        )

    def _positions(self, axis: AxisKey, values: Any) -> Any:
        """Note that an axis gets names of categories, if the values are names."""
        if _names(values):
            self.names.add(axis)
        return values

    def _group(self, label: str, arguments: Mapping[str, Any], kind: str) -> str:
        """Get the legend group of an element with a label and its arguments.

        The look of an element is its kind, its style and its error bars; an
        element without a colour takes the colour of the renderer at its
        position, so its position is part of its look.
        """
        look: tuple[Any, ...] = (
            kind,
            _look(arguments),
            arguments.get("xerr") is not None,
            arguments.get("yerr") is not None,
            arguments.get("median") is not None,
        )
        if "color" not in arguments:
            look = (*look, self.sid, len(self.marks))
        return self.groups.of(label, look)

    def _add(
        self,
        method: str,
        arguments: dict[str, Any],
        label: str | None,
        legend: str | None = None,
    ) -> None:
        """Add an element, in the legend group of its label and look by default."""
        if label is not None:
            arguments["label"] = label
            arguments["legend"] = (
                legend if legend is not None else self._group(label, arguments, method)
            )
        elif legend is not None:
            arguments["legend"] = legend
        self.marks.append(_Mark(method, arguments))

    def _set_colorbar(self, styles: PointStyle, color: str | None) -> None:
        """Give the plot the colour bar of the points of a curve or band.

        The points of a colour of their own are its shades, which no colour
        scale of the renderer has; they get no colour bar. A plot has one
        colour bar, the one of its first curve which has one.
        """
        span = colorbar_range(styles) if color is None else None
        if span is None:
            logger.debug(
                "The points of '%s' in the plot '%s' get no colour bar.",
                styles.title,
                self.sid,
            )
            return
        if self.colorbar is not None:
            logger.warning(
                "The plot '%s' shows the colour bar of '%s'; the colour bar of "
                "'%s' is not drawn, a plot has one colour bar.",
                self.sid,
                self.colorbar.title,
                styles.title,
            )
            return
        self.colorbar = styles

    def _scan(self, curve: Curve | Band, over: Sequence[str]) -> tuple[list[Any], Any]:
        """Get the dimensions of a scan of `over` and the units of the model of a curve."""
        task = curve.y.task_id or curve.x.task_id
        dimensions = [
            self.experiment.scan_dimension(task, d) if task else None for d in over
        ]
        units = self.experiment.model_units(task) if task else {}
        return dimensions, units

    def _curve(self, curve: Curve, axis: AxisKey) -> None:
        """Add the elements of a curve: a curve, bars or a curve per point."""
        sid = curve.sid or curve.name or ""
        if curve.type not in BAR_TYPES and curve.type != CurveType.POINTS:
            raise ValueError(
                f"The curve '{sid}' is of the type {curve.type.name}: stacked bars "
                f"are not drawn, draw bars side by side ({CurveType.BAR.name}) or "
                f"points ({CurveType.POINTS.name})."
            )
        x = self._data(curve, curve.x, "x")
        y = self._data(curve, curve.y, axis)
        xerr = self._data(curve, curve.xerr, "x")
        yerr = self._data(curve, curve.yerr, axis)
        lines = curve_lines(sid, curve.over, x, y, xerr, yerr)
        if curve.type in BAR_TYPES:
            (line,) = lines
            self._bar(curve, line, axis, BAR_TYPES[curve.type])
            return
        style = _curve_style(curve.style)
        if not curve.over:
            (line,) = lines
            self._add(
                "curve",
                {
                    "x": self._positions("x", line.x),
                    "y": line.y,
                    "xerr": line.xerr,
                    "yerr": line.yerr,
                    "axis": axis,
                    **style,
                },
                curve.name or None,
            )
            return
        self._points(curve, lines, style, axis)

    def _bar(self, curve: Curve, line: Line, axis: AxisKey, horizontal: bool) -> None:
        """Add a bar element, the positions on the y axis for horizontal bars."""
        if line.xerr is not None:
            logger.warning(
                "The bars '%s' of the plot '%s' draw the errors of their heights, "
                "not of their positions.",
                curve.sid or curve.name,
                self.sid,
            )
        self._add(
            "bar",
            {
                "x": self._positions(axis if horizontal else "x", line.x),
                "height": line.y,
                "err": line.yerr,
                "horizontal": horizontal,
                "axis": axis,
                **_fill_style(curve.style),
            },
            curve.name or None,
        )

    def _points(
        self,
        curve: Curve,
        lines: list[Line],
        style: dict[str, Any],
        axis: AxisKey,
    ) -> None:
        """Add a curve per point of the `over` dimensions of a curve.

        Under `COLORBAR_FROM` points of one dimension every line has its own
        legend entry, labelled by its point; with two dimensions the entries
        name the colours of the first and the line styles of the second; from
        `COLORBAR_FROM` points the lines share one entry of the name of the
        curve, if it has one, and a colour bar shows the first dimension.
        """
        dimensions, units = self._scan(curve, curve.over)
        color = style.get("color")
        styles = point_styles(lines, curve.over, color, dimensions, units)
        prefix = f"{curve.name}, " if curve.name else ""
        two = styles.linestyles is not None
        shared = (
            self.groups.of(curve.name, ("over", _look(style, ("color",))))
            if styles.colorbar and curve.name
            else None
        )
        colors = {}
        for k, line in enumerate(lines):
            i = line.index[0]
            arguments = {
                "x": self._positions("x", line.x),
                "y": line.y,
                "xerr": line.xerr,
                "yerr": line.yerr,
                "axis": axis,
                **style,
                "color": styles.colors[i],
            }
            if "marker_fill" in arguments:
                arguments["marker_fill"] = styles.colors[i]
            if two and styles.linestyles is not None:
                arguments["line"] = POINT_LINE_TYPES[styles.linestyles[line.index[1]]]
            if styles.colorbar:
                label = curve.name if k == 0 and shared else None
                self._add("curve", arguments, label, shared)
            elif two:
                text = f"{prefix}{styles.labels[0][i]}"
                colors[i] = self._group(text, {**arguments, "line": "solid"}, "point")
                self._add("curve", arguments, None, colors[i])
            else:
                self._add("curve", arguments, f"{prefix}{styles.labels[0][i]}")
        if two and styles.linestyles is not None:
            proxy = {k: v for k, v in style.items() if k != "line"}
            if not styles.colorbar:
                for i, text in enumerate(styles.labels[0]):
                    arguments = {
                        **proxy,
                        "x": [],
                        "y": [],
                        "axis": axis,
                        "color": styles.colors[i],
                        "line": "solid",
                    }
                    if "marker_fill" in arguments:
                        arguments["marker_fill"] = styles.colors[i]
                    self._add("curve", arguments, f"{prefix}{text}", colors.get(i))
            for linestyle, text in zip(
                styles.linestyles, styles.labels[1], strict=True
            ):
                arguments = {
                    **proxy,
                    "x": [],
                    "y": [],
                    "axis": axis,
                    "color": NEUTRAL,
                    "line": POINT_LINE_TYPES[linestyle],
                }
                if "marker_fill" in arguments:
                    arguments["marker_fill"] = NEUTRAL
                self._add("curve", arguments, text)
        if styles.colorbar:
            self._set_colorbar(styles, color)

    def _band(self, band: Band, axis: AxisKey) -> None:
        """Add a band per point of `over`: its quantiles, filled, and its median.

        The legend entry of a band is its area, labelled `<name> (<low>-<high> %)`
        (`<name>, <point label> (<low>-<high> %)` with `over`); from
        `COLORBAR_FROM` points the bands share one entry of the name.
        """
        sid = band.sid or band.name or ""
        x = self._data(band, band.x, "x")
        y = self._data(band, band.y, axis)
        bands: list[BandLine] = band_lines(
            sid, band.over, band.across, band.quantiles, x, y
        )
        color = normalize_color(band.color) if band.color else None
        styles: PointStyle | None = None
        if band.over:
            dimensions, units = self._scan(band, band.over)
            styles = point_styles(bands, band.over, color, dimensions, units)
        shared = (
            self.groups.of(
                band_label(band.name, band.quantiles),
                ("bands", band.alpha, band.median),
            )
            if styles is not None and styles.colorbar and band.name
            else None
        )
        for k, b in enumerate(bands):
            arguments: dict[str, Any] = {
                "x": b.x,
                "low": b.low,
                "high": b.high,
                "median": b.median if band.median else None,
                "fill_opacity": band.alpha,
                "axis": axis,
            }
            c = styles.colors[b.index[0]] if styles is not None else color
            if c is not None:
                arguments["color"] = c
            if styles is not None and styles.colorbar:
                label = (
                    band_label(band.name, band.quantiles) if k == 0 and shared else None
                )
                self._add("band", arguments, label, shared)
                continue
            name = (
                f"{band.name}, {styles.labels[0][b.index[0]]}"
                if styles is not None
                else band.name
            )
            self._add("band", arguments, band_label(name, band.quantiles))
        if styles is not None and styles.colorbar:
            self._set_colorbar(styles, color)

    def _area(self, area: ShadedArea, axis: AxisKey) -> None:
        """Add the area of a shaded area."""
        x, low, high = line_values(
            area.sid or area.name or "",
            self._data(area, area.x, "x"),
            self._data(area, area.yfrom, axis),
            self._data(area, area.yto, axis),
        )
        self._add(
            "area",
            {"x": x, "low": low, "high": high, "axis": axis, **_fill_style(area.style)},
            area.name or None,
        )
