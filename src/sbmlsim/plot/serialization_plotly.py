"""Rendering the figures of an experiment as interactive plotly figures.

`sbmlsim.plot.plotting` describes a figure without saying how it is drawn and
`serialization_matplotlib` is one way of drawing it. This is a second one: the
same `Figure` becomes a plotly figure, which is written as an HTML page the
reader can zoom, pan and hover, and whose legend switches curves on and off.

It exists because rendering is what a report spends its time on. Building a
matplotlib figure of four panels with six curves takes 17 ms and turning it
into SVG takes another 156 ms, while the same figure as plotly HTML takes
17 ms in total: a plotly figure carries its data and the browser draws it.

The javascript is written once next to the pages (`include_plotlyjs` of
`write_html`), so a report still loads nothing from the network.

!!! warning "Prototype"
    plotly is imported where it is used and is in the `dev` extra, not a
    dependency of `sbmlsim`. This is a prototype to judge the approach on
    real figures. It covers the
    curves, the shaded areas, the axes and the styles the experiments of
    `examples/` use; it is not a complete replacement of the matplotlib
    serializer, and plotly is not a dependency of `sbmlsim`.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from matplotlib.colors import Colormap, LogNorm, to_hex, to_rgb

from sbmlsim.plot.padding import line_values
from sbmlsim.plot.plotting import (
    Axis,
    AxisScale,
    Band,
    Curve,
    CurveType,
    Figure,
    LineType,
    MarkerType,
    ShadedArea,
    Style,
    SubPlot,
    YAxisPosition,
)
from sbmlsim.plot.points import (
    BandLine,
    Line,
    PointStyle,
    band_lines,
    curve_lines,
    cycle_color,
    format_value,
    point_styles,
    tick_positions,
)

logger = logging.getLogger(__name__)

#: dash of a line, by the line type of the figure model
DASH_BY_LINE: dict[LineType, str] = {
    LineType.NONE: "solid",
    LineType.SOLID: "solid",
    LineType.DOT: "dot",
    LineType.DASH: "dash",
    LineType.DASHDOT: "dashdot",
    LineType.DASHDOTDOT: "longdashdot",
}

#: symbol of a marker, by the marker type of the figure model
SYMBOL_BY_MARKER: dict[MarkerType, str] = {
    MarkerType.NONE: "circle",
    MarkerType.SQUARE: "square",
    MarkerType.CIRCLE: "circle",
    MarkerType.DIAMOND: "diamond",
    MarkerType.XCROSS: "x",
    MarkerType.PLUS: "cross",
    MarkerType.STAR: "star",
    MarkerType.TRIANGLEUP: "triangle-up",
    MarkerType.TRIANGLEDOWN: "triangle-down",
    MarkerType.TRIANGLELEFT: "triangle-left",
    MarkerType.TRIANGLERIGHT: "triangle-right",
    MarkerType.HDASH: "line-ew",
    MarkerType.VDASH: "line-ns",
}


#: the dash of plotly of every line style of `sbmlsim.plot.points.LINE_STYLES`
DASH_BY_LINESTYLE = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}

#: the width of the lines of the quantiles of a band
_BAND_EDGE_WIDTH = 0.6

#: the margins of the plot area of plotly in pixels, left and right
_MARGIN = 80

#: the gap in pixels between a panel (or the ticks and the title of its right y
#: axis), each of its colour bars and the legend
_GAP = 10

#: the width of a colour bar in pixels
_BAR_THICKNESS = 15

#: the padding of plotly left and right of a colour bar in pixels (`xpad`)
_BAR_PAD = 10

#: the length and the padding of the ticks of a colour bar in pixels
_BAR_TICKS = 8

#: the width in pixels of the title of a colour bar, turned right of its ticks
_BAR_TITLE = 26

#: the width in pixels of a character of a tick label or a legend entry
_CHAR = 7.0

#: the width in pixels of the tick labels and the title of a right y axis
_RIGHT_AXIS = 80

#: the width in pixels of the symbol and the padding of a legend entry
_LEGEND_SYMBOL = 50


def _rgba(color: str, alpha: float) -> str:
    """Get a colour with an opacity as the rgba string of plotly."""
    r, g, b = (round(255 * c) for c in to_rgb(color))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _colorscale(colormap: Colormap) -> list[list[Any]]:
    """Sample a colour map of matplotlib into a colour scale of plotly."""
    return [[float(f), to_hex(colormap(float(f)))] for f in np.linspace(0.0, 1.0, 21)]


def _legend_trace(
    name: str, color: str, dash: str, style: Style | None, group: str | None = None
) -> Any:
    """Get a trace without data which only gives a legend entry."""
    import plotly.graph_objects as go

    return go.Scatter(
        x=[None],
        y=[None],
        mode=_mode(style),
        line={**_line_options(style), "color": color, "dash": dash},
        marker={**_marker_options(style), "color": color},
        name=name,
        legendgroup=group,
        showlegend=True,
        hoverinfo="skip",
    )


def _log_ticks(vmin: float, vmax: float) -> list[float]:
    """Get the powers of ten between two positive values, the ticks of a log scale."""
    low = math.ceil(math.log10(vmin) - 1e-9)
    high = math.floor(math.log10(vmax) + 1e-9)
    return [10.0**k for k in range(low, high + 1)]


def _colorbar_trace(styles: PointStyle) -> Any:
    """Get the invisible trace which shows the colour bar of a dimension.

    The scale is the colour map over the norm of the colours of the lines: a
    linear scale of the values, the logarithms of the values with the values
    as ticks, or a step per point with the labels as ticks.
    """
    import plotly.graph_objects as go

    colorbar: dict[str, Any] = {"title": {"text": styles.title, "side": "right"}}
    marker: dict[str, Any] = {"showscale": True, "colorbar": colorbar}
    if styles.ticks is not None:
        n = len(styles.ticks)
        scale: list[list[Any]] = []
        for k, color in enumerate(styles.colors):
            scale += [[k / n, color], [(k + 1) / n, color]]
        positions = tick_positions(n)
        marker.update(color=[-0.5, n - 0.5], cmin=-0.5, cmax=n - 0.5, colorscale=scale)
        colorbar.update(
            tickvals=positions, ticktext=[styles.ticks[k] for k in positions]
        )
    else:
        # the norm of the values spans them, from the smallest to the largest
        vmin, vmax = float(np.min(styles.values)), float(np.max(styles.values))
        marker["colorscale"] = _colorscale(styles.colormap)
        if isinstance(styles.norm, LogNorm):
            ticks = _log_ticks(vmin, vmax)
            marker["color"] = [math.log10(vmin), math.log10(vmax)]
            colorbar.update(
                tickvals=[math.log10(t) for t in ticks],
                ticktext=[format_value(t) for t in ticks],
            )
        else:
            marker["color"] = [vmin, vmax]
    return go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        showlegend=False,
        hoverinfo="skip",
        marker=marker,
    )


def _colorbar_width(trace: Any) -> float:
    """Estimate the width in pixels of a colour bar with its ticks and its title."""
    colorbar = trace.marker.colorbar
    if colorbar.ticktext is not None:
        texts = list(colorbar.ticktext)
    else:
        texts = [format_value(float(v)) for v in trace.marker.color]
    chars = max(len(text) for text in texts) + 1
    return 2 * _BAR_PAD + _BAR_THICKNESS + _BAR_TICKS + chars * _CHAR + _BAR_TITLE


def _values(data: Any, experiment: Any, unit: str | None) -> xr.DataArray | None:
    """Resolve a `Data` of a curve to its labelled array.

    Args:
        data: the `Data` of the curve, or `None`.
        experiment: the experiment the data is resolved against.
        unit: the unit the values are converted to.

    Returns:
        The values of the data, or `None`.
    """
    if data is None:
        return None
    return data.get_data(experiment=experiment, to_units=unit)


def _axis_options(axis: Axis | None) -> dict[str, Any]:
    """Get the plotly options of an axis of the figure model."""
    if axis is None:
        return {}
    options: dict[str, Any] = {
        "title": {"text": axis.name if axis.label_visible else None},
        "showgrid": axis.grid,
        "type": "log" if axis.scale == AxisScale.LOG10 else "linear",
        "showticklabels": axis.ticks_visible,
    }
    if axis.min is not None or axis.max is not None:
        low = np.log10(axis.min) if axis.min and options["type"] == "log" else axis.min
        high = np.log10(axis.max) if axis.max and options["type"] == "log" else axis.max
        if low is not None and high is not None:
            options["range"] = [low, high]
    if axis.reverse:
        options["autorange"] = "reversed"
    return options


def _line_options(style: Style | None) -> dict[str, Any]:
    """Get the plotly line of a style of the figure model."""
    if style is None or style.line is None:
        return {}
    line: dict[str, Any] = {"dash": DASH_BY_LINE.get(style.line.type, "solid")}
    if style.line.color is not None:
        line["color"] = style.line.color.color
    if style.line.thickness is not None:
        line["width"] = style.line.thickness
    return line


def _marker_options(style: Style | None) -> dict[str, Any]:
    """Get the plotly marker of a style of the figure model."""
    if style is None or style.marker is None:
        return {}
    marker: dict[str, Any] = {
        "symbol": SYMBOL_BY_MARKER.get(style.marker.type, "circle")
    }
    if style.marker.size is not None:
        marker["size"] = style.marker.size
    if style.marker.fill is not None:
        marker["color"] = style.marker.fill.color
    return marker


def _error_options(err: Any) -> dict[str, Any] | None:
    """Get the plotly error bars of an array of errors, `None` without."""
    if err is None:
        return None
    return {"type": "data", "array": err, "visible": True}


def _mode(style: Style | None) -> str:
    """Get whether a curve is drawn as a line, as markers or as both."""
    has_line = (
        style is not None
        and style.line is not None
        and (style.line.type is not LineType.NONE)
    )
    has_marker = (
        style is not None
        and style.marker is not None
        and (style.marker.type is not MarkerType.NONE)
    )
    if has_line and has_marker:
        return "lines+markers"
    if has_marker:
        return "markers"
    return "lines"


class PlotlyFigureSerializer:
    """Render a `Figure` of the figure model as a plotly figure."""

    @classmethod
    def to_figure(cls, experiment: Any, figure: Figure) -> Any:
        """Convert a figure of the model into a plotly figure.

        Args:
            experiment: the experiment whose results the curves read.
            figure: the figure to render.

        Returns:
            The `plotly.graph_objects.Figure`.
        """
        from plotly.subplots import make_subplots

        titles = [
            subplot.plot.name if subplot.plot.name else ""
            for subplot in figure.subplots
        ]
        specs = [
            [{"secondary_y": True} for _ in range(figure.num_cols)]
            for _ in range(figure.num_rows)
        ]
        fig = make_subplots(
            rows=figure.num_rows,
            cols=figure.num_cols,
            subplot_titles=titles,
            specs=specs,
        )

        subplot: SubPlot
        bars: list[tuple[Any, int, int, bool]] = []
        # the legend lists the entries in the order of the curves, whatever
        # the order the traces are drawn in
        ranks = itertools.count(1)
        for subplot in figure.subplots:
            if subplot.row is None or subplot.col is None:
                raise ValueError(f"SubPlot requires row and col: {subplot}")
            before = len(fig.data)
            cls._add_subplot(fig, experiment, subplot, ranks)
            bars.extend(
                (trace, subplot.row, subplot.col, bool(subplot.plot.yaxis_right))
                for trace in fig.data[before:]
                if trace.marker.showscale
            )

        fig.update_layout(
            title={"text": figure.name} if figure.name else None,
            width=int(figure.width * Figure.fig_dpi),
            height=int(figure.height * Figure.fig_dpi),
            template="plotly_white",
            hovermode="closest",
            legend={"traceorder": "normal"},
        )
        cls._merge_legend_entries(fig)
        if bars:
            cls._place_colorbars(fig, figure, bars)
        return fig

    @staticmethod
    def _merge_legend_entries(fig: Any) -> None:
        """Show an entry of the one legend of the figure once.

        Entries which read and look the same, e.g. the simulation of every
        panel, are one entry, the first in the legend, and switch the traces
        of all of them.
        """
        first: dict[tuple[Any, ...], str] = {}
        traces = sorted(fig.data, key=lambda t: t.legendrank or 0)
        for k, trace in enumerate(traces):
            if not trace.name or trace.showlegend is False:
                continue
            key = (
                trace.name,
                trace.mode,
                trace.fill,
                trace.fillcolor,
                trace.line.color,
                trace.line.dash,
                trace.marker.symbol,
                trace.marker.color if isinstance(trace.marker.color, str) else None,
            )
            if key not in first:
                if trace.legendgroup is None:
                    trace.legendgroup = f"_entry{k}"
                first[key] = trace.legendgroup
                continue
            trace.showlegend = False
            group, target = trace.legendgroup, first[key]
            if group is None:
                trace.legendgroup = target
            elif group != target:
                for other in fig.data:
                    if other.legendgroup == group:
                        other.legendgroup = target

    @classmethod
    def _place_colorbars(
        cls, fig: Any, figure: Figure, bars: list[tuple[Any, int, int, bool]]
    ) -> None:
        """Put the colour bars of every panel side by side right of the panel.

        Every column of panels gets the room its widest row of bars needs:
        the ticks and the title of a right y axis, then per bar a gap and the
        bar with its ticks and title. The figure is that much wider, so the
        panels keep their width, and the legend is right of every panel and
        bar. The right margin is the width of the legend, so that plotly does
        not shrink the plot area for it and the positions, which are fractions
        of the plot area, keep their pixels.
        """
        names = [t.name for t in fig.data if t.name and t.showlegend is not False]
        legend = max((len(name) for name in names), default=0) * _CHAR
        right_margin = _MARGIN
        if names:
            right_margin = max(_MARGIN, int(_GAP + legend + _LEGEND_SYMBOL))
        width = int(figure.width * Figure.fig_dpi)
        plot = width - _MARGIN - right_margin

        # the room of the bars of every panel and column, in pixels
        offsets: dict[tuple[int, int], float] = {}
        starts: list[tuple[Any, int, int, float, float]] = []
        for trace, row, col, has_right in bars:
            key = (row, col)
            if key not in offsets:
                offsets[key] = _RIGHT_AXIS if has_right else 0.0
            bar = _colorbar_width(trace)
            starts.append((trace, row, col, offsets[key] + _GAP, bar))
            offsets[key] += _GAP + bar
        rooms = [
            max((v for (_, c), v in offsets.items() if c == col), default=0.0)
            for col in range(1, figure.num_cols + 1)
        ]
        total = plot + sum(rooms)

        def shifted(col: int, fraction: float) -> float:
            """Get a position of a panel in a column as a fraction of the wider plot."""
            return (fraction * plot + sum(rooms[: col - 1])) / total

        for row in range(1, figure.num_rows + 1):
            for col in range(1, figure.num_cols + 1):
                axes = fig.get_subplot(row, col)
                if axes is not None:
                    a, b = axes.xaxis.domain
                    axes.xaxis.domain = [shifted(col, a), shifted(col, b)]
        right = max(
            fig.get_subplot(row, col).xaxis.domain[1]
            for row in range(1, figure.num_rows + 1)
            for col in range(1, figure.num_cols + 1)
            if fig.get_subplot(row, col) is not None
        )
        for trace, row, col, offset, bar in starts:
            axes = fig.get_subplot(row, col)
            low, high = axes.yaxis.domain
            x = axes.xaxis.domain[1] + offset / total
            right = max(right, x + bar / total)
            trace.marker.colorbar.update(
                x=x,
                xanchor="left",
                xpad=_BAR_PAD,
                y=(low + high) / 2,
                yanchor="middle",
                len=high - low,
                lenmode="fraction",
                thickness=_BAR_THICKNESS,
            )
        fig.update_layout(
            width=int(total + _MARGIN + right_margin),
            margin={"l": _MARGIN, "r": right_margin},
            legend={"x": right + _GAP / total, "xanchor": "left"},
        )

    @classmethod
    def _add_subplot(
        cls, fig: Any, experiment: Any, subplot: SubPlot, ranks: Iterator[int]
    ) -> None:
        """Add the curves, the areas and the bands of one panel to the figure.

        The areas of shaded areas and bands are drawn first, under every line,
        like matplotlib draws them; the legend keeps the order of the curves.
        A band without a colour takes the colour of the colour cycle at its
        position among the bands of the plot.
        """
        plot = subplot.plot
        row, col = subplot.row, subplot.col
        xaxis = plot.xaxis if plot.xaxis else Axis()
        yaxis = plot.yaxis if plot.yaxis else Axis()

        ranges: set[str] = set()
        drawn: list[tuple[Any, bool, bool]] = []
        for abstract_curve in sorted(
            [*plot.curves, *plot.areas, *plot.bands],
            key=lambda c: c.order if c.order is not None else 0,
        ):
            right = abstract_curve.yaxis_position == YAxisPosition.RIGHT
            yax = plot.yaxis_right if right and plot.yaxis_right else yaxis
            yunit = yax.unit if yax else None
            if isinstance(abstract_curve, Band):
                color = abstract_curve.color or cycle_color(
                    plot.bands.index(abstract_curve)
                )
                traces = cls._band_traces(
                    experiment, abstract_curve, xaxis.unit, yunit, color, ranges
                )
            else:
                area = isinstance(abstract_curve, ShadedArea)
                traces = [
                    (trace, area)
                    for trace in cls._traces(
                        experiment, abstract_curve, xaxis.unit, yunit
                    )
                ]
            for trace, under in traces:
                trace.legendrank = next(ranks)
                drawn.append((trace, under, right))
        for trace, _, right in sorted(drawn, key=lambda item: not item[1]):
            fig.add_trace(trace, row=row, col=col, secondary_y=right)

        fig.update_xaxes(**_axis_options(xaxis), row=row, col=col)
        fig.update_yaxes(**_axis_options(yaxis), row=row, col=col, secondary_y=False)
        if plot.yaxis_right:
            fig.update_yaxes(
                **_axis_options(plot.yaxis_right), row=row, col=col, secondary_y=True
            )

    @classmethod
    def _traces(
        cls, experiment: Any, abstract_curve: Any, xunit: str | None, yunit: str | None
    ) -> list[Any]:
        """Convert one curve or shaded area into plotly traces.

        A curve without `over` and a shaded area give one trace, a curve over
        scan points one per point; a band is `_band_traces`.
        """
        import plotly.graph_objects as go

        style = abstract_curve.style.resolve_style() if abstract_curve.style else None

        if isinstance(abstract_curve, ShadedArea):
            area: ShadedArea = abstract_curve
            x, yfrom, yto = line_values(
                area.sid or area.name or "",
                _values(area.x, experiment, xunit),
                _values(area.yfrom, experiment, yunit),
                _values(area.yto, experiment, yunit),
            )
            if x is None or yfrom is None or yto is None:
                return []
            color = None
            if style is not None and style.fill is not None:
                color = style.fill.color.color
            trace = go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([yto, yfrom[::-1]]),
                fill="toself",
                fillcolor=color,
                line={"width": 0},
                name=area.name or "",
                hoverinfo="skip",
                showlegend=area.name is not None,
            )
            return [trace]

        curve: Curve = abstract_curve
        if curve.type != CurveType.POINTS:
            logger.warning(
                "Only 'POINTS' curves are rendered by the plotly prototype, "
                "'%s' is skipped",
                curve.type,
            )
            return []

        if curve.over:
            return cls._point_traces(experiment, curve, style, xunit, yunit)

        x, y, yerr, xerr = line_values(
            curve.sid or curve.name or "",
            _values(curve.x, experiment, xunit),
            _values(curve.y, experiment, yunit),
            _values(curve.yerr, experiment, yunit),
            _values(curve.xerr, experiment, xunit),
        )
        if x is None or y is None:
            return []

        return [
            go.Scatter(
                x=x,
                y=y,
                name=curve.name or "",
                mode=_mode(style),
                line=_line_options(style),
                marker=_marker_options(style),
                error_x=_error_options(xerr),
                error_y=_error_options(yerr),
                showlegend=bool(curve.name),
            )
        ]

    @classmethod
    def _point_traces(
        cls,
        experiment: Any,
        curve: Curve,
        style: Style | None,
        xunit: str | None,
        yunit: str | None,
    ) -> list[Any]:
        """Convert a curve over scan points into one trace per point.

        The colours, dashes and names are those of the matplotlib figure. From
        `COLORBAR_FROM` points the lines have no legend entry, a colour bar
        shows the first dimension and a named curve keeps one entry of its
        name in the colour of the middle of the colour map.
        """
        import plotly.graph_objects as go

        x = curve.x.get_data(experiment=experiment, to_units=xunit)
        y = curve.y.get_data(experiment=experiment, to_units=yunit)
        xerr = (
            curve.xerr.get_data(experiment=experiment, to_units=xunit)
            if curve.xerr is not None
            else None
        )
        yerr = (
            curve.yerr.get_data(experiment=experiment, to_units=yunit)
            if curve.yerr is not None
            else None
        )
        sid = curve.sid or curve.name or ""
        lines: list[Line] = curve_lines(sid, curve.over, x, y, xerr, yerr)
        task = curve.y.task_id or curve.x.task_id
        dimensions = [
            experiment.scan_dimension(task, d) if task else None for d in curve.over
        ]
        units = experiment.model_units(task) if task else {}
        color = (
            style.line.color.color
            if style and style.line and style.line.color
            else None
        )
        styles = point_styles(lines, curve.over, color, dimensions, units)
        prefix = f"{curve.name}, " if curve.name else ""
        traces = []
        for line in lines:
            point = [styles.labels[k][i] for k, i in enumerate(line.index)]
            line_style = {
                **_line_options(style),
                "color": styles.colors[line.index[0]],
            }
            if styles.linestyles is not None:
                line_style["dash"] = DASH_BY_LINESTYLE[styles.linestyles[line.index[1]]]
            marker = {**_marker_options(style), "color": styles.colors[line.index[0]]}
            traces.append(
                go.Scatter(
                    x=line.x,
                    y=line.y,
                    name=prefix + ", ".join(point),
                    legendgroup=sid,
                    showlegend=not styles.colorbar and styles.linestyles is None,
                    mode=_mode(style),
                    line=line_style,
                    marker=marker,
                    error_x=_error_options(line.xerr),
                    error_y=_error_options(line.yerr),
                )
            )
        if styles.colorbar and curve.name:
            traces.append(
                _legend_trace(curve.name, styles.middle, "solid", style, group=sid)
            )
        if styles.linestyles is not None:
            # two dimensions: the lines have no entries, proxies name the colours
            # of the first dimension and the dashes of the second, as matplotlib
            if not styles.colorbar:
                for c, text in zip(styles.colors, styles.labels[0], strict=True):
                    traces.append(_legend_trace(f"{prefix}{text}", c, "solid", style))
            for ls, text in zip(styles.linestyles, styles.labels[1], strict=True):
                traces.append(
                    _legend_trace(text, to_hex("0.4"), DASH_BY_LINESTYLE[ls], style)
                )
        if styles.colorbar:
            traces.append(_colorbar_trace(styles))
        return traces

    @classmethod
    def _band_traces(
        cls,
        experiment: Any,
        band: Band,
        xunit: str | None,
        yunit: str | None,
        color: str,
        ranges: set[str],
    ) -> list[tuple[Any, bool]]:
        """Convert a band into traces: per point the quantiles and the median.

        The legend has one entry per point (the median, else the upper
        quantile), or one of the name of a band with a colour bar, and a grey
        entry of the quantile range once per panel and range, like matplotlib.

        Args:
            experiment: the experiment the data is resolved against.
            band: the band.
            xunit: the unit of the x axis.
            yunit: the unit of the y axis of the band.
            color: the colour of a band without `over`.
            ranges: the quantile ranges of the panel with a legend entry, which
                this one is added to.

        Returns:
            The traces, each with whether it belongs to the area, which is
            drawn under the lines.
        """
        import plotly.graph_objects as go

        x = band.x.get_data(experiment=experiment, to_units=xunit)
        y = band.y.get_data(experiment=experiment, to_units=yunit)
        sid = band.sid or band.name or ""
        bands: list[BandLine] = band_lines(
            sid, band.over, band.across, band.quantiles, x, y
        )
        low, high = (f"{100 * q:g}" for q in band.quantiles)
        styles = None
        if band.over:
            task = band.y.task_id
            dimension = experiment.scan_dimension(task, band.over[0]) if task else None
            units = experiment.model_units(task) if task else {}
            styles = point_styles(bands, band.over, band.color, [dimension], units)
        show = styles is None or not styles.colorbar
        traces: list[tuple[Any, bool]] = []
        for b in bands:
            c = to_hex(styles.colors[b.index[0]] if styles else color)
            if styles:
                name = f"{band.name}, {styles.labels[0][b.index[0]]}"
            else:
                name = f"{band.name} median" if band.median else band.name
            edge = {
                "color": _rgba(c, min(1.0, 2 * band.alpha)),
                "width": _BAND_EDGE_WIDTH,
            }
            traces.append(
                (
                    go.Scatter(
                        x=b.x,
                        y=b.low,
                        mode="lines",
                        line=edge,
                        legendgroup=sid,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    True,
                )
            )
            traces.append(
                (
                    go.Scatter(
                        x=b.x,
                        y=b.high,
                        mode="lines",
                        line=edge,
                        fill="tonexty",
                        fillcolor=_rgba(c, band.alpha),
                        name=name,
                        legendgroup=sid,
                        showlegend=show and not band.median,
                        hoverinfo="skip",
                    ),
                    True,
                )
            )
            if band.median:
                traces.append(
                    (
                        go.Scatter(
                            x=b.x,
                            y=b.median,
                            mode="lines",
                            line={"color": c, "width": 2},
                            name=name,
                            legendgroup=sid,
                            showlegend=show,
                        ),
                        False,
                    )
                )
        if styles is not None and styles.colorbar and band.name:
            traces.append(
                (
                    _legend_trace(band.name, styles.middle, "solid", None, group=sid),
                    False,
                )
            )
        quantiles = f"{low}-{high} %"
        if quantiles not in ranges:
            ranges.add(quantiles)
            traces.append(
                (
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line={"color": _rgba("0.5", max(band.alpha, 0.3)), "width": 8},
                        name=quantiles,
                        legendgroup=sid,
                        showlegend=True,
                        hoverinfo="skip",
                    ),
                    False,
                )
            )
        if styles is not None and styles.colorbar:
            traces.append((_colorbar_trace(styles), False))
        return traces


def figures_to_html(
    experiment: Any,
    output_path: Path,
    figures: dict[str, Figure] | None = None,
) -> dict[str, Path]:
    """Write the figures of an experiment as interactive HTML pages.

    The javascript of plotly is written once next to the pages, so the pages
    need no network.

    Args:
        experiment: the experiment whose figures are written.
        output_path: directory of the pages, created if it is missing.
        figures: the figures to write, those of the experiment by default.

    Returns:
        The path of the page of every figure, by its key.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    figures = figures if figures is not None else experiment._figures

    paths: dict[str, Path] = {}
    for key, figure in figures.items():
        plotly_figure = PlotlyFigureSerializer.to_figure(experiment, figure)
        path = output_path / f"{experiment.sid}_{key}.html"
        plotly_figure.write_html(path, include_plotlyjs="directory")
        paths[key] = path
    return paths
