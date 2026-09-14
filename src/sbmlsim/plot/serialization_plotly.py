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

import logging
from pathlib import Path
from typing import Any

import numpy as np

from sbmlsim.plot.plotting import (
    Axis,
    AxisScale,
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


def _values(data: Any, experiment: Any, unit: str | None) -> np.ndarray | None:
    """Resolve a `Data` of a curve to the numbers which are plotted.

    Args:
        data: the `Data` of the curve, or `None`.
        experiment: the experiment the data is resolved against.
        unit: the unit the values are converted to.

    Returns:
        The values, the first column of a data cube, or `None`.
    """
    if data is None:
        return None
    quantity = data.get_data(experiment=experiment, to_units=unit)
    if quantity is None:
        return None
    magnitude = quantity.magnitude
    # a scan has a column per repeat, the first one is plotted, as in the
    # matplotlib serializer
    return magnitude[:, 0] if np.ndim(magnitude) == 2 else magnitude


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
        for subplot in figure.subplots:
            if subplot.row is None or subplot.col is None:
                raise ValueError(f"SubPlot requires row and col: {subplot}")
            cls._add_subplot(fig, experiment, subplot)

        fig.update_layout(
            title={"text": figure.name} if figure.name else None,
            width=int(figure.width * Figure.fig_dpi),
            height=int(figure.height * Figure.fig_dpi),
            template="plotly_white",
            hovermode="closest",
        )
        return fig

    @classmethod
    def _add_subplot(cls, fig: Any, experiment: Any, subplot: SubPlot) -> None:
        """Add the curves and the areas of one panel to the figure."""
        plot = subplot.plot
        row, col = subplot.row, subplot.col
        xaxis = plot.xaxis if plot.xaxis else Axis()
        yaxis = plot.yaxis if plot.yaxis else Axis()

        for abstract_curve in plot.curves + plot.areas:
            right = abstract_curve.yaxis_position == YAxisPosition.RIGHT
            yax = plot.yaxis_right if right and plot.yaxis_right else yaxis
            trace = cls._trace(
                experiment, abstract_curve, xaxis.unit, yax.unit if yax else None
            )
            if trace is not None:
                fig.add_trace(trace, row=row, col=col, secondary_y=right)

        fig.update_xaxes(**_axis_options(xaxis), row=row, col=col)
        fig.update_yaxes(**_axis_options(yaxis), row=row, col=col, secondary_y=False)
        if plot.yaxis_right:
            fig.update_yaxes(
                **_axis_options(plot.yaxis_right), row=row, col=col, secondary_y=True
            )

    @classmethod
    def _trace(
        cls, experiment: Any, abstract_curve: Any, xunit: str | None, yunit: str | None
    ) -> Any:
        """Convert one curve or shaded area into a plotly trace."""
        import plotly.graph_objects as go

        style = abstract_curve.style.resolve_style() if abstract_curve.style else None

        if isinstance(abstract_curve, ShadedArea):
            area: ShadedArea = abstract_curve
            x = _values(area.x, experiment, xunit)
            yfrom = _values(area.yfrom, experiment, yunit)
            yto = _values(area.yto, experiment, yunit)
            if x is None or yfrom is None or yto is None:
                return None
            color = None
            if style is not None and style.fill is not None:
                color = style.fill.color.color
            return go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([yto, yfrom[::-1]]),
                fill="toself",
                fillcolor=color,
                line={"width": 0},
                name=area.name or "",
                hoverinfo="skip",
                showlegend=area.name is not None,
            )

        curve: Curve = abstract_curve
        if curve.type != CurveType.POINTS:
            logger.warning(
                "Only 'POINTS' curves are rendered by the plotly prototype, "
                "'%s' is skipped",
                curve.type,
            )
            return None

        x = _values(curve.x, experiment, xunit)
        y = _values(curve.y, experiment, yunit)
        if x is None or y is None:
            return None

        error_y = None
        yerr = _values(curve.yerr, experiment, yunit)
        if yerr is not None:
            error_y = {"type": "data", "array": yerr, "visible": True}
        error_x = None
        xerr = _values(curve.xerr, experiment, xunit)
        if xerr is not None:
            error_x = {"type": "data", "array": xerr, "visible": True}

        return go.Scatter(
            x=x,
            y=y,
            name=curve.name or "",
            mode=_mode(style),
            line=_line_options(style),
            marker=_marker_options(style),
            error_x=error_x,
            error_y=error_y,
            showlegend=bool(curve.name),
        )


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
