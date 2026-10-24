"""Serialization of Figure object to matplotlib."""

from __future__ import annotations

import functools
import logging
import textwrap
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes as AxesMPL
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.cm import ScalarMappable
from matplotlib.collections import Collection, LineCollection, PolyCollection
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm, to_hex, to_rgba_array
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.figure import Figure as FigureMPL
from matplotlib.gridspec import GridSpec
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.transforms import Bbox

from sbmlsim.plot import Axis, Curve, Figure, Plot, SubPlot
from sbmlsim.plot.padding import line_values
from sbmlsim.plot.plotting import (
    AbstractCurve,
    AxisScale,
    Band,
    CurveType,
    FigureSettings,
    LineType,
    ShadedArea,
    Style,
    YAxisPosition,
    log_limits,
)
from sbmlsim.plot.points import (
    BandLine,
    Line,
    PointStyle,
    band_label,
    band_lines,
    curve_lines,
    cycle_color,
    format_value,
    point_styles,
    tick_positions,
)

logger = logging.getLogger(__name__)

#: the gap in inches between a panel (or the ticks and the title of its right y
#: axis), each of its colour bars and an outside legend
COLORBAR_GAP = 0.12

#: the width of a colour bar in inches
COLORBAR_WIDTH = 0.15

#: the least room in inches between the ticks and labels of the panels and the
#: edges of the figure (the padding of `bbox_inches="tight"`)
EDGE_PAD = 0.1


def _measuring_renderer(fig: FigureMPL) -> RendererAgg:
    """Get a renderer which measures the text of a figure without drawing it.

    A text or an axes measured without a renderer draws the whole figure into
    a renderer of the size of the figure and keeps it: the margins, the
    wrapping of the labels and the room right of the panels drew a figure
    dozens of times, half the time of the simulation experiments of a model,
    and the texts kept renderers of 10 GB until the figures were saved. The
    extents of a text depend on the resolution of the renderer, not on its
    size, so one renderer of a pixel at the resolution of the figure measures
    everything. It is one renderer per resolution: matplotlib caches the
    metrics of a text per renderer, so the texts measured when a figure is
    built are not measured again when its box is, see `saved_box`.

    Args:
        fig: the figure.

    Returns:
        The renderer.
    """
    return _renderer_at(fig.dpi)


@functools.cache
def _renderer_at(dpi: float) -> RendererAgg:
    """Get the renderer of a pixel at a resolution, see `_measuring_renderer`."""
    return RendererAgg(1, 1, dpi)


#: the figures `MatplotlibFigureSerializer.to_figure` built, see `saved_box`
_BUILT: weakref.WeakSet[FigureMPL] = weakref.WeakSet()


def saved_box(fig: FigureMPL) -> Bbox | Literal["tight"]:
    """Get the box of a figure to save, `bbox_inches` of `savefig`.

    `bbox_inches="tight"` draws the whole figure to find the ticks, labels,
    legends and titles which reach beyond it, and once more to write it, 15 %
    of the time of the simulation experiments of a model. A figure which the
    serializer built has its texts measured already and needs no drawing for
    its box: the tight box of its artists, measured with `_measuring_renderer`
    and padded by `savefig.pad_inches`, is the box of `"tight"` up to the
    metrics of the fonts of the format (a fraction of a point). Any other
    figure, and one with a layout engine, which lays it out when it is drawn,
    is saved `"tight"`.

    Args:
        fig: the figure.

    Returns:
        The box in inches, or `"tight"`.
    """
    pad = rcParams["savefig.pad_inches"]
    if (
        fig not in _BUILT
        or fig.get_layout_engine() is not None
        or not isinstance(pad, (int, float))
    ):
        return "tight"
    return fig.get_tightbbox(_measuring_renderer(fig)).padded(float(pad))


@dataclass
class _Panel:
    """A drawn panel, whose colour bars and legend are placed after all panels.

    Attributes:
        plot: the plot of the panel.
        axes: its axes, the left one first and the right one if it has one.
        first: the first column of the grid the panel spans, from 0.
        last: the last column of the grid the panel spans.
        top: the first row of the grid the panel spans, from 0.
        bottom: the last row of the grid the panel spans.
        pending: the styles of its colour bars, in the order of its curves.
    """

    plot: Plot
    axes: list[AxesMPL]
    first: int
    last: int
    top: int
    bottom: int
    pending: list[PointStyle] = field(default_factory=list)


@dataclass
class _Slot:
    """A legend outside, placed right of its panel or right of the grid.

    Attributes:
        legend: the legend.
        panel: the panel the legend is right of, after its right y axis and
            its colour bars; `None` for the legend of a row, which is right of
            the grid, after the right y axes and the colour bars of every
            panel of the last column.
        top: the top of the legend in figure coordinates before the figure
            grows, the top of its panel or of its row.
        height: the height the legend has before it overflows, in figure
            coordinates, of its panel or of its row.
        top_row: the row the legend hangs from, which moves down by the
            overflow of the rows above it.
        bottom_row: the row below which the overflow of the legend goes.
    """

    legend: Legend
    panel: _Panel | None
    top: float
    height: float
    top_row: int
    bottom_row: int


def _entries(panel: _Panel) -> tuple[list[Any], list[str]]:
    """Get the legend entries of a panel, of its left axes and then of its right ones."""
    handles: list[Any] = []
    labels: list[str] = []
    for ax in panel.axes:
        more_handles, more_labels = ax.get_legend_handles_labels()
        handles += more_handles
        labels += more_labels
    return handles, labels


def _colors(values: Any) -> tuple[tuple[float, ...], ...]:
    """Get colours as a tuple of RGBA tuples, which compares and hashes."""
    return tuple(tuple(float(v) for v in rgba) for rgba in to_rgba_array(values))


def _entry_key(handle: Any, label: str) -> tuple[Any, ...]:
    """Get what an entry of a legend reads and looks like, see `_row_legends`.

    Two entries with the same key are one entry of the legend of a row: the
    label and the style of the handle (colours, line, marker, hatch, error
    bars), as the legend of plotly merges its entries. A handle of an unknown
    kind is never merged.
    """
    if isinstance(handle, ErrorbarContainer):
        line, _, bars = handle.lines
        return (
            label,
            "errorbar",
            _entry_key(line, label) if line is not None else None,
            handle.has_xerr,
            handle.has_yerr,
            tuple(_colors(bar.get_color()) for bar in bars),
        )
    if isinstance(handle, BarContainer):
        return (
            label,
            "bar",
            *(_entry_key(handle.patches[0], label)[1:] if handle.patches else ()),
        )
    if isinstance(handle, Line2D):
        return (
            label,
            "line",
            _colors(handle.get_color()),
            handle.get_linestyle(),
            handle.get_linewidth(),
            handle.get_drawstyle(),
            str(handle.get_marker()),
            handle.get_markersize(),
            _colors(handle.get_markerfacecolor()),
            _colors(handle.get_markeredgecolor()),
            handle.get_alpha(),
        )
    if isinstance(handle, Patch):
        return (
            label,
            "patch",
            _colors(handle.get_facecolor()),
            _colors(handle.get_edgecolor()),
            handle.get_hatch(),
            handle.get_linewidth(),
            handle.get_linestyle(),
            handle.get_alpha(),
        )
    if isinstance(handle, Collection):
        return (
            label,
            type(handle).__name__,
            _colors(handle.get_facecolor()),
            _colors(handle.get_edgecolor()),
            handle.get_hatch(),
            tuple(float(w) for w in np.atleast_1d(handle.get_linewidth())),
            # the dashes of the edges, a list of (offset, dashes) of matplotlib
            repr(handle.get_linestyle()),
            handle.get_alpha(),
        )
    return (label, id(handle))


def limit_log_axis(
    ax: AxesMPL,
    axis: str = "y",
    low: float | None = None,
    high: float | None = None,
) -> None:
    """Limit a log axis of matplotlib to `LOG_DECADES` below its largest value.

    The limits of the data drawn so far, see `sbmlsim.plot.plotting.log_limits`;
    for a figure of a model drawn with matplotlib itself, after its data and
    `set_yscale("log")`.

    Args:
        ax: the axes.
        axis: `"x"` or `"y"`.
        low: the lower limit set on the axis, `None` without.
        high: the upper limit set on the axis, `None` without.
    """
    data = ax.dataLim
    smallest, largest = (
        (data.minposx, data.x1) if axis == "x" else (data.minposy, data.y1)
    )
    limits = log_limits(smallest, largest, low, high)
    if limits is None:
        return
    if axis == "x":
        ax.set_xlim(*limits)
    else:
        ax.set_ylim(*limits)


def limit_log_axes(figure: FigureMPL) -> None:
    """Limit every automatic log axis of a figure, see `limit_log_axis`.

    The custom figures of an experiment (`figures_mpl`) are drawn by the
    model; a log axis whose limits it set itself (autoscale off) is kept.
    """
    for ax in figure.axes:
        if ax.get_xscale() == "log" and ax.get_autoscalex_on():
            limit_log_axis(ax, "x")
        if ax.get_yscale() == "log" and ax.get_autoscaley_on():
            limit_log_axis(ax, "y")


class MatplotlibFigureSerializer:
    """Serializer for figures to matplotlib."""

    @classmethod
    def _get_scale(cls, axis: Axis) -> str:
        """Get string representation of the scale."""
        if axis.scale == AxisScale.LINEAR:
            return "linear"
        if axis.scale == AxisScale.LOG10:
            return "log"
        raise ValueError(f"Unsupported axis scale: '{axis.scale}'")

    @classmethod
    def _curve_kwargs(cls, curve: Curve) -> dict[str, Any]:
        """Get the matplotlib keyword arguments of the style of a curve."""
        if not curve.style:
            return {}
        style: Style = curve.style.resolve_style()
        if curve.type == CurveType.POINTS:
            return style.to_mpl_points_kwargs()
        return style.to_mpl_bar_kwargs()

    @classmethod
    def _style_color(cls, style: Any) -> str | None:
        """Get the colour of the line of a style, `None` if it sets none."""
        if not style:
            return None
        resolved: Style = style.resolve_style()
        if resolved.line is not None and resolved.line.color is not None:
            return resolved.line.color.color
        return None

    @classmethod
    def _draw_line(
        cls, ax: AxesMPL, line: Line, kwargs: dict[str, Any], label: str
    ) -> None:
        """Draw a line, with error bars if it has errors."""
        kwargs = dict(kwargs)
        if line.xerr is None and line.yerr is None:
            # `errorbar` builds the containers of the bars whether or not there
            # are any, and is twice the cost of `plot` for the same line
            kwargs.pop("capsize", None)
            ax.plot(line.x, line.y, label=label, **kwargs)
        else:
            ax.errorbar(
                x=line.x,
                y=line.y,
                xerr=line.xerr,
                yerr=line.yerr,
                label=label,
                **kwargs,
            )

    @classmethod
    def _draw_curve(
        cls,
        ax: AxesMPL,
        curve: Curve,
        line: Line,
        kwargs: dict[str, Any],
        label: str,
        stacks: dict[str, Any],
    ) -> None:
        """Draw the one line of a curve of its type."""
        x_data, y_data, xerr_data, yerr_data = line.x, line.y, line.xerr, line.yerr
        if curve.type == CurveType.POINTS:
            cls._draw_line(ax, line, kwargs, label)

        elif curve.type == CurveType.BAR:
            ax.bar(
                x=x_data,
                height=y_data,
                xerr=xerr_data,
                yerr=yerr_data,
                label=label,
                **kwargs,
            )

        elif curve.type == CurveType.HORIZONTALBAR:
            ax.barh(
                y=x_data,
                width=y_data,
                xerr=yerr_data,
                yerr=xerr_data,
                label=label,
                **kwargs,
            )

        elif curve.type == CurveType.BARSTACKED:
            if "barstack_x" not in stacks:
                stacks["barstack_x"] = x_data
                stacks["barstack_y"] = np.zeros_like(y_data)

            if not np.all(np.isclose(stacks["barstack_x"], x_data)):
                raise ValueError("x data must match for stacked bars.")
            ax.bar(
                x=x_data,
                height=y_data,
                bottom=stacks["barstack_y"],
                xerr=xerr_data,
                yerr=yerr_data,
                label=label,
                **kwargs,
            )
            stacks["barstack_y"] = stacks["barstack_y"] + y_data

        elif curve.type == CurveType.HORIZONTALBARSTACKED:
            if "barhstack_x" not in stacks:
                stacks["barhstack_x"] = x_data
                stacks["barhstack_y"] = np.zeros_like(y_data)

            if not np.all(np.isclose(stacks["barhstack_x"], x_data)):
                raise ValueError("x data must match for stacked bars.")
            ax.barh(
                y=x_data,
                width=y_data,
                left=stacks["barhstack_y"],
                xerr=yerr_data,
                yerr=xerr_data,
                label=label,
                **kwargs,
            )
            stacks["barhstack_y"] = stacks["barhstack_y"] + y_data

    @classmethod
    def _proxy(
        cls,
        ax: AxesMPL,
        kwargs: dict[str, Any],
        color: str,
        label: str,
        linestyle: str | None = None,
    ) -> None:
        """Add a legend entry without data in the style of the lines of a curve."""
        kw = {k: v for k, v in kwargs.items() if k != "capsize"}
        kw["color"] = color
        if "markerfacecolor" in kw:
            kw["markerfacecolor"] = color
        if linestyle is not None:
            kw["linestyle"] = linestyle
        ax.plot([], [], label=label, **kw)

    @classmethod
    def _draw_points(
        cls,
        ax: AxesMPL,
        experiment: Any,
        curve: Curve,
        lines: list[Line],
        kwargs: dict[str, Any],
    ) -> PointStyle | None:
        """Draw the lines of a curve over scan points with their legend entries.

        A curve with a colour bar keeps one entry of its name, in the colour of
        the middle of the colour map.

        Returns:
            The style of the points if they get a colour bar, else `None`.
        """
        task = curve.y.task_id or curve.x.task_id
        dimensions = [
            experiment.scan_dimension(task, d) if task else None for d in curve.over
        ]
        units = experiment.model_units(task) if task else {}
        color = cls._style_color(curve.style)
        styles = point_styles(lines, curve.over, color, dimensions, units)
        prefix = f"{curve.name}, " if curve.name else ""
        labelled = len(curve.over) == 1 and not styles.colorbar
        for line in lines:
            kw = dict(kwargs)
            kw["color"] = styles.colors[line.index[0]]
            if "markerfacecolor" in kw:
                kw["markerfacecolor"] = kw["color"]
            if styles.linestyles is not None:
                kw["linestyle"] = styles.linestyles[line.index[1]]
            label = (
                f"{prefix}{styles.labels[0][line.index[0]]}"
                if labelled
                else "_nolegend_"
            )
            cls._draw_line(ax, line, kw, label)
        if styles.colorbar and curve.name:
            cls._proxy(ax, kwargs, styles.middle, curve.name)
        if styles.linestyles is not None:
            if not styles.colorbar:
                for c, text in zip(styles.colors, styles.labels[0], strict=True):
                    cls._proxy(ax, kwargs, c, f"{prefix}{text}")
            for ls, text in zip(styles.linestyles, styles.labels[1], strict=True):
                cls._proxy(ax, kwargs, to_hex("0.4"), text, linestyle=ls)
        return styles if styles.colorbar else None

    @classmethod
    def _colorbar(
        cls,
        fig: FigureMPL,
        cax: AxesMPL,
        styles: PointStyle,
        settings: FigureSettings,
    ) -> Colorbar:
        """Draw the colour bar of the first dimension of a curve over scan points.

        It maps the values through the norm of the colours of the lines, with
        the values as ticks of a logarithmic norm and the labels of the points
        as ticks of a discrete one, and is labelled like the axes.
        """
        mappable = ScalarMappable(norm=styles.norm, cmap=styles.colormap)
        colorbar = fig.colorbar(mappable, cax=cax)
        if styles.ticks is not None:
            positions = tick_positions(len(styles.ticks))
            colorbar.set_ticks(positions, labels=[styles.ticks[k] for k in positions])
        elif isinstance(styles.norm, LogNorm):
            colorbar.ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: format_value(value))
            )
            colorbar.ax.yaxis.set_minor_formatter(NullFormatter())
        colorbar.set_label(
            styles.title,
            fontsize=settings.axes_labelsize,
            fontweight=settings.axes_labelweight,
        )
        colorbar.ax.tick_params(labelsize=settings.ytick_labelsize)
        return colorbar

    @classmethod
    def _place_right(
        cls,
        fig: FigureMPL,
        grid: tuple[int, int],
        panels: list[_Panel],
        settings: FigureSettings,
        title: Text | None = None,
        slots: list[_Slot] | None = None,
        grid_right: float = 0.0,
        renderer: RendererBase | None = None,
    ) -> None:
        """Place the colour bars and the outside legend of every panel right of it.

        A panel needs room for the ticks and the title of a right y axis and,
        per bar, a gap, the bar and its ticks and label, which are measured on
        the drawn bar, and for a legend outside a gap and the legend, measured
        on the drawn legend. Every column of the grid gets the room its widest
        panel needs and the figure is that much wider, so the panels keep
        their size and the columns their alignment: a legend outside widens
        the gap right of its column and no other gap, e.g. one legend per row
        on the last plot of every row widens the figure by its legend alone.
        A legend outside hangs from the top of its panel; the row of a panel
        whose legend is taller than the panel gets the room by which it
        overflows the panel below it, and the figure is that much taller, so
        nothing overlaps. The title of a wider figure is centered over its
        panels and keeps its distance to the top of a taller one. The legends
        of the rows are right of the grid, from the top of their row, all at
        the same distance right of the right y axes and the colour bars of the
        panels of the last column, which gets the room of the widest of them
        (#327).

        Args:
            fig: the figure.
            grid: the number of rows and of columns of the grid.
            panels: the drawn panels.
            settings: the settings of the figure.
            title: the title of the figure, if it has one.
            slots: the legends outside, see `_Slot`.
            grid_right: the right edge of the last column of the grid in
                figure coordinates, which the legends of the rows are right
                of when no panel is in the last column.
            renderer: the renderer which measures the bars and the legends,
                see `_measuring_renderer`; one of its own by default.
        """
        renderer = renderer or _measuring_renderer(fig)
        num_rows, num_cols = grid
        last = num_cols - 1
        slots = slots or []
        # the legends of the rows, right of the grid
        grid_legends = [slot for slot in slots if slot.panel is None]
        width, height, dpi = fig.get_figwidth(), fig.get_figheight(), fig.dpi
        boxes = [panel.axes[0].get_position() for panel in panels]
        rooms: list[list[float]] = []
        bars: list[list[Colorbar]] = []
        legends: list[float] = []
        for panel, box in zip(panels, boxes, strict=True):
            panel_rooms: list[float] = []
            panel_bars: list[Colorbar] = []
            widths = [
                slot.legend.get_window_extent(renderer).width / dpi
                for slot in slots
                if slot.panel is panel
            ]
            legends.append(max(widths, default=0.0))
            if panel.pending or widths or (grid_legends and panel.last == last):
                # the ticks and the title of a right y axis, in inches
                right = 0.0
                for ax in panel.axes[1:]:
                    tight = ax.yaxis.get_tightbbox(renderer)
                    if tight is not None:
                        right = max(right, tight.x1 / dpi - box.x1 * width)
                panel_rooms.append(right)
                for styles in panel.pending:
                    cax = fig.add_axes(
                        (box.x1, box.y0, COLORBAR_WIDTH / width, box.height)
                    )
                    panel_bars.append(cls._colorbar(fig, cax, styles, settings))
                    tight = cax.get_tightbbox(renderer)
                    end = tight.x1 / dpi if tight is not None else 0.0
                    panel_rooms.append(max(0.0, end - box.x1 * width - COLORBAR_WIDTH))
            rooms.append(panel_rooms)
            bars.append(panel_bars)

        def need(panel_rooms: list[float], legend: float) -> float:
            """Get the room of the bars and the legend of a panel right of it, in inches."""
            if not panel_rooms:
                return 0.0
            right, *labels = panel_rooms
            bars = sum(COLORBAR_GAP + COLORBAR_WIDTH + r for r in labels)
            return right + bars + (COLORBAR_GAP + legend if legend else 0.0)

        columns = [0.0] * num_cols
        for panel, panel_rooms, legend in zip(panels, rooms, legends, strict=True):
            columns[panel.last] = max(columns[panel.last], need(panel_rooms, legend))
        if grid_legends:
            # the legends of the rows after the right y axes and the colour
            # bars of every panel of the last column
            columns[last] = max(
                columns[last],
                max(
                    (
                        need(panel_rooms, 0.0)
                        for panel, panel_rooms in zip(panels, rooms, strict=True)
                        if panel.last == last
                    ),
                    default=0.0,
                )
                + COLORBAR_GAP
                + max(
                    slot.legend.get_window_extent(renderer).width / dpi
                    for slot in grid_legends
                ),
            )
        # the overflow of a legend below its panel or its row, in inches
        rows = [0.0] * num_rows
        for slot in slots:
            overflow = (
                slot.legend.get_window_extent(renderer).height / dpi
                - slot.height * height
            )
            rows[slot.bottom_row] = max(rows[slot.bottom_row], overflow)
        if not any(columns) and not any(rows):
            return
        wider, taller = width + sum(columns), height + sum(rows)
        fig.set_size_inches(wider, taller)
        # the right edge of the colour bars of a panel, in inches
        edges: dict[int, float] = {}
        # the right edge of the grid, the last column and its colour bars
        grid_edge = grid_right * width + sum(columns[:last])
        for panel, box, panel_rooms, panel_bars in zip(
            panels, boxes, rooms, bars, strict=True
        ):
            x0 = box.x0 * width + sum(columns[: panel.first])
            x1 = box.x1 * width + sum(columns[: panel.last])
            # the distance of the top to the top of the figure stays, the rows
            # above it may have grown, and the height of the panel grows by the
            # room which the legends add below the rows it spans
            top = (1.0 - box.y1) * height + sum(rows[: panel.top])
            y1 = 1.0 - top / taller
            y0 = (
                y1
                - (box.height * height + sum(rows[panel.top : panel.bottom])) / taller
            )
            for ax in panel.axes:
                ax.set_position((x0 / wider, y0, (x1 - x0) / wider, y1 - y0))
            if not panel_rooms:
                continue
            x = x1 + panel_rooms[0]
            for bar, room in zip(panel_bars, panel_rooms[1:], strict=True):
                x += COLORBAR_GAP
                bar.ax.set_position((x / wider, y0, COLORBAR_WIDTH / wider, y1 - y0))
                x += COLORBAR_WIDTH + room
            edges[id(panel)] = x
            if panel.last == last:
                grid_edge = max(grid_edge, x)
        for slot in slots:
            x = grid_edge if slot.panel is None else edges[id(slot.panel)]
            top = (1.0 - slot.top) * height + sum(rows[: slot.top_row])
            slot.legend.set_bbox_to_anchor(
                ((x + COLORBAR_GAP) / wider, 1.0 - top / taller),
                transform=fig.transFigure,
            )
        if title is not None:
            left = min(ax.get_position().x0 for panel in panels for ax in panel.axes)
            right = max(ax.get_position().x1 for panel in panels for ax in panel.axes)
            title.set_x(0.5 * (left + right))
            title.set_y(1.0 - (1.0 - title.get_position()[1]) * height / taller)

    @classmethod
    def _legend(
        cls, fig: FigureMPL, settings: FigureSettings, panel: _Panel
    ) -> Legend | None:
        """Draw the legend of a panel, return it if it is outside.

        Outside, one legend of both axes is right of the panel, right of the
        ticks and the title of its right y axis and of its last colour bar,
        where `_place_right` puts it. Inside, a panel with a right y axis gets
        one legend of both axes as well, on the right axes, which are drawn
        last, placed against the data of both: the lines and areas of the left
        axes are on the right axes as invisible copies, which only the
        placement sees.
        """
        ax1, *others = panel.axes
        handles, labels = _entries(panel)
        if not handles:
            return None
        if settings.legend_position == "outside":
            # anchored right of the axes until `_place_right` places it, the
            # gap to the panel is `COLORBAR_GAP` and its top the top of the
            # panel (no pad of matplotlib)
            return ax1.legend(
                handles,
                labels,
                fontsize=settings.legend_fontsize,
                loc="upper left",
                bbox_to_anchor=(1.0, 1.0),
                borderaxespad=0.0,
            )
        ax = others[-1] if others else ax1
        for artist in [*ax1.lines, *ax1.collections, *ax1.patches] if others else []:
            if isinstance(artist, Line2D):
                ax.add_artist(
                    Line2D(
                        artist.get_xdata(),
                        artist.get_ydata(),
                        transform=ax1.transData,
                        visible=False,
                    )
                )
            elif isinstance(artist, Rectangle):
                ax.add_artist(
                    Rectangle(
                        artist.get_xy(),
                        artist.get_width(),
                        artist.get_height(),
                        transform=ax1.transData,
                        visible=False,
                    )
                )
            elif isinstance(artist, LineCollection):
                ax.add_artist(
                    LineCollection(
                        [path.vertices for path in artist.get_paths()],
                        transform=ax1.transData,
                        visible=False,
                    )
                )
            elif isinstance(artist, PolyCollection):
                ax.add_artist(
                    PolyCollection(
                        [path.vertices for path in artist.get_paths()],
                        transform=ax1.transData,
                        visible=False,
                    )
                )
        ax.legend(
            handles,
            labels,
            fontsize=settings.legend_fontsize,
            loc=settings.legend_loc,  # ty: ignore[invalid-argument-type] -- str setting, matplotlib expects its Literal
        )
        return None

    @classmethod
    def _row_legends(
        cls,
        fig: FigureMPL,
        cells: GridSpec,
        panels: list[_Panel],
        settings: FigureSettings,
    ) -> list[_Slot]:
        """Draw one legend per row, right of the grid (#327).

        The legend of a row has the entries of the plots of the row (a plot
        which spans rows belongs to its first row), in the order of the plots
        and of their entries; a plot whose `legend` is false has none in it,
        and an entry which reads and looks the same as an earlier one, e.g.
        the simulation of every plot, is shown once (`_entry_key`). The legend
        hangs from the top of its row, also when the last cell of the row is
        empty, and is placed right of the grid by `_place_right`.

        Args:
            fig: the figure.
            cells: the grid of the panels.
            panels: the drawn panels.
            settings: the settings of the figure.

        Returns:
            The legends, to be placed by `_place_right`.
        """
        slots: list[_Slot] = []
        for row in range(cells.nrows):
            members = sorted(
                (panel for panel in panels if panel.plot.legend and panel.top == row),
                key=lambda panel: panel.first,
            )
            seen: set[tuple[Any, ...]] = set()
            handles: list[Any] = []
            labels: list[str] = []
            for member in members:
                for handle, label in zip(*_entries(member), strict=True):
                    key = _entry_key(handle, label)
                    if key not in seen:
                        seen.add(key)
                        handles.append(handle)
                        labels.append(label)
            if not handles:
                continue
            # a legend of the figure, anchored until `_place_right` places it
            legend = fig.legend(
                handles,
                labels,
                fontsize=settings.legend_fontsize,
                loc="upper left",
                bbox_to_anchor=(1.0, 1.0),
                borderaxespad=0.0,
            )
            box = cells[row, cells.ncols - 1].get_position(fig)
            slots.append(_Slot(legend, None, box.y1, box.height, row, row))
        return slots

    @classmethod
    def _draw_bands(
        cls,
        ax: AxesMPL,
        experiment: Any,
        band: Band,
        bands: list[BandLine],
        color: str,
    ) -> PointStyle | None:
        """Draw the quantile areas and medians of a band, one per point of `over`.

        The boundaries of the area are thin lines, which the placement of a
        legend takes into account, unlike an area. The legend has one entry
        per point, its area in its colour labelled `<name> (<low>-<high> %)`
        (`<name>, <point label> (<low>-<high> %)` with `over`), or one of the
        name of a band with a colour bar, in the middle colour of the map
        (#302).

        Args:
            ax: the axes the band is drawn on.
            experiment: the experiment the dimensions are read from.
            band: the band.
            bands: its quantiles and medians, one per point of `over`.
            color: the colour of a band without `over`.

        Returns:
            The style of the points if they get a colour bar, else `None`.
        """
        styles = None
        if band.over:
            task = band.y.task_id
            dimension = experiment.scan_dimension(task, band.over[0]) if task else None
            units = experiment.model_units(task) if task else {}
            styles = point_styles(bands, band.over, band.color, [dimension], units)
        show = styles is None or not styles.colorbar
        for b in bands:
            c = styles.colors[b.index[0]] if styles else color
            name = (
                f"{band.name}, {styles.labels[0][b.index[0]]}" if styles else band.name
            )
            ax.fill_between(
                b.x,
                b.low,
                b.high,
                color=c,
                alpha=band.alpha,
                linewidth=0,
                label=band_label(name, band.quantiles) if show else "_nolegend_",
            )
            edge: dict[str, Any] = {
                "color": c,
                "linewidth": 0.6,
                "alpha": min(1.0, 2 * band.alpha),
                "label": "_nolegend_",
            }
            ax.plot(b.x, b.low, **edge)
            ax.plot(b.x, b.high, **edge)
            if band.median:
                ax.plot(b.x, b.median, color=c, linewidth=2.0, label="_nolegend_")
        if styles is not None and styles.colorbar and band.name:
            ax.fill_between(
                [],
                [],
                [],
                color=styles.middle,
                alpha=band.alpha,
                linewidth=0,
                label=band_label(band.name, band.quantiles),
            )
        return styles if styles is not None and styles.colorbar else None

    @classmethod
    def to_figure(
        cls,
        experiment: Any,
        figure: Figure,
        settings: FigureSettings | Mapping[str, Any] | None = None,
    ) -> FigureMPL:
        """Convert sbmlsim.Figure to matplotlib figure.

        Args:
            experiment: the experiment the data of the figure is read from,
                whose `figure_settings` override the settings of the run.
            figure: the figure.
            settings: the settings of the run, a `FigureSettings` or overrides
                of the defaults, which the settings of the experiment and of
                the figure override, see `Figure.resolved_settings`.

        Returns:
            The matplotlib figure.
        """
        resolved = figure.resolved_settings(experiment, run=settings)
        # the figure is created directly and not through `pyplot`, which keeps
        # every figure it creates in a global registry until someone closes it:
        # a library which renders many figures leaks them, and matplotlib warns
        # about it from the twentieth one on. Nothing here needs the state
        # machine, `savefig` works on the figure itself, and
        # `SimulationExperiment.show_mpl_figures` attaches a manager when a
        # figure is actually shown
        fig: FigureMPL = FigureMPL(
            figsize=figure.size(resolved),
            dpi=resolved.fig_dpi,
            facecolor=resolved.fig_facecolor,
        )

        title: Text | None = None
        if figure.name:
            title = fig.suptitle(
                figure.name,
                fontsize=resolved.fig_titlesize,
                fontweight=resolved.fig_titleweight,
            )

        # create grid for figure
        gs = fig.add_gridspec(nrows=figure.num_rows, ncols=figure.num_cols)

        # the spacing comes first, over the whole figure: the colour bars are
        # placed from the positions of the axes, and `subplots_adjust` would
        # move the axes back into their cells afterwards
        # a legend outside widens the gap right of its column, see `_place_right`
        wspace = resolved.fig_subplots_wspace
        hspace = resolved.fig_subplots_hspace
        fig.subplots_adjust(
            **cls._margins(figure, resolved), wspace=wspace, hspace=hspace
        )

        panels: list[_Panel] = []
        subplot: SubPlot
        for subplot in figure.subplots:
            plot = subplot.plot
            xax: Axis = plot.xaxis if plot.xaxis else Axis()
            yax: Axis = plot.yaxis if plot.yaxis else Axis()
            yax_right = plot.yaxis_right

            if subplot.row is None or subplot.col is None:
                raise ValueError(f"SubPlot requires row and col: {subplot}")
            ridx = subplot.row - 1
            cidx = subplot.col - 1
            ax1: AxesMPL = fig.add_subplot(
                gs[ridx : ridx + subplot.row_span, cidx : cidx + subplot.col_span]
            )
            # secondary axis
            ax2: AxesMPL | None = None
            axes: list[AxesMPL] = [ax1]
            if yax_right:
                for curve in [*plot.curves, *plot.areas, *plot.bands]:
                    if (
                        curve.yaxis_position
                        and curve.yaxis_position == YAxisPosition.RIGHT
                    ):
                        ax2 = ax1.twinx()
                        axes.append(ax2)
                        break
                else:
                    logger.error("Position right defined by no yAxis right.")

            # `xax` and `yax` fall back to an empty `Axis` above, so a plot
            # which names neither is drawn without units rather than refused;
            # the spines of an axis which is not there are hidden further down
            if plot.xaxis is None:
                logger.warning("No xaxis in plot: %s", subplot)
            if plot.yaxis is None:
                logger.warning("No yaxis in plot: %s", subplot)

            xunit = xax.unit
            yunit_left = yax.unit
            yunit_right = yax_right.unit if yax_right else None

            # the plot decides the colour of its panel
            if plot.facecolor:
                ax1.set_facecolor(plot.facecolor.color)

            # memory for stacked bars
            stacks: dict[str, Any] = {}
            # the colour bars of the panel are placed after every panel got
            # its labels
            panel = _Panel(
                plot=plot,
                axes=axes,
                first=cidx,
                last=cidx + subplot.col_span - 1,
                top=ridx,
                bottom=ridx + subplot.row_span - 1,
            )
            panels.append(panel)
            pending = panel.pending

            # plot ordered curves
            abstract_curves: list[AbstractCurve] = sorted(
                [*plot.curves, *plot.areas, *plot.bands],
                key=lambda x: x.order if x.order is not None else 0,
            )
            ax: AxesMPL
            for abstract_curve in abstract_curves:
                if (
                    abstract_curve.yaxis_position
                    and abstract_curve.yaxis_position == YAxisPosition.RIGHT
                ):
                    # right axis
                    if ax2 is None:
                        raise ValueError("Curve on right yaxis, but no right yaxis.")
                    yunit = yunit_right
                    ax = ax2
                else:
                    # left axis
                    yunit = yunit_left
                    ax = ax1

                if isinstance(abstract_curve, Curve):
                    # --- Curve ---
                    curve: Curve = abstract_curve
                    x = curve.x.get_data(experiment=experiment, to_units=xunit)
                    y = curve.y.get_data(experiment=experiment, to_units=yunit)
                    xerr = None
                    if curve.xerr is not None:
                        xerr = curve.xerr.get_data(
                            experiment=experiment, to_units=xunit
                        )
                    yerr = None
                    if curve.yerr is not None:
                        yerr = curve.yerr.get_data(
                            experiment=experiment, to_units=yunit
                        )
                    sid = curve.sid or curve.name or ""
                    lines = curve_lines(sid, curve.over, x, y, xerr, yerr)
                    kwargs = cls._curve_kwargs(curve)
                    if not curve.over:
                        (line,) = lines
                        cls._draw_curve(
                            ax, curve, line, kwargs, curve.name or "_nolegend_", stacks
                        )
                        continue
                    styles = cls._draw_points(ax, experiment, curve, lines, kwargs)
                    if styles is not None:
                        pending.append(styles)

                elif isinstance(abstract_curve, Band):
                    # --- Band ---
                    band: Band = abstract_curve
                    x = band.x.get_data(experiment=experiment, to_units=xunit)
                    y = band.y.get_data(experiment=experiment, to_units=yunit)
                    bands = band_lines(
                        band.sid or band.name or "",
                        band.over,
                        band.across,
                        band.quantiles,
                        x,
                        y,
                    )
                    # a band without a colour takes the colour of the cycle at
                    # its position among the bands of the plot
                    styles = cls._draw_bands(
                        ax,
                        experiment,
                        band,
                        bands,
                        band.color or cycle_color(plot.bands.index(band)),
                    )
                    if styles is not None:
                        pending.append(styles)

                elif isinstance(abstract_curve, ShadedArea):
                    # --- ShadedArea ---
                    area: ShadedArea = abstract_curve
                    x = area.x.get_data(experiment=experiment, to_units=xunit)
                    yfrom = area.yfrom.get_data(experiment=experiment, to_units=yunit)
                    yto = area.yto.get_data(experiment=experiment, to_units=yunit)

                    x_data, yfrom_data, yto_data = line_values(
                        area.sid or area.name or "", x, yfrom, yto
                    )

                    label = area.name if area.name else "_nolegend_"
                    kwargs: dict[str, Any] = {}
                    if area.style:
                        style: Style = area.style.resolve_style()
                        kwargs = style.to_mpl_area_kwargs()

                    ax.fill_between(
                        x=x_data, y1=yfrom_data, y2=yto_data, label=label, **kwargs
                    )

            # plot settings
            if plot.name and plot.title_visible:
                ax1.set_title(plot.name)

            def apply_axis_settings(sax: Axis, ax: AxesMPL, axis_type: str):
                """Apply settings to all axis."""
                if axis_type not in ["x", "y"]:
                    raise ValueError

                # the scale first: a bound set on a linear axis fixes the
                # other limit at the linear autoscale limit, without the
                # margin of a log axis
                if axis_type == "x":
                    ax.set_xscale(cls._get_scale(sax))
                elif axis_type == "y":
                    ax.set_yscale(cls._get_scale(sax))

                # a log axis shows at most `LOG_DECADES` below the largest
                # value, not the tiny first steps of the integrator
                if sax.scale == AxisScale.LOG10:
                    limit_log_axis(ax, axis_type, sax.min, sax.max)

                if sax.min is not None:
                    if axis_type == "x":
                        ax.set_xlim(left=sax.min)
                    elif axis_type == "y":
                        ax.set_ylim(bottom=sax.min)
                if sax.max is not None:
                    if axis_type == "x":
                        ax.set_xlim(right=sax.max)
                    elif axis_type == "y":
                        ax.set_ylim(top=sax.max)

                # the bounds are the bounds of the data, `reverse` is the
                # direction they are drawn in, so it is applied to whatever the
                # limits ended up being: swapping `min` and `max` above did
                # nothing unless both of them were set
                if sax.reverse:
                    if axis_type == "x":
                        ax.invert_xaxis()
                    elif axis_type == "y":
                        ax.invert_yaxis()

                if sax.label_visible and sax.name:
                    if axis_type == "x":
                        ax.set_xlabel(sax.name)
                    elif axis_type == "y":
                        ax.set_ylabel(sax.name)

                if not sax.ticks_visible:
                    # `set_xticklabels([])` needs a fixed locator to be correct
                    # and leaves the tick marks drawn; `tick_params` is what
                    # hides the labels of whatever the locator produces
                    if axis_type == "x":
                        ax.tick_params(axis="x", labelbottom=False)
                    elif axis_type == "y":
                        ax.tick_params(axis="y", labelleft=False)

                # style
                # https://matplotlib.org/stable/api/spines_api.html
                # http://matplotlib.org/examples/pylab_examples/multiple_yaxis_with_spines.html
                if sax.style and sax.style.line:
                    if axis_type == "x":
                        directions = ["bottom", "top"]
                    elif axis_type == "y":
                        directions = ["left", "right"]

                    style: Style = sax.style.resolve_style()
                    if style.line:
                        if style.line.thickness:
                            linewidth = style.line.thickness
                            for axis in directions:
                                ax.tick_params(width=linewidth)
                                if np.isclose(linewidth, 0.0):
                                    ax.spines[axis].set_visible(False)
                                else:
                                    ax.spines[axis].set_linewidth(linewidth)

                        if style.line.color:
                            color = style.line.color
                            for axis in directions:
                                ax.spines[axis].set_color(str(color))

                        # a spine which is not drawn is hidden, painting it in
                        # the colour of the figure only works while the panel
                        # has that colour, see `Plot.facecolor`
                        if style.line.type == LineType.NONE:
                            for axis in directions:
                                ax.spines[axis].set_visible(False)

            apply_axis_settings(xax, ax1, axis_type="x")
            apply_axis_settings(yax, ax1, axis_type="y")
            if yax_right and ax2 is not None:
                apply_axis_settings(yax_right, ax2, axis_type="y")

            # recompute the ax.dataLim
            # ax.relim()
            # update ax.viewLim using the new dataLim
            # ax.autoscale_view()

            # figure styling
            for ax in axes:
                ax.title.set_fontsize(resolved.axes_titlesize)
                ax.title.set_fontweight(resolved.axes_titleweight)
                ax.xaxis.label.set_fontsize(resolved.axes_labelsize)
                ax.xaxis.label.set_fontweight(resolved.axes_labelweight)
                ax.yaxis.label.set_fontsize(resolved.axes_labelsize)
                ax.yaxis.label.set_fontweight(resolved.axes_labelweight)
                ax.tick_params(axis="x", labelsize=resolved.xtick_labelsize)
                ax.tick_params(axis="y", labelsize=resolved.ytick_labelsize)

            # hide none-existing axes; the horizontal spines belong to the x
            # axis and the vertical ones to the y axis, and they are hidden on
            # `ax1` and not on whatever `ax` was left over from the loop above
            if plot.xaxis is None:
                ax1.spines["bottom"].set_visible(False)
                ax1.spines["top"].set_visible(False)
                ax1.xaxis.set_visible(False)

            if plot.yaxis is None:
                ax1.spines["left"].set_visible(False)
                ax1.spines["right"].set_visible(False)
                ax1.yaxis.set_visible(False)

            xgrid = xax.grid
            ygrid = yax.grid

            if xgrid and ygrid:
                ax1.grid(True, axis="both")
            elif xgrid:
                ax1.grid(True, axis="x")
            elif ygrid:
                ax1.grid(True, axis="y")
            else:
                ax1.grid(False)

        # the margins fit the ticks and the labels of the panels at the edges,
        # before the legends and the colour bars are placed from the positions
        renderer = _measuring_renderer(fig)
        cls._fit_margins(fig, panels, title, renderer)

        # the legends, then the colour bars and the legends outside, once every
        # axes has its ticks and labels
        slots: list[_Slot] = []
        if resolved.legend_scope == "row":
            slots = cls._row_legends(fig, gs, panels, resolved)
        else:
            for panel in panels:
                if not panel.plot.legend:
                    continue
                legend = cls._legend(fig, resolved, panel)
                if legend is not None:
                    box = panel.axes[0].get_position()
                    slots.append(
                        _Slot(
                            legend, panel, box.y1, box.height, panel.top, panel.bottom
                        )
                    )
        cls._place_right(
            fig,
            (figure.num_rows, figure.num_cols),
            panels,
            resolved,
            title,
            slots,
            grid_right=gs[0, figure.num_cols - 1].get_position(fig).x1,
            renderer=renderer,
        )

        _BUILT.add(fig)
        return fig

    @staticmethod
    def _wrap_label(
        label: Text, length: float, vertical: bool, renderer: RendererBase
    ) -> None:
        """Wrap the label of an axis which is longer than the axis.

        The label is broken at its words into the fewest lines each of which
        is at most as long as the axis (a word longer than it stays a line of
        its own); a label of the y axis longer than the panel reached far above
        and below it.

        Args:
            label: the label of an axis.
            length: the length of the axis in pixels.
            vertical: whether the label is along a vertical axis.
            renderer: the renderer which measures the label.
        """
        text = label.get_text()

        def extent() -> float:
            box = label.get_window_extent(renderer)
            return box.height if vertical else box.width

        if not text or "\n" in text or extent() <= length:
            return
        for width in range(len(text) - 1, 0, -1):
            label.set_text(textwrap.fill(text, width, break_long_words=False))
            if extent() <= length:
                return
        label.set_text(textwrap.fill(text, 1, break_long_words=False))

    @classmethod
    def _fit_margins(
        cls,
        fig: FigureMPL,
        panels: list[_Panel],
        title: Text | None,
        renderer: RendererBase,
    ) -> None:
        """Grow the figure by what the ticks and labels need beyond its margins.

        The margins of matplotlib are fractions sized for its fonts of 10 pt,
        the fonts of sbmlsim are larger: the label of the x axis ended below
        the figure, cut off unless it was saved with `bbox_inches="tight"`
        (#362). A label longer than its axis is wrapped first, see
        `_wrap_label`. The tight boxes of the drawn axes give the room the ticks and
        the labels need around the panels (below the title of the figure), at
        least `EDGE_PAD`; the figure grows by what is missing, so the panels
        keep their size in inches and a figure whose labels fit keeps its
        size. The colour bars and the legends outside get their room right of
        the panels from `_place_right` afterwards.
        """
        for panel in panels:
            for ax in panel.axes:
                box = ax.get_window_extent(renderer)
                cls._wrap_label(ax.xaxis.label, box.width, False, renderer)
                cls._wrap_label(ax.yaxis.label, box.height, True, renderer)
        width, height, dpi = fig.get_figwidth(), fig.get_figheight(), fig.dpi
        boxes = [
            box
            for panel in panels
            for ax in panel.axes
            if (box := ax.get_tightbbox(renderer)) is not None
        ]
        if not boxes:
            return
        ceiling = height
        if title is not None:
            ceiling = title.get_window_extent(renderer).y0 / dpi
        left = max(0.0, EDGE_PAD - min(box.x0 for box in boxes) / dpi)
        bottom = max(0.0, EDGE_PAD - min(box.y0 for box in boxes) / dpi)
        top = max(0.0, max(box.y1 for box in boxes) / dpi + EDGE_PAD - ceiling)
        right = max(0.0, max(box.x1 for box in boxes) / dpi + EDGE_PAD - width)
        if not (left or right or bottom or top):
            return
        params = fig.subplotpars
        wider, taller = width + left + right, height + bottom + top
        fig.set_size_inches(wider, taller)
        fig.subplots_adjust(
            left=(params.left * width + left) / wider,
            right=(params.right * width + left) / wider,
            bottom=(params.bottom * height + bottom) / taller,
            top=(params.top * height + bottom) / taller,
        )
        if title is not None:
            # the title keeps its distance to the top of the figure
            title.set_y(1.0 - (1.0 - title.get_position()[1]) * height / taller)

    @staticmethod
    def _margins(figure: Figure, settings: FigureSettings) -> dict[str, float]:
        """Get the margins of the grid as fractions of the size of the figure.

        The margins are the ones of a figure of one panel, the fractions of
        matplotlib (`figure.subplot.left`, `right`, `bottom` and `top`) of
        the size of a panel, i.e., of the size of the figure divided by its
        number of columns or rows. The fractions of the whole figure grew with
        the number of panels, whose size is fixed in inches: a figure of six
        rows had 3.3 in below its last row, one of six columns 5.25 in left of
        its first (#323).

        The title of the figure and the titles of the plots below it need a
        band of a fixed height, i.e., the smaller the figure, the larger the
        part of it the band takes: the title of the figure is placed at 98% of
        the height, followed by its line, the line of the title of a plot and
        the padding of both. The plots begin right below the band whatever the
        height of the figure; the fraction of matplotlib capped them, so the
        gap below the title grew with the height, 3.6 in above the plots of a
        figure of six rows (#312).
        """
        cols, rows = figure.num_cols, figure.num_rows
        margins = {
            "left": float(rcParams["figure.subplot.left"]) / cols,
            "right": 1.0 - (1.0 - float(rcParams["figure.subplot.right"])) / cols,
            "bottom": float(rcParams["figure.subplot.bottom"]) / rows,
            "top": 1.0 - (1.0 - float(rcParams["figure.subplot.top"])) / rows,
        }
        if figure.name:
            band = (
                1.2 * (settings.fig_titlesize + settings.axes_titlesize) + 12.0
            ) / 72.0
            margins["top"] = 0.98 - band / figure.size(settings)[1]
        return margins
