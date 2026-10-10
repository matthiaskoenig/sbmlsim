"""Serialization of Figure object to matplotlib."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes as AxesMPL
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm, to_hex
from matplotlib.figure import Figure as FigureMPL
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

from sbmlsim.plot import Axis, Curve, Figure, Plot, SubPlot
from sbmlsim.plot.padding import line_values
from sbmlsim.plot.plotting import (
    AbstractCurve,
    AxisScale,
    Band,
    CurveType,
    LineType,
    ShadedArea,
    Style,
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

#: the gap in inches between a panel (or the ticks and the title of its right y
#: axis), each of its colour bars and an outside legend
COLORBAR_GAP = 0.12

#: the width of a colour bar in inches
COLORBAR_WIDTH = 0.15


@dataclass
class _Panel:
    """A drawn panel, whose colour bars and legend are placed after all panels.

    Attributes:
        plot: the plot of the panel.
        axes: its axes, the left one first and the right one if it has one.
        first: the first column of the grid the panel spans, from 0.
        last: the last column of the grid the panel spans.
        pending: the styles of its colour bars, in the order of its curves.
        edge: the right edge of its last colour bar with its ticks and label,
            in figure coordinates, `None` without colour bars.
    """

    plot: Plot
    axes: list[AxesMPL]
    first: int
    last: int
    pending: list[PointStyle] = field(default_factory=list)
    edge: float | None = None


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
    def _colorbar(cls, fig: FigureMPL, cax: AxesMPL, styles: PointStyle) -> Colorbar:
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
            fontsize=Figure.axes_labelsize,
            fontweight=Figure.axes_labelweight,
        )
        colorbar.ax.tick_params(labelsize=Figure.ytick_labelsize)
        return colorbar

    @classmethod
    def _place_colorbars(
        cls, fig: FigureMPL, num_cols: int, panels: list[_Panel]
    ) -> None:
        """Place the colour bars of every panel side by side right of it.

        A panel needs room for the ticks and the title of a right y axis and,
        per bar, a gap, the bar and its ticks and label, which are measured on
        the drawn bar. Every column of the grid gets the room its widest panel
        needs and the figure is that much wider, so the panels keep their size
        and the columns their alignment, and nothing overlaps. The right edge
        of the last bar of a panel, where an outside legend starts, is set as
        its `edge`.
        """
        width, dpi = fig.get_figwidth(), fig.dpi
        boxes = [panel.axes[0].get_position() for panel in panels]
        rooms: list[list[float]] = []
        bars: list[list[Colorbar]] = []
        for panel, box in zip(panels, boxes, strict=True):
            panel_rooms: list[float] = []
            panel_bars: list[Colorbar] = []
            if panel.pending:
                # the ticks and the title of a right y axis, in inches
                right = 0.0
                for ax in panel.axes[1:]:
                    tight = ax.yaxis.get_tightbbox()
                    if tight is not None:
                        right = max(right, tight.x1 / dpi - box.x1 * width)
                panel_rooms.append(right)
                for styles in panel.pending:
                    cax = fig.add_axes(
                        (box.x1, box.y0, COLORBAR_WIDTH / width, box.height)
                    )
                    panel_bars.append(cls._colorbar(fig, cax, styles))
                    tight = cax.get_tightbbox()
                    end = tight.x1 / dpi if tight is not None else 0.0
                    panel_rooms.append(max(0.0, end - box.x1 * width - COLORBAR_WIDTH))
            rooms.append(panel_rooms)
            bars.append(panel_bars)

        def need(panel_rooms: list[float]) -> float:
            """Get the room of the bars of a panel right of it, in inches."""
            if not panel_rooms:
                return 0.0
            right, *labels = panel_rooms
            return right + sum(COLORBAR_GAP + COLORBAR_WIDTH + r for r in labels)

        columns = [0.0] * num_cols
        for panel, panel_rooms in zip(panels, rooms, strict=True):
            columns[panel.last] = max(columns[panel.last], need(panel_rooms))
        if not any(columns):
            return
        wider = width + sum(columns)
        fig.set_size_inches(wider, fig.get_figheight())
        for panel, box, panel_rooms, panel_bars in zip(
            panels, boxes, rooms, bars, strict=True
        ):
            x0 = box.x0 * width + sum(columns[: panel.first])
            x1 = box.x1 * width + sum(columns[: panel.last])
            for ax in panel.axes:
                ax.set_position((x0 / wider, box.y0, (x1 - x0) / wider, box.height))
            if not panel_bars:
                continue
            x = x1 + panel_rooms[0]
            for bar, room in zip(panel_bars, panel_rooms[1:], strict=True):
                x += COLORBAR_GAP
                bar.ax.set_position(
                    (x / wider, box.y0, COLORBAR_WIDTH / wider, box.height)
                )
                x += COLORBAR_WIDTH + room
            panel.edge = x / wider

    @classmethod
    def _legend(cls, fig: FigureMPL, figure: Figure, panel: _Panel) -> None:
        """Draw the legend of a panel.

        Outside, one legend of both axes is right of the panel, or right of its
        last colour bar. Inside, a panel with a right y axis gets one legend of
        both axes as well, on the right axes, which are drawn last, placed
        against the data of both: the lines and areas of the left axes are on
        the right axes as invisible copies, which only the placement sees.
        """
        ax1, *others = panel.axes
        handles, labels = ax1.get_legend_handles_labels()
        for ax in others:
            more_handles, more_labels = ax.get_legend_handles_labels()
            handles += more_handles
            labels += more_labels
        if not handles:
            return
        if figure.legend_position == "outside":
            anchor: dict[str, Any] = {"bbox_to_anchor": (1.04, 1)}
            if panel.edge is not None:
                anchor = {
                    "bbox_to_anchor": (
                        panel.edge + COLORBAR_GAP / fig.get_figwidth(),
                        ax1.get_position().y1,
                    ),
                    "bbox_transform": fig.transFigure,
                }
            ax1.legend(
                handles,
                labels,
                fontsize=Figure.legend_fontsize,
                loc="upper left",
                **anchor,
            )
            return
        ax = others[-1] if others else ax1
        for artist in [*ax1.lines, *ax1.collections] if others else []:
            if isinstance(artist, Line2D):
                ax.add_artist(
                    Line2D(
                        artist.get_xdata(),
                        artist.get_ydata(),
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
            fontsize=Figure.legend_fontsize,
            loc=Figure.legend_loc,  # ty: ignore[invalid-argument-type] -- str setting, matplotlib expects its Literal
        )

    @classmethod
    def _draw_bands(
        cls,
        ax: AxesMPL,
        experiment: Any,
        band: Band,
        bands: list[BandLine],
        color: str,
        ranges: set[str],
    ) -> PointStyle | None:
        """Draw the quantile areas and medians of a band, one per point of `over`.

        The boundaries of the area are thin lines, which the placement of a
        legend takes into account, unlike an area. The legend has one entry
        per point (the median, else the upper boundary), or one of the name of
        a band with a colour bar, and a grey entry of the quantile range once
        per panel and range.

        Args:
            ax: the axes the band is drawn on.
            experiment: the experiment the dimensions are read from.
            band: the band.
            bands: its quantiles and medians, one per point of `over`.
            color: the colour of a band without `over`.
            ranges: the quantile ranges of the panel with a legend entry, which
                this one is added to.

        Returns:
            The style of the points if they get a colour bar, else `None`.
        """
        low, high = (f"{100 * q:g}" for q in band.quantiles)
        styles = None
        if band.over:
            task = band.y.task_id
            dimension = experiment.scan_dimension(task, band.over[0]) if task else None
            units = experiment.model_units(task) if task else {}
            styles = point_styles(bands, band.over, band.color, [dimension], units)
        show = styles is None or not styles.colorbar
        for b in bands:
            c = styles.colors[b.index[0]] if styles else color
            if styles:
                label = f"{band.name}, {styles.labels[0][b.index[0]]}"
            else:
                label = f"{band.name} median" if band.median else band.name
            label = label if show else "_nolegend_"
            ax.fill_between(b.x, b.low, b.high, color=c, alpha=band.alpha, linewidth=0)
            edge: dict[str, Any] = {
                "color": c,
                "linewidth": 0.6,
                "alpha": min(1.0, 2 * band.alpha),
            }
            ax.plot(b.x, b.low, label="_nolegend_", **edge)
            ax.plot(b.x, b.high, label="_nolegend_" if band.median else label, **edge)
            if band.median:
                ax.plot(b.x, b.median, color=c, linewidth=2.0, label=label)
        if styles is not None and styles.colorbar and band.name:
            ax.plot([], [], color=styles.middle, linewidth=2.0, label=band.name)
        quantiles = f"{low}-{high} %"
        if quantiles not in ranges:
            ranges.add(quantiles)
            ax.fill_between(
                [],
                [],
                [],
                color="0.5",
                alpha=band.alpha,
                linewidth=0,
                label=quantiles,
            )
        return styles if styles is not None and styles.colorbar else None

    @classmethod
    def to_figure(
        cls,
        experiment,  # "SimulationExperiment",
        figure: Figure,
    ) -> FigureMPL:
        """Convert sbmlsim.Figure to matplotlib figure."""
        # the figure is created directly and not through `pyplot`, which keeps
        # every figure it creates in a global registry until someone closes it:
        # a library which renders many figures leaks them, and matplotlib warns
        # about it from the twentieth one on. Nothing here needs the state
        # machine, `savefig` works on the figure itself, and
        # `SimulationExperiment.show_mpl_figures` attaches a manager when a
        # figure is actually shown
        fig: FigureMPL = FigureMPL(
            figsize=(figure.width, figure.height),
            dpi=Figure.fig_dpi,
            facecolor=Figure.fig_facecolor,
        )

        if figure.name:
            fig.suptitle(
                figure.name,
                fontsize=Figure.fig_titlesize,
                fontweight=Figure.fig_titleweight,
            )

        # create grid for figure
        gs = fig.add_gridspec(nrows=figure.num_rows, ncols=figure.num_cols)

        # the spacing comes first, over the whole figure: the colour bars are
        # placed from the positions of the axes, and `subplots_adjust` would
        # move the axes back into their cells afterwards
        wspace = figure.fig_subplots_wspace
        hspace = figure.fig_subplots_hspace
        if figure.legend_position == "outside":
            wspace += 1.0
        fig.subplots_adjust(top=cls._top(figure), wspace=wspace, hspace=hspace)

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
            # its labels; the quantile ranges with a legend entry
            panel = _Panel(
                plot=plot, axes=axes, first=cidx, last=cidx + subplot.col_span - 1
            )
            panels.append(panel)
            pending = panel.pending
            ranges: set[str] = set()

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
                        ranges,
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
                ax.title.set_fontsize(Figure.axes_titlesize)
                ax.title.set_fontweight(Figure.axes_titleweight)
                ax.xaxis.label.set_fontsize(Figure.axes_labelsize)
                ax.xaxis.label.set_fontweight(Figure.axes_labelweight)
                ax.yaxis.label.set_fontsize(Figure.axes_labelsize)
                ax.yaxis.label.set_fontweight(Figure.axes_labelweight)
                ax.tick_params(axis="x", labelsize=Figure.xtick_labelsize)
                ax.tick_params(axis="y", labelsize=Figure.ytick_labelsize)

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

        # the colour bars and the legends, once every axes has its ticks and
        # labels
        cls._place_colorbars(fig, figure.num_cols, panels)
        for panel in panels:
            if panel.plot.legend:
                cls._legend(fig, figure, panel)

        return fig

    @staticmethod
    def _top(figure: Figure) -> float:
        """Get the top of the plots as a fraction of the height of the figure.

        The title of the figure and the titles of the plots below it need a
        band of a fixed height, i.e., the smaller the figure, the larger the
        part of it the band takes: the title of the figure is placed at 98% of
        the height, followed by its line, the line of the title of a plot and
        the padding of both. A figure without a title keeps the top of
        matplotlib.
        """
        top = float(rcParams["figure.subplot.top"])
        if not figure.name:
            return top
        band = (1.2 * (Figure.fig_titlesize + Figure.axes_titlesize) + 12.0) / 72.0
        return min(top, 0.98 - band / figure.height)
