"""Serialization of Figure object to matplotlib."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from matplotlib.axes import Axes as AxesMPL
from matplotlib.figure import Figure as FigureMPL

from sbmlsim.plot import Axis, Curve, Figure, SubPlot
from sbmlsim.plot.plotting import (
    AbstractCurve,
    AxisScale,
    CurveType,
    LineType,
    ShadedArea,
    Style,
    YAxisPosition,
)

logger = logging.getLogger(__name__)


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

        # create grid for figure; the spacing is applied with `subplots_adjust`
        # at the end, over the whole figure
        gs = fig.add_gridspec(nrows=figure.num_rows, ncols=figure.num_cols)

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
                for curve in plot.curves:
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
            barstack_x = None
            barstack_y = None
            barhstack_x = None
            barhstack_y = None

            # plot ordered curves
            abstract_curves: list[AbstractCurve] = sorted(
                [*plot.curves, *plot.areas],
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

                    label = curve.name if curve.name else "_nolegend_"

                    # FIXME: necessary to get the individual curves out of the data cube
                    # TODO: iterate over all repeats in the data
                    if x is None:
                        x_data = None
                    else:
                        x_data = (
                            x.magnitude[:, 0]
                            if np.ndim(x.magnitude) == 2
                            else x.magnitude
                        )

                    if y is None:
                        y_data = None
                    else:
                        y_data = (
                            y.magnitude[:, 0]
                            if np.ndim(y.magnitude) == 2
                            else y.magnitude
                        )

                    if xerr is None:
                        xerr_data = None
                    else:
                        xerr_data = (
                            xerr.magnitude[:, 0]
                            if np.ndim(xerr.magnitude) == 2
                            else xerr.magnitude
                        )

                    if yerr is None:
                        yerr_data = None
                    else:
                        yerr_data = (
                            yerr.magnitude[:, 0]
                            if np.ndim(yerr.magnitude) == 2
                            else yerr.magnitude
                        )

                    kwargs: dict[str, Any] = {}
                    if curve.style:
                        style: Style = curve.style.resolve_style()
                        if curve.type == CurveType.POINTS:
                            kwargs = style.to_mpl_points_kwargs()
                        else:
                            # bar plot
                            kwargs = style.to_mpl_bar_kwargs()

                    if curve.type == CurveType.POINTS:
                        if xerr_data is None and yerr_data is None:
                            # `errorbar` builds the containers of the bars
                            # whether or not there are any, and is twice the
                            # cost of `plot` for the same line
                            kwargs.pop("capsize", None)
                            ax.plot(x_data, y_data, label=label, **kwargs)
                        else:
                            ax.errorbar(
                                x=x_data,
                                y=y_data,
                                xerr=xerr_data,
                                yerr=yerr_data,
                                label=label,
                                **kwargs,
                            )

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
                        if barstack_x is None:
                            barstack_x = x_data
                            barstack_y = np.zeros_like(y_data)

                        if not np.all(np.isclose(barstack_x, x_data)):
                            raise ValueError("x data must match for stacked bars.")
                        ax.bar(
                            x=x_data,
                            height=y_data,
                            bottom=barstack_y,
                            xerr=xerr_data,
                            yerr=yerr_data,
                            label=label,
                            **kwargs,
                        )
                        barstack_y = barstack_y + y_data

                    elif curve.type == CurveType.HORIZONTALBARSTACKED:
                        if barhstack_x is None:
                            barhstack_x = x_data
                            barhstack_y = np.zeros_like(y_data)

                        if not np.all(np.isclose(barhstack_x, x_data)):
                            raise ValueError("x data must match for stacked bars.")
                        ax.barh(
                            y=x_data,
                            width=y_data,
                            left=barhstack_y,
                            xerr=yerr_data,
                            yerr=xerr_data,
                            label=label,
                            **kwargs,
                        )
                        barhstack_y = barhstack_y + y_data

                elif isinstance(abstract_curve, ShadedArea):
                    # --- ShadedArea ---
                    area: ShadedArea = abstract_curve
                    x = area.x.get_data(experiment=experiment, to_units=xunit)
                    yfrom = area.yfrom.get_data(experiment=experiment, to_units=yunit)
                    yto = area.yto.get_data(experiment=experiment, to_units=yunit)

                    # FIXME: support multidimensional results
                    x_data = x.magnitude[:, 0] if x is not None else None
                    yfrom_data = yfrom.magnitude[:, 0] if yfrom is not None else None
                    yto_data = yto.magnitude[:, 0] if yto is not None else None

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

                if axis_type == "x":
                    ax.set_xscale(cls._get_scale(sax))
                elif axis_type == "y":
                    ax.set_yscale(cls._get_scale(sax))

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

            if plot.legend:
                outside = figure.legend_position == "outside"
                if ax2 is None:
                    handles1, _ = ax1.get_legend_handles_labels()
                    if handles1:
                        if outside:
                            ax1.legend(
                                fontsize=Figure.legend_fontsize,
                                loc="upper left",
                                bbox_to_anchor=(1.04, 1),
                            )
                        else:
                            ax1.legend(
                                fontsize=Figure.legend_fontsize,
                                loc=Figure.legend_loc,  # ty: ignore[invalid-argument-type] -- str setting, matplotlib expects its Literal
                            )
                elif outside:
                    # two legends outside would sit on top of each other, so
                    # the curves of both axes go into one; `legend_position`
                    # was honoured for a single axis only
                    handles1, labels1 = ax1.get_legend_handles_labels()
                    handles2, labels2 = ax2.get_legend_handles_labels()
                    if handles1 or handles2:
                        ax1.legend(
                            handles1 + handles2,
                            labels1 + labels2,
                            fontsize=Figure.legend_fontsize,
                            loc="upper left",
                            bbox_to_anchor=(1.04, 1),
                        )
                else:
                    handles1, _ = ax1.get_legend_handles_labels()
                    if handles1:
                        ax1.legend(fontsize=Figure.legend_fontsize, loc="upper left")
                    handles2, _ = ax2.get_legend_handles_labels()
                    if handles2:
                        ax2.legend(fontsize=Figure.legend_fontsize, loc="upper right")

        wspace = figure.fig_subplots_wspace
        hspace = figure.fig_subplots_hspace
        if figure.legend_position == "outside":
            wspace += 1.0
        fig.subplots_adjust(wspace=wspace, hspace=hspace)

        return fig
