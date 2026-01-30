"""Serialization of Figure object to matplotlib."""

from __future__ import annotations
from typing import Any, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure as FigureMPL
from matplotlib.gridspec import GridSpec
from pymetadata import log

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


logger = log.get_logger(__name__)


def interp(x, xp, fp):
    """Interpolation for speedup of plots.

    :param x:
    :param xp:
    :param fp:
    :return:
    """
    y = np.interp(x=x, xp=xp, fp=fp)
    # better spline interpolation, but NaN issues with zero values
    # tck, fp, ier, msg = interpolate.splrep(xp, fp, full_output=True)
    # if ier > 0:
    #     logger.error(f"Spline fitting failed: '{msg}'")
    #
    # y = interpolate.splev(x, tck, der=0)
    if not np.all(np.isfinite(y)):
        logger.error(f"NaN or Inf values in interpolation: {fp} -> {y}")
    return y


class MatplotlibFigureSerializer:
    """Serializer for figures to matplotlib."""

    @classmethod
    def _get_scale(cls, axis: Axis) -> str:
        """Get string representation of the scale."""
        if axis.scale == AxisScale.LINEAR:
            return "linear"
        elif axis.scale == AxisScale.LOG10:
            return "log"
        else:
            raise ValueError(f"Unsupported axis scale: '{axis.scale}'")

    @classmethod
    def to_figure(
        cls,
        experiment,  # "SimulationExperiment",
        figure: Figure,  # noqa: F821
    ) -> FigureMPL:
        """Convert sbmlsim.Figure to matplotlib figure."""

        # create new figure
        fig: plt.Figure = plt.figure(
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
        gs = GridSpec(
            nrows=figure.num_rows,
            ncols=figure.num_cols,
            figure=fig,
            # done via subplots adjust below
            # hspace=figure.fig_subplots_hspace,
            # wspace=figure.fig_subplots_wspace,
        )

        subplot: SubPlot
        for subplot in figure.subplots:
            plot = subplot.plot
            xax: Axis = plot.xaxis if plot.xaxis else Axis()
            yax: Axis = plot.yaxis if plot.yaxis else Axis()
            yax_right = plot.yaxis_right

            ridx = subplot.row - 1
            cidx = subplot.col - 1
            ax1: plt.Axes = fig.add_subplot(
                gs[ridx : ridx + subplot.row_span, cidx : cidx + subplot.col_span]
            )
            # secondary axis
            ax2: Optional[plt.Axes] = None
            axes: list[plt.Axes] = [ax1]
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

            # units
            if xax is None:
                logger.warning(f"No xaxis in plot: {subplot}")
                ax1.spines["bottom"].set_color(Figure.fig_facecolor)
                ax1.spines["top"].set_color(Figure.fig_facecolor)
            if yax is None:
                logger.warning(f"No yaxis in plot: {subplot}")
                ax1.spines["right"].set_color(Figure.fig_facecolor)
                ax1.spines["left"].set_color(Figure.fig_facecolor)
            if (not xax) or (not yax):
                if len(plot.curves) > 0:
                    raise ValueError(
                        f"xaxis and yaxis are required for plotting curves, but "
                        f"'xaxis={xax}' and 'yaxis={yax}'."
                    )

            xunit = xax.unit if xax else None
            yunit_left = yax.unit if yax else None
            yunit_right = yax_right.unit if yax_right else None

            # memory for stacked bars
            barstack_x = None
            barstack_y = None
            barhstack_x = None
            barhstack_y = None

            # plot ordered curves
            abstract_curves: list[AbstractCurve] = sorted(
                plot.curves + plot.areas, key=lambda x: x.order
            )
            for abstract_curve in abstract_curves:
                if (
                    abstract_curve.yaxis_position
                    and abstract_curve.yaxis_position == YAxisPosition.RIGHT
                ):
                    # right axis
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

                    label = curve.name if curve.name else "__nolabel__"

                    # FIXME: necessary to get the individual curves out of the data cube
                    # TODO: iterate over all repeats in the data
                    if x is None:
                        x_data = None
                    else:
                        x_data = x.magnitude[:, 0] if len(x.shape) == 2 else x.magnitude

                    if y is None:
                        y_data = None
                    else:
                        y_data = y.magnitude[:, 0] if len(y.shape) == 2 else y.magnitude

                    if xerr is None:
                        xerr_data = None
                    else:
                        xerr_data = (
                            xerr.magnitude[:, 0]
                            if len(xerr.shape) == 2
                            else xerr.magnitude
                        )

                    if yerr is None:
                        yerr_data = None
                    else:
                        yerr_data = (
                            yerr.magnitude[:, 0]
                            if len(yerr.shape) == 2
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

                    label = area.name if area.name else "__nolabel__"
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

            def apply_axis_settings(sax: Axis, ax: plt.Axes, axis_type: str):
                """Apply settings to all axis."""
                if axis_type not in ["x", "y"]:
                    raise ValueError

                # handle the reverse flag
                if sax.reverse:
                    ax_min, ax_max = sax.max, sax.min
                else:
                    ax_min, ax_max = sax.min, sax.max

                if sax.min is not None:
                    if axis_type == "x":
                        ax.set_xlim(xmin=ax_min)
                    elif axis_type == "y":
                        ax.set_ylim(ymin=ax_min)
                if sax.max is not None:
                    if axis_type == "x":
                        ax.set_xlim(xmax=ax_max)
                    elif axis_type == "y":
                        ax.set_ylim(ymax=ax_max)

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
                    if axis_type == "x":
                        ax.set_xticklabels([])  # hide ticks
                    elif axis_type == "y":
                        ax.set_yticklabels([])  # hide ticks

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
                                    ax.spines[axis].set_color(Figure.fig_facecolor)
                                else:
                                    ax.spines[axis].set_linewidth(linewidth)
                                    ax.tick_params(width=linewidth)

                        if style.line.color:
                            color = style.line.color
                            for axis in directions:
                                ax.spines[axis].set_color(str(color))

                        if style.line.type and style.line.type == LineType.NONE:
                            for axis in directions:
                                ax.spines[axis].set_color(Figure.fig_facecolor)

            if xax:
                apply_axis_settings(xax, ax1, axis_type="x")
            if yax:
                apply_axis_settings(yax, ax1, axis_type="y")
            if yax_right:
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

            # hide none-existing axes
            if plot.xaxis is None:
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax1.xaxis.set_visible(False)

            if plot.yaxis is None:
                ax.spines["top"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax1.yaxis.set_visible(False)

            xgrid = xax.grid if xax else None
            ygrid = yax.grid if yax else None

            if xgrid and ygrid:
                ax1.grid(True, axis="both")
            elif xgrid:
                ax1.grid(True, axis="x")
            elif ygrid:
                ax1.grid(True, axis="y")
            else:
                ax1.grid(False)

            if plot.legend:
                if len(axes) == 1:
                    handles1, _ = ax1.get_legend_handles_labels()
                    if handles1:
                        if figure.legend_position == "inside":
                            ax1.legend(
                                fontsize=Figure.legend_fontsize, loc=Figure.legend_loc
                            )
                        elif figure.legend_position == "outside":
                            ax1.legend(
                                fontsize=Figure.legend_fontsize,
                                loc="upper left",
                                bbox_to_anchor=(1.04, 1),
                            )
                elif len(axes) == 2:
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
