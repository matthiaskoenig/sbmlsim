"""Serialization of Figure object to matplotlib."""

from typing import List, Dict, Any, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure as FigureMPL
from matplotlib.gridspec import GridSpec

from sbmlsim.data import Data
from sbmlsim.plot import Figure, Axis, SubPlot, Curve
from sbmlsim.plot.plotting import AxisScale, Style, LineStyle, CurveType, YAxisPosition
from sbmlsim.plot.plotting_matplotlib import logger, interp


class MatplotlibFigureSerializer(object):
    """Serializer for figures to matplotlib."""

    @classmethod
    def _get_scale(cls, axis: Axis) -> str:
        """Gets string representation of the scale."""
        if axis.scale == AxisScale.LINEAR:
            return "linear"
        elif axis.scale == AxisScale.LOG10:
            return "log"
        else:
            raise ValueError(f"Unsupported axis scale: '{axis.scale}'")

    @classmethod
    def to_figure(cls, experiment: 'SimulationExperiment', figure: Figure) -> FigureMPL:
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
        gs = GridSpec(figure.num_rows, figure.num_cols, figure=fig)

        subplot: SubPlot
        for subplot in figure.subplots:
            plot = subplot.plot
            xax: Axis = plot.xaxis if plot.xaxis else Axis()
            yax: Axis = plot.yaxis if plot.yaxis else Axis()
            yax_right = plot.yaxis_right
            print(plot.__repr__())

            ridx = subplot.row - 1
            cidx = subplot.col - 1
            ax1: plt.Axes = fig.add_subplot(
                gs[ridx : ridx + subplot.row_span, cidx : cidx + subplot.col_span]
            )
            # secondary axis
            ax2: Optional[plt.Axes] = None
            axes: List[plt.Axes] = [ax1]
            if yax_right:
                for curve in plot.curves:
                    if curve.yaxis_position and curve.yaxis_position == YAxisPosition.RIGHT:
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
            yunit = yax.unit if yax else None
            yunit_right = yax_right.unit if yax_right else None

            # memory for stacked bars
            barstack_x = None
            barstack_y = None
            barhstack_x = None
            barhstack_y = None

            # plot ordered curves
            curves: List[Curve] = sorted(plot.curves, key=lambda x: x.order)
            for kc, curve in enumerate(curves):

                print(curve)
                if curve.yaxis_position and curve.yaxis_position == YAxisPosition.RIGHT:
                    # right axis
                    yaxis_position = YAxisPosition.RIGHT
                    ax = ax2
                else:
                    # left axis
                    yaxis_position = YAxisPosition.LEFT
                    ax = ax1

                # process data
                x = curve.x.get_data(experiment=experiment, to_units=xunit)
                if yaxis_position == YAxisPosition.LEFT:
                    y = curve.y.get_data(experiment=experiment, to_units=yunit)
                else:
                    y = curve.y.get_data(experiment=experiment, to_units=yunit_right)
                xerr = None
                if curve.xerr is not None:
                    xerr = curve.xerr.get_data(experiment=experiment, to_units=xunit)
                yerr = None
                if curve.yerr is not None:
                    if yaxis_position == YAxisPosition.LEFT:
                        yerr = curve.yerr.get_data(experiment=experiment, to_units=yunit)
                    else:
                        yerr = curve.yerr.get_data(experiment=experiment,
                                                   to_units=yunit_right)

                label = curve.name if curve.name else "__nolabel__"

                # FIXME: necessary to get the individual curves out of the data cube
                # TODO: iterate over all repeats in the data
                x_data = x.magnitude[:, 0] if x is not None else None
                y_data = y.magnitude[:, 0] if y is not None else None
                xerr_data = xerr.magnitude[:, 0] if xerr is not None else None
                yerr_data = yerr.magnitude[:, 0] if yerr is not None else None

                # print("xshape")
                # print("x", x)
                # print("y", y)
                # print("x_data", x_data)
                # print("y_data", y_data)

                kwargs: Dict[str, Any] = {}
                if curve.style:
                    if curve.type == CurveType.POINTS:
                        kwargs = curve.style.to_mpl_points_kwargs()
                    else:
                        # bar plot
                        kwargs = curve.style.to_mpl_bar_kwargs()

                if curve.type == CurveType.POINTS:
                    ax.errorbar(
                        x=x_data,
                        y=y_data,
                        xerr=xerr_data,
                        yerr=yerr_data,
                        label=label,
                        **kwargs
                    )

                elif curve.type == CurveType.BAR:
                    ax.bar(
                        x=x_data,
                        height=y_data,
                        xerr=xerr_data,
                        yerr=yerr_data,
                        label=label,
                        **kwargs
                    )

                elif curve.type == CurveType.HORIZONTALBAR:
                    ax.barh(
                        y=x_data,
                        width=y_data,
                        xerr=xerr_data,
                        yerr=yerr_data,
                        label=label,
                        **kwargs
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
                        **kwargs
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
                        **kwargs
                    )
                    barhstack_y = barhstack_y + y_data

            # plot settings
            if plot.name and plot.title_visible:
                ax1.set_title(plot.name)

            if xax:
                if xax.min is not None:
                    ax1.set_xlim(xmin=xax.min)
                if xax.max is not None:
                    ax1.set_xlim(xmax=xax.max)
                ax1.set_xscale(cls._get_scale(xax))

                if xax.label_visible:
                    if xax.name:
                        ax1.set_xlabel(xax.name)
                if not xax.ticks_visible:
                    ax1.set_xticklabels([])  # hide ticks

                # style
                # https://matplotlib.org/stable/api/spines_api.html
                # http://matplotlib.org/examples/pylab_examples/multiple_yaxis_with_spines.html
                if xax.style and xax.style.line:
                    style: Style = xax.style
                    if style.line:
                        if style.line.thickness:
                            linewidth = style.line.thickness
                            for axis in ["bottom", "top"]:
                                ax1.tick_params(width=linewidth)
                                if np.isclose(linewidth, 0.0):
                                    ax1.spines[axis].set_color(Figure.fig_facecolor)
                                else:
                                    ax1.spines[axis].set_linewidth(linewidth)
                                    ax1.tick_params(width=linewidth)
                        if style.line.color:
                            color = style.line.color
                            for axis in ["bottom", "top"]:
                                ax1.spines[axis].set_color(str(color))

                        if style.line.style and style.line.style == LineStyle.NONE:
                            for axis in ["bottom", "top"]:
                                ax1.tick_params(width=linewidth)
                                ax1.spines[axis].set_color(Figure.fig_facecolor)

            if yax:
                if yax.min is not None:
                    ax1.set_ylim(ymin=yax.min)
                if yax.max is not None:
                    ax1.set_ylim(ymax=yax.max)

                ax1.set_yscale(cls._get_scale(yax))

                if yax.label_visible:
                    if yax.name:
                        ax1.set_ylabel(yax.name)
                if not yax.ticks_visible:
                    ax1.set_yticklabels([])  # hide ticks

                if yax.style and yax.style.line:
                    style: Style = yax.style
                    if style.line:
                        if style.line.thickness:
                            linewidth = style.line.thickness
                            for axis in ["left", "right"]:
                                ax1.tick_params(width=linewidth)
                                if np.isclose(linewidth, 0.0):
                                    ax1.spines[axis].set_color(Figure.fig_facecolor)
                                else:
                                    ax1.spines[axis].set_linewidth(linewidth)
                                    ax1.tick_params(width=linewidth)
                        if style.line.color:
                            color = style.line.color
                            for axis in ["left", "right"]:
                                ax1.spines[axis].set_color(str(color))

                        if style.line.style and style.line.style == LineStyle.NONE:
                            for axis in ["left", "right"]:
                                ax1.tick_params(width=linewidth)
                                ax1.spines[axis].set_color(Figure.fig_facecolor)

            if yax_right:
                if yax_right.min is not None:
                    ax2.set_ylim(ymin=yax_right.min)
                if yax_right.max is not None:
                    ax2.set_ylim(ymax=yax_right.max)

                ax2.set_yscale(cls._get_scale(yax_right))

                if yax_right.label_visible:
                    if yax_right.name:
                        ax2.set_ylabel(yax_right.name)
                if not yax_right.ticks_visible:
                    ax2.set_yticklabels([])  # hide ticks

                if yax_right.style and yax_right.style.line:
                    style: Style = yax_right.style
                    if style.line:
                        if style.line.thickness:
                            linewidth = style.line.thickness
                            for axis in ["left", "right"]:
                                ax2.tick_params(width=linewidth)
                                if np.isclose(linewidth, 0.0):
                                    ax2.spines[axis].set_color(Figure.fig_facecolor)
                                else:
                                    ax2.spines[axis].set_linewidth(linewidth)
                                    ax2.tick_params(width=linewidth)
                        if style.line.color:
                            color = style.line.color
                            for axis in ["left", "right"]:
                                ax2.spines[axis].set_color(str(color))

                        if style.line.style and style.line.style == LineStyle.NONE:
                            for axis in ["left", "right"]:
                                ax2.tick_params(width=linewidth)
                                ax2.spines[axis].set_color(Figure.fig_facecolor)


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
            if xax is None:
                ax1.tick_params(axis="x", colors=Figure.fig_facecolor)
                ax1.xaxis.label.set_color(Figure.fig_facecolor)
            if yax is None:
                ax1.tick_params(axis="y", colors=Figure.fig_facecolor)
                ax1.yaxis.label.set_color(Figure.fig_facecolor)

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
                    ax1.legend(fontsize=Figure.legend_fontsize, loc=Figure.legend_loc)
                elif len(axes) == 2:
                    ax1.legend(fontsize=Figure.legend_fontsize, loc="upper left")
                    ax2.legend(fontsize=Figure.legend_fontsize, loc="upper right")

        fig.subplots_adjust(
            wspace=Figure.fig_subplots_wspace, hspace=Figure.fig_subplots_hspace
        )

        return fig
