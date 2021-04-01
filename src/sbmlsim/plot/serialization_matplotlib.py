"""Serialization of Figure object to matplotlib."""

from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure as FigureMPL
from matplotlib.gridspec import GridSpec

from sbmlsim.data import Data
from sbmlsim.plot import Figure, Axis, SubPlot, Curve
from sbmlsim.plot.plotting import AxisScale, Style, LineStyle, CurveType
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

            ridx = subplot.row - 1
            cidx = subplot.col - 1
            ax: plt.Axes = fig.add_subplot(
                gs[ridx : ridx + subplot.row_span, cidx : cidx + subplot.col_span]
            )

            plot = subplot.plot
            xax: Axis = plot.xaxis
            yax: Axis = plot.yaxis
            print(plot.__repr__())

            # units
            if xax is None:
                logger.warning(f"No xaxis in plot: {subplot}")
                ax.spines["bottom"].set_color(Figure.fig_facecolor)
                ax.spines["top"].set_color(Figure.fig_facecolor)
            if yax is None:
                logger.warning(f"No yaxis in plot: {subplot}")
                ax.spines["right"].set_color(Figure.fig_facecolor)
                ax.spines["left"].set_color(Figure.fig_facecolor)
            if (not xax) or (not yax):
                if len(plot.curves) > 0:
                    raise ValueError(
                        f"xaxis and yaxis are required for plotting curves, but "
                        f"'xaxis={xax}' and 'yaxis={yax}'."
                    )

            xunit = xax.unit if xax else None
            yunit = yax.unit if yax else None

            # memory for stacked bars
            barstack_x = None
            barstack_y = None
            barhstack_x = None
            barhstack_y = None

            # plot ordered curves
            curves: List[Curve] = sorted(plot.curves, key=lambda x: x.order)
            print("order:", [c.order for c in curves])
            for curve in curves:
                print(curve)

                # process data
                x = curve.x.get_data(experiment=experiment, to_units=xunit)
                y = curve.y.get_data(experiment=experiment, to_units=yunit)
                xerr = None
                if curve.xerr is not None:
                    xerr = curve.xerr.get_data(experiment=experiment, to_units=xunit)
                yerr = None
                if curve.yerr is not None:
                    yerr = curve.yerr.get_data(experiment=experiment, to_units=yunit)

                label = curve.name if curve.name else "__nolabel__"

                points_kwargs = curve.style.to_mpl_kwargs() if curve.style else {}
                bar_kwargs = {**points_kwargs}
                for key in ["marker", "markersize", "markerfacecolor", "markeredgewidth"]:
                    if key in bar_kwargs:
                        bar_kwargs.pop(key)
                    if 'markeredgecolor' in bar_kwargs:
                        bar_kwargs['edgecolor'] = bar_kwargs.pop('markeredgecolor')


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

                # directly plot the bars with whiskers
                error_kwargs = {
                    'error_kw': {
                        'ecolor': "black",
                        'elinewidth': 3.0,
                    }

                }
                if curve.type == CurveType.POINTS:
                    ax.errorbar(
                        x=x_data,
                        y=y_data,
                        xerr=xerr_data,
                        yerr=yerr_data,
                        label=label,
                        **error_kwargs['error_kw'], **points_kwargs
                    )

                elif curve.type == CurveType.BAR:
                    ax.bar(
                        x=x_data,
                        height=y_data,
                        xerr=xerr_data,
                        yerr=yerr_data,
                        label=label,
                        **error_kwargs,
                        **bar_kwargs,
                    )

                elif curve.type == CurveType.HORIZONTALBAR:
                    ax.barh(
                        y=x_data,
                        width=y_data,
                        xerr=xerr_data,
                        yerr=yerr_data,
                        label=label,
                        **error_kwargs,
                        **bar_kwargs
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
                        **error_kwargs,
                        **bar_kwargs
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
                        **error_kwargs,
                        **bar_kwargs
                    )
                    barhstack_y = barhstack_y + y_data


            # plot settings
            if plot.name and plot.title_visible:
                ax.set_title(plot.name)

            if xax:
                if (xax.min is not None) or (xax.max is not None):
                    # (None, None) locks the axis limits to defaults [0,1]
                    ax.set_xlim(xmin=xax.min, xmax=xax.max)

                ax.set_xscale(cls._get_scale(xax))

                if xax.label_visible:
                    if xax.name:
                        ax.set_xlabel(xax.name)
                if not xax.ticks_visible:
                    ax.set_xticklabels([])  # hide ticks

                # style
                # https://matplotlib.org/stable/api/spines_api.html
                # http://matplotlib.org/examples/pylab_examples/multiple_yaxis_with_spines.html
                if xax.style and xax.style.line:
                    style: Style = xax.style
                    if style.line:
                        if style.line.thickness:
                            linewidth = style.line.thickness
                            for axis in ["bottom", "top"]:
                                ax.tick_params(width=linewidth)
                                if np.isclose(linewidth, 0.0):
                                    ax.spines[axis].set_color(Figure.fig_facecolor)
                                else:
                                    ax.spines[axis].set_linewidth(linewidth)
                                    ax.tick_params(width=linewidth)
                        if style.line.color:
                            color = style.line.color
                            for axis in ["bottom", "top"]:
                                ax.spines[axis].set_color(str(color))

                        if style.line.style and style.line.style == LineStyle.NONE:
                            for axis in ["bottom", "top"]:
                                ax.tick_params(width=linewidth)
                                ax.spines[axis].set_color(Figure.fig_facecolor)

            if yax:
                if (yax.min is not None) or (yax.max is not None):
                    # (None, None) locks the axis limits to defaults [0,1]
                    ax.set_ylim(ymin=yax.min, ymax=yax.max)

                ax.set_yscale(cls._get_scale(yax))

                if yax.label_visible:
                    if yax.name:
                        ax.set_ylabel(yax.name)
                if not yax.ticks_visible:
                    ax.set_yticklabels([])  # hide ticks

                if yax.style and yax.style.line:
                    style: Style = yax.style
                    if style.line:
                        if style.line.thickness:
                            linewidth = style.line.thickness
                            for axis in ["left", "right"]:
                                ax.tick_params(width=linewidth)
                                if np.isclose(linewidth, 0.0):
                                    ax.spines[axis].set_color(Figure.fig_facecolor)
                                else:
                                    ax.spines[axis].set_linewidth(linewidth)
                                    ax.tick_params(width=linewidth)
                        if style.line.color:
                            color = style.line.color
                            for axis in ["left", "right"]:
                                ax.spines[axis].set_color(str(color))

                        if style.line.style and style.line.style == LineStyle.NONE:
                            for axis in ["left", "right"]:
                                ax.tick_params(width=linewidth)
                                ax.spines[axis].set_color(Figure.fig_facecolor)

            # recompute the ax.dataLim
            # ax.relim()
            # update ax.viewLim using the new dataLim
            # ax.autoscale_view()

            # figure styling
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
                ax.tick_params(axis="x", colors=Figure.fig_facecolor)
                ax.xaxis.label.set_color(Figure.fig_facecolor)
            if yax is None:
                ax.tick_params(axis="y", colors=Figure.fig_facecolor)
                ax.yaxis.label.set_color(Figure.fig_facecolor)

            xgrid = xax.grid if xax else None
            ygrid = yax.grid if yax else None

            if xgrid and ygrid:
                ax.grid(True, axis="both")
            elif xgrid:
                ax.grid(True, axis="x")
            elif ygrid:
                ax.grid(True, axis="y")
            else:
                ax.grid(False)

            if plot.legend:
                ax.legend(fontsize=Figure.legend_fontsize, loc=Figure.legend_loc)

        fig.subplots_adjust(
            wspace=Figure.fig_subplots_wspace, hspace=Figure.fig_subplots_hspace
        )

        return fig
