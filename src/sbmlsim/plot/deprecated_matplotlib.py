"""Serialization of Figure object to matplotlib."""

from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure as FigureMPL
from matplotlib.gridspec import GridSpec

from sbmlsim.data import Data
from sbmlsim.plot import Figure, Axis, SubPlot, Curve
from sbmlsim.plot.plotting import AxisScale, Style, LineStyle
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

            # plot curves ordered by order
            curves: List[Curve] = sorted(plot.curves, key=lambda x: x.order)
            for curve in curves:
                print(curve)
                label = curve.name if curve.name else "__nolabel__"

                kwargs = {}
                if curve.style:
                    kwargs = curve.style.to_mpl_kwargs()

                # mean quantity
                x = curve.x.get_data(experiment=experiment, to_units=xunit)
                y = curve.y.get_data(experiment=experiment, to_units=yunit)

                # additional data exists in xres
                # FIXME: get access to the full data matrix and support plotting
                # FIXME: this does not plot the single lines !!! (same problem with full data access)
                # if isinstance(x, XResult):
                #    x_mean = x.mean(dim=self._op_dims(), skipna=True).values * self.ureg(self.udict[key])

                # FIXME: clean separation in data preprocessing and actual plotting
                # FIXME: this should be done via additional areas outside of plotting
                x_std, x_min, x_max = None, None, None
                if curve.x.dtype == Data.Types.TASK:
                    data = curve.x
                    xres = experiment.results[data.task_id]  # type: XResult
                    if not xres.is_timecourse():
                        x_std = xres.dim_std(data.index).to(xunit)
                        x_min = xres.dim_min(data.index).to(xunit)
                        x_max = xres.dim_max(data.index).to(xunit)

                y_std, y_min, y_max = None, None, None
                if curve.y.dtype == Data.Types.TASK:
                    data = curve.y
                    xres = experiment.results[data.task_id]  # type: XResult
                    if not xres.is_timecourse():
                        y_std = xres.dim_std(data.index).to(yunit)
                        y_min = xres.dim_min(data.index).to(yunit)
                        y_max = xres.dim_max(data.index).to(yunit)

                xerr = None
                if curve.xerr is not None:
                    xerr = curve.xerr.get_data(experiment=experiment, to_units=xunit)

                yerr = None
                if curve.yerr is not None:
                    yerr = curve.yerr.get_data(experiment=experiment, to_units=yunit)

                xmin, xmax = x.magnitude[0], x.magnitude[-1]
                if (xax.min is not None) and (xax.min > xmin):
                    xmin = xax.min
                if (xax.max is not None) and (xax.max < xmax):
                    xmax = xax.max
                x_ip = np.linspace(
                    start=xmin, stop=xmax, num=Figure._area_interpolation_points
                )

                if (y_min is not None) and (y_max is not None):
                    # areas can be very slow to render!
                    # ax.fill_between(x.magnitude, y_min.magnitude, y_max.magnitude,
                    ax.fill_between(
                        x_ip,
                        interp(x_ip, x.magnitude, y_min.magnitude),
                        interp(x_ip, x.magnitude, y_max.magnitude),
                        color=kwargs.get("color", "black"),
                        alpha=0.1,
                        label="__nolabel__",
                    )

                if y_std is not None:
                    # areas can be very slow to render!
                    # ax.plot(x.magnitude, y.magnitude, label="curve.name", **kwargs)

                    if not np.all(np.isfinite(y_std)):
                        logger.error(f"Not all values finite in y_std: {y_std}")

                    y_std_up = interp(
                        x=x_ip, xp=x.magnitude, fp=y.magnitude + y_std.magnitude
                    )
                    y_std_down = interp(
                        x=x_ip, xp=x.magnitude, fp=y.magnitude - y_std.magnitude
                    )

                    # ax.fill_between(x.magnitude, y.magnitude - y_std.magnitude, y.magnitude + y_std.magnitude,
                    ax.fill_between(
                        x_ip,
                        y_std_down,
                        y_std_up,
                        color=kwargs.get("color", "black"),
                        alpha=0.25,
                        label="__nolabel__",
                    )

                if (xerr is None) and (yerr is None):
                    if y_std is None:
                        # single trajectory
                        ax.plot(x.magnitude, y.magnitude, label=label, **kwargs)
                elif yerr is not None:
                    ax.errorbar(
                        x.magnitude,
                        y.magnitude,
                        yerr.magnitude,
                        label=label,
                        **kwargs,
                    )

            # plot settings
            if plot.name and plot.title_visible:
                ax.set_title(plot.name)

            def unit_str(ax: Axis):
                return ax.unit if len(str(ax.unit)) > 0 else "-"

            if xax:
                if (xax.min is not None) or (xax.max is not None):
                    # (None, None) locks the axis limits to defaults [0,1]
                    ax.set_xlim(xmin=xax.min, xmax=xax.max)

                ax.set_xscale(get_scale(xax))

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

                ax.set_yscale(get_scale(yax))

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
