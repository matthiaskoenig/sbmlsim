import logging
import pandas as pd
import numpy as np
import xarray as xr
import itertools

from sbmlsim.result import XResult
from sbmlsim.data import DataSet, Data
from sbmlsim.plot import Figure, SubPlot, Plot, Curve, Axis

from matplotlib.pyplot import GridSpec
from matplotlib import pyplot as plt
from sbmlsim.utils import deprecated

logger = logging.getLogger(__name__)

kwargs_data = {'marker': 's', 'linestyle': '--', 'linewidth': 1, 'capsize': 3}
kwargs_sim = {'marker': None, 'linestyle': '-', 'linewidth': 2}

# global settings for plots
plt.rcParams.update({
    'axes.labelsize': 'large',
    'axes.labelweight': 'bold',
    'axes.titlesize': 'medium',
    'axes.titleweight': 'bold',
    'legend.fontsize': 'small',
    'xtick.labelsize': 'large',
    'ytick.labelsize': 'large',
    'figure.facecolor': '1.00',
    'figure.dpi': '72',
})


def to_figure(figure: Figure):
    """Convert sbmlsim.Figure to matplotlib figure."""
    fig = plt.figure(figsize=(figure.width, figure.height))  # type: plt.Figure
    # FIXME: check that the settings are applied
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    gs = GridSpec(figure.num_rows, figure.num_cols, figure=fig)
    # TODO: subplots adjust

    def get_scale(axis):
        if axis.scale == Axis.AxisScale.LINEAR:
            return "linear"
        elif axis.scale == Axis.AxisScale.LOG10:
            return "log"
        else:
            raise ValueError(f"Unsupported axis scale: '{axis.scale}'")

    for subplot in figure.subplots:  # type: SubPlot

        ridx = subplot.row - 1
        cidx = subplot.col - 1
        ax = fig.add_subplot(
            gs[ridx:ridx+subplot.row_span, cidx:cidx+subplot.col_span]
        )  # type: plt.Axes
        # axes labels and legends
        plot = subplot.plot
        if plot.name:
            ax.set_title(plot.name)
        xgrid = False
        if plot.xaxis:
            xax = plot.xaxis  # type: Axis
            ax.set_xscale(get_scale(xax))
            if xax.name:
                ax.set_xlabel(f"{xax.name} [{xax.unit}]")
            if xax.min:
                ax.set_xlim(left=xax.min)
            if xax.max:
                ax.set_xlim(right=xax.max)
            if xax.grid:
                xgrid = True

        ygrid = False
        if plot.yaxis:
            yax = plot.yaxis  # type: Axis
            ax.set_yscale(get_scale(yax))
            if yax.name:
                ax.set_ylabel(f"{yax.name} [{yax.unit}]")
            if yax.min:
                ax.set_ylim(bottom=yax.min)
            if yax.max:
                ax.set_ylim(top=xax.max)
            if yax.grid:
                ygrid = True

        if xgrid and ygrid:
            ax.grid(True, axis="both")
        elif xgrid:
            ax.grid(True, axis="x")
        elif ygrid:
            ax.grid(True, axis="y")
        else:
            ax.grid(False)

        for curve in plot.curves:
            # TODO: sort by order

            kwargs = {}
            if curve.style:
                kwargs = curve.style.to_mpl_kwargs()

            # mean
            x = curve.x.data
            y = curve.y.data

            # additional data exists in xres
            x_std = None
            if curve.x.dtype == Data.Types.TASK:
                data = curve.x
                xres = data.experiment.results[data.task_id]  # type: XResult
                x_std = xres.dim_std(data.index).to(data.unit)
                x_min = xres.dim_min(data.index).to(data.unit)
                x_max = xres.dim_max(data.index).to(data.unit)

            y_std = None
            if curve.y.dtype == Data.Types.TASK:
                data = curve.y
                xres = data.experiment.results[data.task_id]  # type: XResult
                y_std = xres.dim_std(data.index).to(data.unit)
                y_min = xres.dim_min(data.index).to(data.unit)
                y_max = xres.dim_max(data.index).to(data.unit)
            # print(y_std)


            # FIXME: get access to the full data matrix and support plotting
            #if isinstance(x, XResult):
            #    x_mean = x.mean(dim=self._op_dims(), skipna=True).values * self.ureg(self.udict[key])

            yerr = None
            if curve.yerr is not None:
                yerr = curve.yerr.data

            xerr = None
            if curve.xerr is not None:
                xerr = curve.xerr.data

            if (xerr is None) and (yerr is None):
                ax.plot(x.magnitude, y.magnitude, label=curve.name, **kwargs)
            elif yerr is not None:
                ax.errorbar(x.magnitude, y.magnitude, yerr.magnitude,
                            label=curve.name, **kwargs)


            if y_std is not None:
                # ax.plot(x.magnitude, y.magnitude + y_std.magnitude, label="__nolabel__", **kwargs)
                ax.fill_between(x.magnitude, y.magnitude - y_std.magnitude,
                                y.magnitude + y_std.magnitude,
                                alpha=0.4, label="__nolabel__", color="darkblue")

        if plot.legend:
            ax.legend()

    return fig


@deprecated
def add_data(ax: plt.Axes, data: DataSet,
             xid: str, yid: str, yid_sd=None, yid_se=None, count=None,
             xunit=None, yunit=None, xf=1.0, yf=1.0, label='__nolabel__', **kwargs):
    """ Add experimental data to a matplotlib axes.

    This is deprecated the plotting Figure, Plot, Curves, should be used
    instead.

    :param ax:
    :param data:
    :param xid:
    :param yid:
    :param xunit:
    :param yunit:
    :param label:
    :param kwargs:
    :return:
    """
    if isinstance(data, DataSet):
        dset = data
    elif isinstance(data, pd.DataFrame):
        dset = DataSet.from_df(df=data, udict=None, ureg=None)

    if dset.empty:
        logger.error(f"Empty dataset in adding data: {dset}")

    if abs(xf-1.0) > 1E-8:
        logger.warning("xf attributes are deprecated, use units instead.")
    if abs(yf - 1.0) > 1E-8:
        logger.warning("yf attributes are deprecated, use units instead.")

    # add default styles
    if 'marker' not in kwargs:
        kwargs['marker'] = 's'
    if 'linestyle' not in kwargs:
        kwargs['linestyle'] = '--'

    # data with units
    x = dset[xid].values * dset.ureg(dset.udict[xid]) * xf
    y = dset[yid].values * dset.ureg(dset.udict[yid]) * yf
    y_err = None
    y_err_type = None
    if yid_sd:
        y_err = dset[yid_sd].values * dset.ureg(dset.udict[yid]) * yf
        y_err_type = "SD"
    elif yid_se:
        y_err = dset[yid_se].values * dset.ureg(dset.udict[yid]) * yf
        y_err_type = "SE"

    # convert
    if xunit:
        x = x.to(xunit)
    if yunit:
        y = y.to(yunit)
        if y_err is not None:
            y_err = y_err.to(yunit)

    # labels
    if label != "__nolabel__":
        if y_err_type:
            label = f"{label} ± {y_err_type}"
        if count:
            label += f" (n={count})"

    # plot
    if y_err is not None:
        if 'capsize' not in kwargs:
            kwargs['capsize'] = 3
        ax.errorbar(x.magnitude, y.magnitude, y_err.magnitude, label=label, **kwargs)
    else:
        ax.plot(x, y, label=label, **kwargs)


@deprecated
def add_line(ax: plt.Axes, xres: XResult,
             xid: str, yid: str,
             xunit=None, yunit=None, xf=1.0, yf=1.0, all_lines=False,
             label='__nolabel__', **kwargs):
    """ Adding information from a simulation result to a matplotlib figure.

    This is deprecated the sbmlsim.plot.plotting Figure, Plot, Curves, ...
    should be used.

    :param ax: axis to plot to
    :param xres: Result data structure
    :param xid: id for xdata
    :param yid: id for ydata
    :param all_lines: plot all individual lines
    :param xunit: target unit for x
    :param yunit: target unit for y

    :param color:
    :return:
    """
    if not isinstance(xres, XResult):
        raise ValueError(f"Only XResult supported in plotting, but found: "
                         f"'{type(xres)}'")


    # data with units
    x = xres.dim_mean(xid)


    y = xres.dim_mean(yid) * yf
    # print("x", x.units, x)
    # print("y", y.units, y)


    ystd = xres.dim_std(yid) * yf
    ymin = xres.dim_min(yid) * yf
    ymax = xres.dim_max(yid) * yf

    # convert units to requested units
    if xunit:
        x = x.to(xunit)
    if yunit:
        y = y.to(yunit)
        ystd = ystd.to(yunit)
        ymin = ymin.to(yunit)
        ymax = ymax.to(yunit)

    # print("x", x.units, x)
    # print("y", y.units, y)

    # get next color
    prop_cycler = ax._get_lines.prop_cycler
    color = kwargs.get("color", next(prop_cycler)['color'])
    kwargs["color"] = color

    if all_lines:
        Q_ = xres.ureg.Quantity
        # iterate over all dimensions besides time
        # all combinations
        dims = xres._redop_dims()
        index_vecs = [xres.coords[dim].values for dim in dims]
        indices = list(itertools.product(*index_vecs))
        # print("index_vecs:", index_vecs)
        # print(indices)
        for item in indices:
            d = dict(zip(dims, item))
            # print(d)
            # individual timecourses
            xi = Q_(xres[xid].isel(d).values, xres.udict[xid])
            yi = Q_(xres[yid].isel(d).values, xres.udict[yid])
            xi = xi.to(xunit)
            yi = yi.to(yunit)

            ax.plot(xi, yi, color="darkred", linewidth=3)

    # calculate rational ysd, i.e., if the value if y + ysd is larger than ymax take ymax
    ysd_up = y + ystd
    ysd_up[ysd_up > ymax] = ymax[ysd_up > ymax]
    ysd_down = y - ystd
    ysd_down[ysd_down < ymin] = ymin[ysd_down < ymin]

    ax.fill_between(x.magnitude, ysd_down.magnitude, ysd_up.magnitude, color=color, alpha=0.4, label="__nolabel__")
    ax.fill_between(x.magnitude, ysd_up.magnitude, ymax.magnitude, color=color, alpha=0.1, label="__nolabel__")
    ax.fill_between(x.magnitude, ysd_down.magnitude, ymin.magnitude, color=color, alpha=0.1, label="__nolabel__")

    ax.plot(x.magnitude, ysd_up.magnitude, '-', label="__nolabel__", alpha=0.8, **kwargs)
    ax.plot(x.magnitude, ysd_down.magnitude, '-', label="__nolabel__", alpha=0.8, **kwargs)
    ax.plot(x.magnitude, ymin.magnitude, '-', label="__nolabel__", alpha=0.6, **kwargs)
    ax.plot(x.magnitude, ymax.magnitude, '-', label="__nolabel__", alpha=0.6, **kwargs)

    # curve

    ax.plot(x.magnitude, y.magnitude, '-', label=label, **kwargs)

