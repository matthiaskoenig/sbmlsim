"""Deprecated matplotlib functions.

These functions will be removed in future releases.
"""
import itertools

import pandas as pd
from matplotlib import pyplot as plt
from pymetadata import log

from sbmlsim.data import DataSet
from sbmlsim.result import XResult
from sbmlsim.utils import deprecated


logger = log.get_logger(__name__)

kwargs_data = {"marker": "s", "linestyle": "--", "linewidth": 1, "capsize": 3}
kwargs_sim = {"marker": None, "linestyle": "-", "linewidth": 2}


@deprecated
def add_line(
    ax: plt.Axes,
    xres: XResult,
    xid: str,
    yid: str,
    xunit,
    yunit,
    yres: XResult = None,
    xf=1.0,
    yf=1.0,
    all_lines=False,
    label="__nolabel__",
    **kwargs,
):
    """Adding information from a simulation result to a matplotlib figure.

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
        raise ValueError(
            f"Only XResult supported in plotting, but found: " f"'{type(xres)}'"
        )

    # mean data with units
    x = xres.dim_mean(xid)
    y = xres.dim_mean(yid) * yf

    # reduction over all dimensions (not necessarily what is wanted !)
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

    # get next color
    prop_cycler = ax._get_lines.prop_cycler
    color = kwargs.get("color", next(prop_cycler)["color"])
    kwargs["color"] = color

    if all_lines:
        Q_ = xres.uinfo.Q_
        # iterate over all dimensions besides time
        # all combinations
        dims = xres._redop_dims()
        index_vecs = [xres.coords[dim].values for dim in dims]
        indices = list(itertools.product(*index_vecs))
        for k, item in enumerate(indices):
            d = dict(zip(dims, item))
            xi = Q_(xres[xid].isel(d).values, xres.uinfo[xid])
            yi = Q_(xres[yid].isel(d).values, xres.uinfo[yid])
            # FIXME: these conversions should not be necessary
            if xunit:
                xi = xi.to(xunit)
            if yunit:
                yi = yi.to(yunit)
            if k != 0:
                label = "__nolabel__"
            ax.plot(xi.magnitude, yi.magnitude, color=color, label=label)

    else:
        # calculate rational ysd, i.e., if the value if y + ysd is larger than ymax take ymax
        ysd_up = y + ystd
        ysd_up[ysd_up > ymax] = ymax[ysd_up > ymax]
        ysd_down = y - ystd
        ysd_down[ysd_down < ymin] = ymin[ysd_down < ymin]

        ax.fill_between(
            x.magnitude,
            ysd_down.magnitude,
            ysd_up.magnitude,
            color=color,
            alpha=0.4,
            label="__nolabel__",
        )
        ax.fill_between(
            x.magnitude,
            ysd_up.magnitude,
            ymax.magnitude,
            color=color,
            alpha=0.1,
            label="__nolabel__",
        )
        ax.fill_between(
            x.magnitude,
            ysd_down.magnitude,
            ymin.magnitude,
            color=color,
            alpha=0.1,
            label="__nolabel__",
        )

        ax.plot(
            x.magnitude, ysd_up.magnitude, "-", label="__nolabel__", alpha=0.8, **kwargs
        )
        ax.plot(
            x.magnitude,
            ysd_down.magnitude,
            "-",
            label="__nolabel__",
            alpha=0.8,
            **kwargs,
        )
        ax.plot(
            x.magnitude, ymin.magnitude, "-", label="__nolabel__", alpha=0.6, **kwargs
        )
        ax.plot(
            x.magnitude, ymax.magnitude, "-", label="__nolabel__", alpha=0.6, **kwargs
        )

        # curve
        ax.plot(x.magnitude, y.magnitude, "-", label=label, **kwargs)


@deprecated
def add_data(
    ax: plt.Axes,
    data: DataSet,
    xid: str,
    yid: str,
    yid_sd=None,
    yid_se=None,
    count=None,
    xunit=None,
    yunit=None,
    xf=1.0,
    yf=1.0,
    label="__nolabel__",
    **kwargs,
):
    """Add experimental data to a matplotlib axes.

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

    if abs(xf - 1.0) > 1e-8:
        logger.warning("xf attributes are deprecated, use units instead.")
    if abs(yf - 1.0) > 1e-8:
        logger.warning("yf attributes are deprecated, use units instead.")

    # add default styles
    if "marker" not in kwargs:
        kwargs["marker"] = "s"
    if "linestyle" not in kwargs:
        kwargs["linestyle"] = "--"

    # data with units
    x = dset[xid].values * dset.uinfo.ureg(dset.uinfo[xid]) * xf
    y = dset[yid].values * dset.uinfo.ureg(dset.uinfo[yid]) * yf
    y_err = None
    y_err_type = None
    if yid_sd:
        y_err = dset[yid_sd].values * dset.uinfo.ureg(dset.uinfo[yid]) * yf
        y_err_type = "SD"
    elif yid_se:
        y_err = dset[yid_se].values * dset.uinfo.ureg(dset.uinfo[yid]) * yf
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
        if "capsize" not in kwargs:
            kwargs["capsize"] = 3
        ax.errorbar(x.magnitude, y.magnitude, y_err.magnitude, label=label, **kwargs)
    else:
        ax.plot(x, y, label=label, **kwargs)
