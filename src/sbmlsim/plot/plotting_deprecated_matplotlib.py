"""Deprecated matplotlib functions.

These functions will be removed in future releases.
"""

import logging

import pandas as pd
from matplotlib import pyplot as plt

from sbmlsim.data import DataSet
from sbmlsim.utils import deprecated

logger = logging.getLogger(__name__)

kwargs_data = {"marker": "s", "linestyle": "--", "linewidth": 1, "capsize": 3}
kwargs_sim = {"marker": None, "linestyle": "-", "linewidth": 2}


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
        logger.error("Empty dataset in adding data: %s", dset)

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
