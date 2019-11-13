from matplotlib import pyplot as plt
from sbmlsim.simulation_serial import Result
import pandas as pd

import logging
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
    'figure.facecolor': '1.00'

})


def add_line(ax, data: Result,
             xid: str, yid: str,
             xunit=None, yunit=None, xf=1.0, yf=1.0,
             label='__nolabel__', **kwargs):
    """
    :param ax: axis to plot to
    :param data: Result data structure
    :param xid: id for xdata
    :param yid: id for ydata
    :param xunit: target unit for x (conversion is performed automatically)
    :param yunit: target unit for y (conversion is performed automatically)

    :param color:
    :return:
    """
    if not isinstance(data, Result):
        raise ValueError("Only Result objects supported in plotting.")
    if abs(xf-1.0) > 1E-8:
        logger.warning("xf attributes are deprecated, use units instead.")
    if abs(yf - 1.0) > 1E-8:
        logger.warning("yf attributes are deprecated, use units instead.")

    # data with units
    x = data.mean[xid].values * data.units[xid] * xf
    y = data.mean[yid].values * data.units[yid] * yf
    y_sd = data.std[yid].values * data.units[yid] * yf
    y_min = data.min[yid].values * data.units[yid] * yf
    y_max = data.max[yid].values * data.units[yid] * yf

    # convert
    if xunit:
        x = x.to(xunit)
    if yunit:
        y = y.to(yunit)
        y_sd = y_sd.to(yunit)
        y_min = y_min.to(yunit)
        y_max = y_min.to(yunit)

    # get next color
    prop_cycler = ax._get_lines.prop_cycler
    color = kwargs.get("color", next(prop_cycler)['color'])

    if len(data) > 1:
        # FIXME: std areas should be within min/max areas!
        ax.fill_between(x, y - y_sd, y + y_sd, color=color, alpha=0.4, label="__nolabel__")

        ax.fill_between(x, y + y_sd, y_max, color=color, alpha=0.2, label="__nolabel__")
        ax.fill_between(x, y - y_sd, y_min, color=color, alpha=0.2, label="__nolabel__")

    ax.plot(x, y, '-', color=color, label="{}".format(label), **kwargs)
