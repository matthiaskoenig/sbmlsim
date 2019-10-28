from matplotlib import pyplot as plt
from sbmlsim.simulation_serial import Result
import pandas as pd

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


def add_line(ax, data: Result, xid: str, yid: str, xf: float = 1.0, yf: float=1.0,
             color='black', label='', kwargs_sim=kwargs_sim, **kwargs):
    """
    :param ax: axis to plot to
    :param data: Result data structure
    :param xid: id for xdata
    :param yid: id for ydata
    :param xf: scaling factor for x
    :param yf: scaling factor for y

    :param color:
    :return:
    """
    kwargs_plot = dict(kwargs_sim)
    kwargs_plot.update(kwargs)

    if not isinstance(data, Result):
        raise ValueError("Only Result objects supported.")

    x = data.mean[xid] * xf
    y = data.mean[yid] * yf
    y_sd = data.std[yid] * yf
    y_min = data.min[yid] * yf
    y_max = data.max[yid] * yf

    if len(data) > 1:
        # FIXME: std areas should be within min/max areas!
        ax.fill_between(x, y - y_sd, y + y_sd, color=color, alpha=0.4, label="__nolabel__")

        ax.fill_between(x, y + y_sd, y_max, color=color, alpha=0.2, label="__nolabel__")
        ax.fill_between(x, y - y_sd, y_min, color=color, alpha=0.2, label="__nolabel__")

    ax.plot(x, y, '-', color=color, label="sim {}".format(label), **kwargs_plot)
