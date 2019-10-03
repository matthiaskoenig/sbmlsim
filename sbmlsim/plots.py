from matplotlib import pyplot as plt
from sbmlsim.simulation import TimecourseResult

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


def add_line(ax, data, yid, xid="time", color='black', label='', kwargs_sim=kwargs_sim,
             xf=1.0, **kwargs):
    """
    :param ax:
    :param xid:
    :param yid:

    :param color:
    :return:
    """
    kwargs_plot = dict(kwargs_sim)
    kwargs_plot.update(kwargs)


    if isinstance(data, TimecourseResult):
        x = data.mean[xid]*xf

        ax.fill_between(x, data.min[yid], data.mean[yid] - data.std[yid], color=color, alpha=0.3, label="__nolabel__")
        ax.fill_between(x, data.mean[yid] + data.std[yid], data.max[yid], color=color, alpha=0.3, label="__nolabel__")
        ax.fill_between(x, data.mean[yid] - data.std[yid], data.mean[yid] + data.std[yid], color=color, alpha=0.5, label="__nolabel__")

        ax.plot(x, data.mean[yid], '-', color=color, label="sim {}".format(label), **kwargs_plot)
    else:
        x = s[xid] * xf
        ax.plot(x, s[yid], '-', color=color, label="sim {}".format(label), **kwargs_plot)