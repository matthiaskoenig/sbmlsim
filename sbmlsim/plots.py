def add_line(xid, yid, ax, s, color='black', label='', xf=1.0, kwargs_sim=kwargs_sim, **kwargs):
    """

    :param xid:
    :param yid:
    :param ax:
    :param s: namedtuple Result from simulate
    :param color:
    :return:
    """
    kwargs_plot = dict(kwargs_sim)
    kwargs_plot.update(kwargs)

    if isinstance(s, Result):
        x = s.mean[xid]*xf

        ax.fill_between(x, s.min[yid], s.mean[yid] - s.std[yid], color=color, alpha=0.3, label="__nolabel__")
        ax.fill_between(x, s.mean[yid] + s.std[yid], s.max[yid], color=color, alpha=0.3, label="__nolabel__")
        ax.fill_between(x, s.mean[yid] - s.std[yid], s.mean[yid] + s.std[yid], color=color, alpha=0.5, label="__nolabel__")

        ax.plot(x, s.mean[yid], '-', color=color, label="sim {}".format(label), **kwargs_plot)
    else:
        x = s[xid] * xf
        ax.plot(x, s[yid], '-', color=color, label="sim {}".format(label), **kwargs_plot)