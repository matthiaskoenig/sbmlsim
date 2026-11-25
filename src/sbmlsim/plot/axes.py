"""Labels of the axes of matplotlib which plot quantities.

A quantity of a result, `ScanResult.q`, plots with matplotlib in its unit:
the first quantity of an axis sets its unit, every other one is converted
into it, see `sbmlsim.units.setup_matplotlib`. `label_axes` names the axes
and adds their units:

    fig, ax = plt.subplots()
    ax.plot(res.q["time"].to("hr"), res.q["cp"])
    label_axes(ax, "time", "HCTZ plasma")  # "time [h]", "HCTZ plasma [ng/ml]"

`limit_log_axes` limits the automatic log axes of a figure of matplotlib to
the decades a figure of the figure model shows. None of these functions
imports matplotlib, they act on its objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sbmlsim.plot.chart import axis_label as _axis_label
from sbmlsim.plot.plotting import log_limits
from sbmlsim.units import ureg

if TYPE_CHECKING:
    # matplotlib is imported by the code which draws, not by sbmlsim
    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure


def axis_label(name: str, axis: Axis) -> str:
    """Get the label of an axis: its name and the unit of its quantities.

    Args:
        name: the name of what the axis shows.
        axis: the axis of matplotlib.

    Returns:
        `name [unit]` in the short form of the unit, the name for an axis
        without a unit or with a dimensionless one.
    """
    units: Any = axis.get_units()
    if units is None or not isinstance(units, ureg.Unit):
        return name
    return _axis_label(name, units)


def label_axes(ax: Axes, x: str | None = None, y: str | None = None) -> None:
    """Label the axes of a plot of quantities with their names and units.

    Call it after the quantities are plotted, which set the units of the
    axes, e.g. `label_axes(ax, "time", "HCTZ plasma")` gives
    `time [h]` and `HCTZ plasma [ng/ml]`.

    Args:
        ax: the axes.
        x: the name of the x axis, `None` keeps its label.
        y: the name of the y axis, `None` keeps its label.
    """
    if x is not None:
        ax.set_xlabel(axis_label(x, ax.xaxis))
    if y is not None:
        ax.set_ylabel(axis_label(y, ax.yaxis))


def limit_log_axis(
    ax: Axes,
    axis: str = "y",
    low: float | None = None,
    high: float | None = None,
) -> None:
    """Limit a log axis of matplotlib to `LOG_DECADES` below its largest value.

    The limits of the data drawn so far, see `sbmlsim.plot.plotting.log_limits`;
    for a figure drawn with matplotlib itself, after its data and
    `set_yscale("log")`, so the tiny first steps of the integrator do not
    stretch the axis over dozens of decades.

    Args:
        ax: the axes.
        axis: `"x"` or `"y"`.
        low: the lower limit set on the axis, `None` without.
        high: the upper limit set on the axis, `None` without.
    """
    data = ax.dataLim
    smallest, largest = (
        (data.minposx, data.x1) if axis == "x" else (data.minposy, data.y1)
    )
    limits = log_limits(smallest, largest, low, high)
    if limits is None:
        return
    if axis == "x":
        ax.set_xlim(*limits)
    else:
        ax.set_ylim(*limits)


def limit_log_axes(figure: Figure) -> None:
    """Limit every automatic log axis of a figure of matplotlib, see `limit_log_axis`.

    The figures of `SimulationExperiment.figures_mpl()` get it before they are
    saved; a log axis whose limits the figure set itself (autoscale off) is
    kept.

    Args:
        figure: the figure.
    """
    for ax in figure.axes:
        if ax.get_xscale() == "log" and ax.get_autoscalex_on():
            limit_log_axis(ax, "x")
        if ax.get_yscale() == "log" and ax.get_autoscaley_on():
            limit_log_axis(ax, "y")
