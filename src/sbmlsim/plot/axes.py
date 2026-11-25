"""Limits of the log axes of figures of matplotlib.

The figures of the package are drawn by the renderer of sbmlsim-core, see
`sbmlsim.plot.Chart`. `limit_log_axes` limits the automatic log axes of a
figure of matplotlib, e.g. one of `SimulationExperiment.figures_mpl()`, to the
decades a figure of the figure model shows. None of these functions imports
matplotlib, they act on its objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sbmlsim.plot.plotting import log_limits

if TYPE_CHECKING:
    # matplotlib is imported by the code which draws, not by sbmlsim
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


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
