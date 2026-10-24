"""Labels of the axes of matplotlib which plot quantities.

A quantity of a result, `ScanResult.q`, plots with matplotlib in its unit:
the first quantity of an axis sets its unit, every other one is converted
into it, see `sbmlsim.units.setup_matplotlib`. `label_axes` names the axes
and adds their units:

    fig, ax = plt.subplots()
    ax.plot(res.q["time"].to("hr"), res.q["cp"])
    label_axes(ax, "time", "HCTZ plasma")  # "time [h]", "HCTZ plasma [ng/ml]"
"""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.axis import Axis

from sbmlsim.units import short_unit, ureg


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
    text = short_unit(str(units))
    return f"{name} [{text}]" if text else name


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
