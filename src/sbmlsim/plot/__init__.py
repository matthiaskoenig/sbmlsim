"""Plotting in sbmlsim.

The figures are drawn by the renderer of sbmlsim-core (`Chart`); importing
the package imports no matplotlib and no bindings of sbmlsim-core, the names of
the chart builder are read on first use (`sbmlsim.plot.chart`).
"""

from typing import TYPE_CHECKING, Any

from .plotting import (
    Axis,
    Band,
    ColorType,
    Curve,
    Figure,
    FigureSettings,
    MarkerType,
    Plot,
    SubPlot,
)

if TYPE_CHECKING:
    from .chart import Chart, ChartAxis, ChartPlot

#: the names of the chart builder, read from `sbmlsim.plot.chart` on first use
_CHART = ("Chart", "ChartAxis", "ChartPlot")

__all__ = [
    "Axis",
    "Band",
    "Chart",
    "ChartAxis",
    "ChartPlot",
    "ColorType",
    "Curve",
    "Figure",
    "FigureSettings",
    "MarkerType",
    "Plot",
    "SubPlot",
]


def __getattr__(name: str) -> Any:
    """Get a name of the chart builder, importing sbmlsim-core on first use."""
    if name in _CHART:
        from . import chart

        value = getattr(chart, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
