"""Plotting in sbmlsim.

The figures are drawn by the renderer of sbmlsim-core (`Chart`); importing
the package imports no matplotlib. `label_axes` labels the axes of a figure
of matplotlib which plots quantities, whose units `ScanResult.q` and
`sbmlsim.units.setup_matplotlib` register with matplotlib.
"""

from .axes import label_axes
from .chart import Chart, ChartAxis, ChartPlot
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
    "label_axes",
]
