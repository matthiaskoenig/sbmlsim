"""Plotting in sbmlsim."""

from sbmlsim.units import setup_matplotlib

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

# the quantities of the package plot with matplotlib in their units
setup_matplotlib()
