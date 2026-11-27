"""The chart builder of the figure contract `figure/1`, `sbmlsim_core.chart`.

The builder is the python module `sbmlsim_core.chart` of sbmlsim-core, so a
package which sbmlsim depends on draws charts without sbmlsim; this module
re-exports it under its name in sbmlsim. A `Chart` is a figure of plots in a
grid, `ChartPlot`s with the axes `ChartAxis`, whose methods add the elements of
the contract; the data are sequences, numpy arrays, quantities of pint (the
registry of sbmlsim among them, which converts its quantities) or data arrays
of xarray (`attrs["units"]` their unit):

    chart = Chart(title="HCTZ")
    plot = chart.plot(x=ChartAxis(label="time", unit="hr"), y=ChartAxis(label="cp"))
    plot.curve(res.q["time"], res.q["cp"], label="plasma")
    chart.save("hctz.svg")

See the page Charts of the documentation of sbmlsim-core for the elements, the
units, the colours and the thinning of the lines.

The names are read from `sbmlsim_core.chart` when they are first used, not when
this module is imported: importing sbmlsim imports no bindings of sbmlsim-core
(`sbmlsim.engine.import_core`), which are imported on a thread which lives as
long as the process before the chart builder imports them.
"""

import importlib
from typing import TYPE_CHECKING, Any

from sbmlsim.engine import import_core

if TYPE_CHECKING:
    from sbmlsim_core.chart import (
        BAND_OPACITY,
        DEFAULT_DPI,
        FIGURE_CONTRACT,
        LEGEND_LOCS,
        MARKERS,
        SAVE_SUFFIXES,
        THIN_TOLERANCE,
        Aspect,
        AxisName,
        Baseline,
        Chart,
        ChartAxis,
        ChartPlot,
        ColorMap,
        Coordinates,
        Extend,
        FillStyle,
        LineType,
        Placement,
        Values,
        axis_label,
        legend_loc_spec,
    )

__all__ = [
    "BAND_OPACITY",
    "DEFAULT_DPI",
    "FIGURE_CONTRACT",
    "LEGEND_LOCS",
    "MARKERS",
    "SAVE_SUFFIXES",
    "THIN_TOLERANCE",
    "Aspect",
    "AxisName",
    "Baseline",
    "Chart",
    "ChartAxis",
    "ChartPlot",
    "ColorMap",
    "Coordinates",
    "Extend",
    "FillStyle",
    "LineType",
    "Placement",
    "Values",
    "axis_label",
    "legend_loc_spec",
]


def __getattr__(name: str) -> Any:
    """Get a name of the chart builder, importing sbmlsim-core on first use."""
    return lazy_attribute(globals(), "sbmlsim_core.chart", __all__, __name__, name)


def __dir__() -> list[str]:
    """Get the names of the module, those of the chart builder included."""
    return sorted({*globals(), *__all__})


def lazy_attribute(
    namespace: dict[str, Any], module: str, names: list[str], name_of: str, name: str
) -> Any:
    """Get a name of a module of sbmlsim-core and keep it in a namespace.

    The bindings of sbmlsim-core are imported through `import_core` first, so
    they are imported on a thread which lives as long as the process.

    Args:
        namespace: the globals of the module which re-exports the name.
        module: the module of sbmlsim-core which defines the name.
        names: the names the module re-exports.
        name_of: the name of the module which re-exports, for the error.
        name: the name.

    Returns:
        The value of the name.

    Raises:
        AttributeError: for a name which is not re-exported.
    """
    if name not in names:
        raise AttributeError(f"module '{name_of}' has no attribute '{name}'")
    import_core()
    value = getattr(importlib.import_module(module), name)
    namespace[name] = value
    return value
