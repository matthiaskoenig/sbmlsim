"""The uncertainty analysis: bands of timecourses and distributions of values.

The uncertainty of a prediction is a scan over draws of the uncertain
parameters, see `sbmlsim.simulation.sampling` (`random`, `lhs`,
`fit_parameters`, `profile_parameters`, `fit_repeats`, `population`), run by
`Simulator.run`, and its summary over the dimension of the draws,
`ScanResult.summary(dim, quantiles=[0.05, 0.5, 0.95])`. The functions here
draw it: `plot_bands` a timecourse as a median and a band per label of the
other dimensions, `plot_distribution` a value per simulation as a histogram
or a box per label. They return the figure and never show it.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sbmlsim.result.scan import STATISTIC, TIME, ScanResult


def _figure(ax: Axes | None) -> tuple[Figure, Axes]:
    """Get the figure and the axes to draw into."""
    if ax is not None:
        figure = ax.get_figure()
        if not isinstance(figure, Figure):
            raise ValueError("The axes of a plot must belong to a figure.")
        return figure, ax
    figure = Figure(figsize=(6, 4), layout="constrained")
    return figure, figure.add_subplot()


def _labels(data: Any, skip: set[str]) -> list[dict[str, Any]]:
    """Get every combination of the labels of the other dimensions."""
    dims = [str(d) for d in data.dims if str(d) not in skip]
    return [
        dict(zip(dims, values, strict=True))
        for values in itertools.product(*(data[d].values.tolist() for d in dims))
    ]


def _text(selection: dict[str, Any]) -> str:
    """Get the label of a curve."""
    return ", ".join(f"{d}={v}" for d, v in selection.items())


def plot_bands(
    summary: ScanResult,
    key: str,
    *,
    lower: str = "q0.05",
    center: str = "q0.5",
    upper: str = "q0.95",
    ax: Axes | None = None,
    alpha: float = 0.3,
) -> Figure:
    """Draw a timecourse of a summary as a center line and a band per label.

    Args:
        summary: a summary of a scan, `ScanResult.summary(..., quantiles=...)`.
        key: the timecourse.
        lower: the statistic of the lower bound of the band.
        center: the statistic of the line.
        upper: the statistic of the upper bound of the band.
        ax: the axes to draw into, a new figure without.
        alpha: the opacity of the band.

    Returns:
        The figure.

    Raises:
        ValueError: if the variable is no timecourse of a summary or a
            statistic is missing.
    """
    data = summary[key]
    if STATISTIC not in data.dims or TIME not in data.dims:
        raise ValueError(
            f"'{key}' is no timecourse of a summary; summarize a scan first."
        )
    missing = [
        s for s in (lower, center, upper) if s not in data[STATISTIC].values.tolist()
    ]
    if missing:
        raise ValueError(
            f"The summary has not the statistics {missing}, add their quantiles."
        )
    figure, axes = _figure(ax)
    time = data[TIME].values
    for selection in _labels(data, {STATISTIC, TIME}):
        curve = data.sel(selection)
        lines = axes.plot(
            time, curve.sel({STATISTIC: center}).values, label=_text(selection) or key
        )
        axes.fill_between(
            time,
            curve.sel({STATISTIC: lower}).values,
            curve.sel({STATISTIC: upper}).values,
            color=lines[0].get_color(),
            alpha=alpha,
            linewidth=0,
        )
    units = summary.units
    axes.set_xlabel(f"time [{units.get(TIME, '')}]")
    axes.set_ylabel(f"{key} [{units.get(key, '')}]")
    if len(axes.lines) > 1:
        axes.legend()
    return figure


def plot_distribution(
    result: ScanResult,
    key: str,
    *,
    dim: str,
    kind: str = "hist",
    ax: Axes | None = None,
    bins: int = 30,
) -> Figure:
    """Draw a value per simulation over the draws as a histogram or a box per label.

    Args:
        result: the result of a scan over draws.
        key: a value per simulation, e.g. the parameter of a PK observable.
        dim: the dimension of the draws.
        kind: `hist` or `box`.
        ax: the axes to draw into, a new figure without.
        bins: the bins of a histogram.

    Returns:
        The figure.

    Raises:
        ValueError: if the variable is a timecourse, the dimension is not one
            of it, or the kind is unknown.
    """
    if kind not in ("hist", "box"):
        raise ValueError(
            f"The kind of a distribution plot is 'hist' or 'box', not '{kind}'."
        )
    data = result[key]
    if TIME in data.dims or "_point" in data.dims:
        raise ValueError(
            f"'{key}' is a timecourse; draw a value per simulation, e.g. at(x, t)."
        )
    if dim not in data.dims:
        raise ValueError(f"'{dim}' is no dimension of '{key}': {list(data.dims)}.")
    figure, axes = _figure(ax)
    selections = _labels(data, {dim})
    samples = [np.asarray(data.sel(s).values, dtype=float) for s in selections]
    samples = [s[np.isfinite(s)] for s in samples]
    names = [_text(s) or key for s in selections]
    if kind == "hist":
        for values, name in zip(samples, names, strict=True):
            axes.hist(values, bins=bins, alpha=0.5, label=name)
        axes.set_xlabel(f"{key} [{result.units.get(key, '')}]")
        axes.set_ylabel("count")
        if len(samples) > 1:
            axes.legend()
    else:
        axes.boxplot(samples, tick_labels=names)
        axes.set_ylabel(f"{key} [{result.units.get(key, '')}]")
    return figure
