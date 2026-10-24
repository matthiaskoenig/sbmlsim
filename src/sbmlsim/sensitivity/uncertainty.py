"""The uncertainty analysis: bands of timecourses and distributions of values.

The uncertainty of a prediction is a scan over draws of the uncertain
parameters, see `sbmlsim.simulation.sampling` (`random`, `lhs`,
`fit_parameters`, `profile_parameters`, `fit_repeats`, `population`), run by
`Simulator.run`, and its summary over the dimension of the draws,
`ScanResult.summary(dim, quantiles=[0.05, 0.5, 0.95])`. The functions here
draw it: `plot_bands` a timecourse as a median and a band per label of the
other dimensions, `plot_distribution` a value per simulation as a histogram
or a box per label. They return the figure and never show it; called once per
scenario with one `ax` and a `label` each, they draw the scenarios into one
plot with one legend.

`plot_bands` draws quantities, `ScanResult.q`: the axes have the units of the
first quantities drawn into them and convert every other one, so a band
shares an axis with the lines of `ax.plot(res.q["time"], res.q["cp"])`, and
`time_unit` and `unit` choose the units, e.g. `time_unit="hr"`. The axes are
labelled with `sbmlsim.plot.label_axes`, the units in their short form, e.g.
`ng/ml`; `plot_distribution` converts into `unit` and labels the same way.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sbmlsim.plot.axes import label_axes
from sbmlsim.result.scan import STATISTIC, TIME, ScanResult
from sbmlsim.units import setup_matplotlib, unit_label


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


def _text(selection: dict[str, Any], label: str | None, key: str) -> str:
    """Get the label of a curve.

    Args:
        selection: the labels of the other dimensions of the curve.
        label: the label of the call, a prefix of the labels of the dimensions.
        key: the variable, the label of a curve without labels.
    """
    labels = ", ".join(f"{d}={v}" for d, v in selection.items())
    if label is None:
        return labels or key
    return f"{label}: {labels}" if labels else label


def _legend(axes: Axes, label: str | None) -> None:
    """Draw the legend of every labelled curve of the axes, also of earlier calls.

    A legend needs two curves, or one with the label of a call.
    """
    handles, _ = axes.get_legend_handles_labels()
    if len(handles) > 1 or (handles and label is not None):
        axes.legend()


def plot_bands(
    summary: ScanResult,
    key: str,
    *,
    lower: str = "q0.05",
    center: str = "q0.5",
    upper: str = "q0.95",
    ax: Axes | None = None,
    alpha: float = 0.3,
    label: str | None = None,
    time_unit: str | None = None,
    unit: str | None = None,
    name: str | None = None,
) -> Figure:
    """Draw a timecourse of a summary as a center line and a band per label.

    Every line is labelled with the labels of the other dimensions, e.g.
    `dose=1`, or with the key for a summary without other dimensions. A
    `label` is the label of the line of a summary without other dimensions
    and the prefix of the labels of the lines otherwise, e.g. `mid: dose=1`,
    so the summaries of several scenarios drawn into one `ax` are told apart
    in its legend, which every call draws again with every line of the axes.

    Args:
        summary: a summary of a scan, `ScanResult.summary(..., quantiles=...)`.
        key: the timecourse.
        lower: the statistic of the lower bound of the band.
        center: the statistic of the line.
        upper: the statistic of the upper bound of the band.
        ax: the axes to draw into, a new figure without.
        alpha: the opacity of the band.
        label: the label of the lines, see above.
        time_unit: the unit of the time, e.g. `hr`, the one of the summary by
            default; an axis which has a unit keeps it.
        unit: the unit of the values, e.g. `ng/ml`, the one of the summary
            by default; an axis which has a unit keeps it.
        name: the name of the values on their axis, the key by default.

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
    setup_matplotlib()
    figure, axes = _figure(ax)
    time: Any = summary.q[TIME]
    if time_unit is not None:
        time = time.to(time_unit)
    for selection in _labels(data, {STATISTIC, TIME}):
        curve = summary.sel(**selection)
        # quantities, which the stubs of matplotlib do not know as data
        bounds: list[Any] = [
            curve.sel(**{STATISTIC: statistic}).q[key]
            for statistic in (lower, center, upper)
        ]
        if unit is not None:
            bounds = [q.to(unit) for q in bounds]
        low, middle, high = bounds
        lines = axes.plot(time, middle, label=_text(selection, label, key))
        axes.fill_between(
            time, low, high, color=lines[0].get_color(), alpha=alpha, linewidth=0
        )
    label_axes(axes, "time", name or key)
    _legend(axes, label)
    return figure


def plot_distribution(
    result: ScanResult,
    key: str,
    *,
    dim: str,
    kind: str = "hist",
    ax: Axes | None = None,
    bins: int = 30,
    label: str | None = None,
    unit: str | None = None,
    name: str | None = None,
) -> Figure:
    """Draw a value per simulation over the draws as a histogram or a box per label.

    The histograms and boxes are labelled as the lines of `plot_bands`: with
    the labels of the other dimensions, the key without them, and a `label`
    is the label of a result without other dimensions and the prefix of the
    labels otherwise. Histograms of several calls into one `ax` share one
    legend; the bins are the ones of a call.

    Args:
        result: the result of a scan over draws.
        key: a value per simulation, e.g. the parameter of a PK observable.
        dim: the dimension of the draws.
        kind: `hist` or `box`.
        ax: the axes to draw into, a new figure without.
        bins: the number of bins of a histogram, shared by every label.
        label: the label of the distributions, see above.
        unit: the unit of the values, e.g. `ng/ml`, the one of the result by
            default.
        name: the name of the values on their axis, the key by default.

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
    quantities = [result.sel(**s).q[key] for s in selections]
    if unit is not None:
        quantities = [q.to(unit) for q in quantities]
    samples = [np.asarray(q.magnitude, dtype=float).ravel() for q in quantities]
    samples = [s[np.isfinite(s)] for s in samples]
    names = [_text(s, label, key) for s in selections]
    text = unit_label(name or key, unit or result.units.get(key))
    if kind == "hist":
        edges = np.histogram_bin_edges(np.concatenate(samples), bins=bins)
        for values, sample_name in zip(samples, names, strict=True):
            axes.hist(values, bins=edges, alpha=0.5, label=sample_name)
        axes.set_xlabel(text)
        axes.set_ylabel("count")
        _legend(axes, label)
    else:
        axes.boxplot(samples, tick_labels=names)
        axes.set_ylabel(text)
    return figure
