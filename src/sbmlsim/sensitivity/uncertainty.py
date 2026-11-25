"""The uncertainty analysis: bands of timecourses and distributions of values.

The uncertainty of a prediction is a scan over draws of the uncertain
parameters, see `sbmlsim.simulation.sampling` (`random`, `lhs`,
`fit_parameters`, `profile_parameters`, `fit_repeats`, `population`), run by
`Simulator.run`, and its summary over the dimension of the draws,
`ScanResult.summary(dim, quantiles=[0.05, 0.5, 0.95])`. The functions here
draw it as a `sbmlsim.plot.Chart`: `plot_bands` a timecourse as a median and
a band per label of the other dimensions, `plot_distribution` a value per
simulation as a histogram, a box or a violin per label. Given a mapping of
scenarios to their results instead of one result, they draw the scenarios
into one plot with one legend, labelled by their keys.

The values are quantities, `ScanResult.quantity`: the axes have the units of
the first quantities drawn into them and convert every other one, and
`time_unit` and `unit` choose the units, e.g. `time_unit="hr"`. The renderer
labels an axis with its name and the short form of its unit, e.g.
`cp [ng/ml]`.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from sbmlsim.plot.chart import Chart, ChartAxis
from sbmlsim.result.scan import STATISTIC, TIME, ScanResult
from sbmlsim.sensitivity.plots import density, draw_boxes

#: the results of the scenarios of a plot: one result, or results by the
#: labels of their scenarios
type Results = ScanResult | Mapping[str, ScanResult]

#: the kinds of a plot of a distribution
DISTRIBUTION_KINDS: Final = ("hist", "box", "violin")

#: the size of a chart in pixels (6 x 4 inches)
SIZE: Final = (432.0, 288.0)

#: the opacity of the bins of a histogram, through which the others show
HISTOGRAM_OPACITY: Final = 0.5


def _scenarios(
    results: Results, label: str | None
) -> list[tuple[str | None, ScanResult]]:
    """Get the results with the labels of their scenarios.

    Args:
        results: one result or results by the labels of their scenarios.
        label: the label of one result.

    Returns:
        The label and the result of every scenario.
    """
    if isinstance(results, Mapping):
        return [(str(name), result) for name, result in results.items()]
    return [(label, results)]


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


def plot_bands(
    summary: Results,
    key: str,
    *,
    lower: str = "q0.05",
    center: str = "q0.5",
    upper: str = "q0.95",
    label: str | None = None,
    time_unit: str | None = None,
    unit: str | None = None,
    name: str | None = None,
) -> Chart:
    """Draw a timecourse of a summary as a center line and a band per label.

    Every band is labelled with the labels of the other dimensions, e.g.
    `dose=1`, or with the key for a summary without other dimensions. A
    `label` is the label of a summary without other dimensions and the
    prefix of the labels otherwise, e.g. `mid: dose=1`; the summaries of a
    mapping take their keys as their labels, so the scenarios are told apart
    in the legend. A legend is drawn for more than one band or a label.

    Args:
        summary: a summary of a scan, `ScanResult.summary(..., quantiles=...)`,
            or summaries by the labels of their scenarios.
        key: the timecourse.
        lower: the statistic of the lower bound of the band.
        center: the statistic of the line.
        upper: the statistic of the upper bound of the band.
        label: the label of the bands of one summary, see above.
        time_unit: the unit of the time, e.g. `hr`, the one of the first
            summary by default.
        unit: the unit of the values, e.g. `ng/ml`, the one of the first
            summary by default.
        name: the name of the values on their axis, the key by default.

    Returns:
        The chart.

    Raises:
        ValueError: if the variable is no timecourse of a summary or a
            statistic is missing.
    """
    scenarios = _scenarios(summary, label)
    for _, result in scenarios:
        data = result[key]
        if STATISTIC not in data.dims or TIME not in data.dims:
            raise ValueError(
                f"'{key}' is no timecourse of a summary; summarize a scan first."
            )
        statistics = data[STATISTIC].values.tolist()
        missing = [s for s in (lower, center, upper) if s not in statistics]
        if missing:
            raise ValueError(
                f"The summary has not the statistics {missing}, add their quantiles."
            )
    bands = [
        (prefix, result, selection)
        for prefix, result in scenarios
        for selection in _labels(result[key], {STATISTIC, TIME})
    ]
    chart = Chart(*SIZE)
    plot = chart.plot(
        x=ChartAxis(label="time", unit=time_unit),
        y=ChartAxis(label=name or key, unit=unit),
        legend=len(bands) > 1 or any(prefix is not None for prefix, _ in scenarios),
    )
    for prefix, result, selection in bands:
        curve = result.sel(**selection)
        low, middle, high = (
            curve.sel(**{STATISTIC: statistic}).quantity(key)
            for statistic in (lower, center, upper)
        )
        plot.band(
            result.quantity(TIME),
            low,
            high,
            median=middle,
            label=_text(selection, prefix, key),
        )
    return chart


def plot_distribution(
    result: Results,
    key: str,
    *,
    dim: str,
    kind: str = "hist",
    bins: int = 30,
    label: str | None = None,
    unit: str | None = None,
    name: str | None = None,
) -> Chart:
    """Draw a value per simulation over the draws as a histogram, a box or a violin per label.

    The distributions are labelled as the bands of `plot_bands`: with the
    labels of the other dimensions, the key without them, and a `label` is
    the label of a result without other dimensions and the prefix of the
    labels otherwise; the results of a mapping take their keys as their
    labels. The histograms share their bins and a legend; the boxes and
    violins are side by side, labelled on the axis. A violin is the density
    of `sbmlsim.sensitivity.plots.density`, a kernel density between the
    smallest and the largest value; a sample of fewer than two distinct
    values has no violin.

    Args:
        result: the result of a scan over draws, or results by the labels of
            their scenarios.
        key: a value per simulation, e.g. the parameter of a PK observable.
        dim: the dimension of the draws.
        kind: `hist`, `box` or `violin`.
        bins: the number of bins of a histogram, shared by every label.
        label: the label of the distributions of one result, see above.
        unit: the unit of the values, e.g. `ng/ml`, the one of the first
            result by default.
        name: the name of the values on their axis, the key by default.

    Returns:
        The chart.

    Raises:
        ValueError: if the variable is a timecourse, the dimension is not one
            of it, or the kind is unknown.
    """
    if kind not in DISTRIBUTION_KINDS:
        raise ValueError(
            f"The kind of a distribution plot is one of {list(DISTRIBUTION_KINDS)}, "
            f"not '{kind}'."
        )
    names: list[str] = []
    quantities: list[Any] = []
    scenarios = _scenarios(result, label)
    for prefix, scenario in scenarios:
        data = scenario[key]
        if TIME in data.dims or "_point" in data.dims:
            raise ValueError(
                f"'{key}' is a timecourse; draw a value per simulation, e.g. at(x, t)."
            )
        if dim not in data.dims:
            raise ValueError(f"'{dim}' is no dimension of '{key}': {list(data.dims)}.")
        for selection in _labels(data, {dim}):
            names.append(_text(selection, prefix, key))
            quantities.append(scenario.sel(**selection).quantity(key))
    # every sample in one unit: the one asked for, else the one of the first
    target = unit if unit is not None else str(quantities[0].units)
    samples = [
        np.asarray(q.to(target).magnitude, dtype=float).ravel() for q in quantities
    ]
    samples = [s[np.isfinite(s)] for s in samples]
    values = ChartAxis(label=name or key, unit=target)
    chart = Chart(*SIZE)
    if kind == "hist":
        plot = chart.plot(
            x=values,
            y=ChartAxis(label="count"),
            legend=len(samples) > 1 or any(p is not None for p, _ in scenarios),
        )
        edges = np.histogram_bin_edges(np.concatenate(samples), bins=bins)
        for sample, sample_name in zip(samples, names, strict=True):
            counts, _ = np.histogram(sample, bins=edges)
            plot.histogram(
                edges, counts, fill_opacity=HISTOGRAM_OPACITY, label=sample_name
            )
        return chart
    plot = chart.plot(
        x=ChartAxis(scale="category", categories=names), y=values, legend=False
    )
    if kind == "box":
        draw_boxes(plot, samples, names)
        return chart
    spread = [k for k, s in enumerate(samples) if np.unique(s).size > 1]
    if spread:
        plot.violin(
            [names[k] for k in spread],
            [density(samples[k]) for k in spread],
        )
    return chart
