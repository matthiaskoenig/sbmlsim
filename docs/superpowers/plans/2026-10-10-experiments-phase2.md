# Figures over scan dimensions, implementation plan (experiments phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A figure of a simulation experiment draws one line per point of the scan dimensions a curve names (`over=`), a value per simulation over a dimension (cmax over dose), and the median and a quantile band over a dimension of draws (`Plot.band`), with colours, legends and colour bars, in matplotlib and plotly alike.

**Architecture:** A new module `plot/points.py` splits the broadcast arrays of a curve into one `Line` per point of its `over` dimensions and computes the colours, line styles, legend labels and band quantiles; the figure model gains `Curve.over` and a `Band` element (`Plot.bands`); `SimulationExperiment.initialize()` checks from the definitions alone that every dimension of a scan a curve or band reads is on its axis, in `over`/`across` or selected; both serializers draw the lines and bands from `plot/points.py`. The examples (demo, glucose, HCTZ Patel 1984, `experiment_scans.py`) use them.

**Tech Stack:** Python 3.13/3.14, xarray, numpy, matplotlib (`colormaps`, `ScalarMappable`, `colorbar`), plotly (`dev` extra), pint, pytest with `filterwarnings = error`, ruff, ty, zensical.

**Spec:** `docs/superpowers/specs/2026-10-10-experiments-design.md`, section "Figures over scan dimensions" and the figure parts of "The callers"; phase 1 (merged as `ef18c43d`) gives `Data` as labelled arrays, `Data(sel=...)`, the values of a dimension as `Data("<dimension>.<target>")` and `plot/padding.line_values`.

**Not in this phase:** the fits of scalar observables and the PEtab gaps (phase 3).

## Global Constraints

- Python >= 3.13; no new runtime dependency (plotly stays in the `dev` extra).
- `ty check` stays at zero diagnostics (`error-on-warning = true`); suppress only with a rule specific `# ty: ignore[rule]`, never `# type: ignore`.
- Every module, class and function of the package has full type annotations and a google style docstring (ruff `D`); `tests/` and `examples/` are exempt from the docstring rules; test experiment classes need no `typing.override`.
- A `SimulationExperiment` defines its methods in the order `datasets`, `models`, `simulations`, `observables`, `tasks`, `data`, `fit_mappings`, `figures`, each with `typing.override` in the package and the examples.
- Library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting, never prints, never calls `plt.show()`; figures are created with `matplotlib.figure.Figure`, not pyplot.
- `filterwarnings = error`: fix the cause of a warning, never filter it.
- Pixel perfect figures: render every changed figure and look at it; nothing clipped or overlapping, legends never cover data, units in labels.
- Never use the em dash character. Markdown has no hard line wraps.
- Commit messages are full sentences with a body, no conventional prefixes, no attribution lines of any kind.
- Add files by path, never `git add -A`, `git add .` or a directory add: an untracked `docs/README.md` belongs to someone else and must never be committed. Run examples only from a scratch directory under `/tmp/claude-1000/`.
- Do not edit `CHANGELOG.md` or release notes.
- The spec's rules for curves: after `sel`, the data of a curve are broadcast by dimension name; the dimensions of x which are not in `over` must be exactly one, the axis; a dimension of y which is neither the axis nor in `over` raises "y of curve '<sid>' has the dimension '<dim>'; name it with over='<dim>' or select a label with Data(sel=...)"; one `over` dimension: shades of the colour of the curve's style if it sets one, else `viridis`; two: the colour follows the first, the line style (solid, dashed, dotted, dash-dot) the second, more than four points of the second raise; a legend entry names its point by the value and unit of the changed target if the dimension changes exactly one target (`PODOSE_hctz = 25 mg`), else by its label; above ten points a colour bar of the dimension replaces the legend entries.
- The spec's band: `Plot.band(x, y, across="draw", quantiles=(0.05, 0.95), median=True, over=None, style=None, name=None)` reduces the dimension `across` of y to the two quantiles (ignoring `NaN`), draws the area between them and the median as a line; computed when the figure is drawn.

## Review Focus

1. A band over a ragged result (draws without `steps`/`times`): the simulations have their own time points, so quantiles over `_point` positions mix times; it must raise with the advice to use a common grid. Test: Task 1 `test_a_band_of_a_ragged_result_raises`, Task 2 `test_a_band_of_a_ragged_scan_raises_at_initialize`.
2. Two curves of one plot over the same dimension with no colour set: both take `viridis` and are told apart only by their legend names. Test: Task 3 `test_two_curves_over_one_dimension_keep_their_names_in_the_legend`.
3. A point whose simulation failed (`NaN` values) inside a curve over a dimension or a band: the line has a gap, the quantiles ignore it. Test: Task 1 `test_band_quantiles_ignore_nan`.
4. A curve over a dimension of twelve points: a colour bar with the values and unit of the dimension, no legend entries, nothing overlapping. Test: Task 3 `test_eleven_points_and_more_get_a_colour_bar`, Task 4 the same for plotly.
5. A bar curve with `over`: bars draw no lines per point; it must raise a clear error rather than overplot. Test: Task 3 `test_a_bar_curve_over_a_dimension_raises`.

---

### Task 1: The lines, colours, labels and bands of a curve over scan points

**Files:**
- Create: `src/sbmlsim/plot/points.py`
- Test: `tests/plot/test_points.py`

**Interfaces:**
- Consumes: `sbmlsim.plot.padding.without_padding`, `sbmlsim.data.ROW`, `sbmlsim.result.scan.TIME`, `POINT`, `sbmlsim.simulation.Dimension` (`id`, `values`, `labels`), `sbmlsim.units.Quantity`.
- Produces (used by Tasks 2 to 4):
  - `LINE_STYLES: tuple[str, ...] = ("-", "--", ":", "-.")`, `COLORBAR_FROM: int = 11`, `COLORMAP: str = "viridis"`.
  - `@dataclass(frozen=True) class Line: point: tuple[Any, ...]; index: tuple[int, ...]; x: Any; y: Any; xerr: Any = None; yerr: Any = None`.
  - `curve_lines(sid: str, over: Sequence[str], x: xr.DataArray, y: xr.DataArray, xerr: xr.DataArray | None = None, yerr: xr.DataArray | None = None) -> list[Line]`.
  - `@dataclass(frozen=True) class BandLine: point: tuple[Any, ...]; index: tuple[int, ...]; x: Any; low: Any; median: Any; high: Any`.
  - `band_lines(sid: str, over: Sequence[str], across: str, quantiles: tuple[float, float], x: xr.DataArray, y: xr.DataArray) -> list[BandLine]`.
  - `point_colors(n: int, color: str | None) -> list[str]` (hex), `point_colormap(color: str | None) -> Colormap`, `point_linestyles(n: int) -> list[str]`.
  - `point_labels(dim: str, labels: Sequence[Any], dimension: Dimension | None, units: Mapping[str, str]) -> list[str]` and `point_values(dimension: Dimension | None, units: Mapping[str, str]) -> tuple[str, np.ndarray, str] | None` (the target, the values and the unit of a dimension which changes exactly one target).

- [ ] **Step 1: Write the failing tests**

Create `tests/plot/test_points.py`:

```python
"""The lines of a curve over the points of the dimensions of a scan."""

import numpy as np
import pytest
import xarray as xr
from matplotlib import colormaps
from matplotlib.colors import to_hex, to_rgb

from sbmlsim import Q
from sbmlsim.plot.points import (
    LINE_STYLES,
    band_lines,
    curve_lines,
    point_colors,
    point_labels,
    point_linestyles,
    point_values,
)
from sbmlsim.simulation import Dimension

TIME = np.array([0.0, 1.0, 2.0])


def _time() -> xr.DataArray:
    return xr.DataArray(TIME, dims=("time",), coords={"time": TIME})


def _scan(*dims: tuple[str, list]) -> xr.DataArray:
    shape = [len(labels) for _, labels in dims] + [TIME.size]
    values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    coords = {name: labels for name, labels in dims} | {"time": TIME}
    return xr.DataArray(values, dims=(*[n for n, _ in dims], "time"), coords=coords)


def test_one_line_per_point_of_a_dimension() -> None:
    y = _scan(("dose", ["lo", "mid", "hi"]))
    lines = curve_lines("c", ("dose",), _time(), y)
    assert [line.point for line in lines] == [("lo",), ("mid",), ("hi",)]
    assert [line.index for line in lines] == [(0,), (1,), (2,)]
    for k, line in enumerate(lines):
        np.testing.assert_array_equal(line.x, TIME)
        np.testing.assert_array_equal(line.y, y.values[k])


def test_two_dimensions_the_last_fastest() -> None:
    y = _scan(("a", [1, 2]), ("b", ["x", "y", "z"]))
    lines = curve_lines("c", ("a", "b"), _time(), y)
    assert [line.point for line in lines] == [
        (1, "x"), (1, "y"), (1, "z"), (2, "x"), (2, "y"), (2, "z"),
    ]
    np.testing.assert_array_equal(lines[4].y, y.values[1, 1])


def test_the_order_of_over_decides_and_not_the_order_of_the_data() -> None:
    y = _scan(("a", [1, 2]), ("b", ["x", "y", "z"]))
    lines = curve_lines("c", ("b", "a"), _time(), y)
    assert [line.point for line in lines][:2] == [("x", 1), ("x", 2)]
    np.testing.assert_array_equal(lines[1].y, y.values[1, 0])


def test_a_value_per_simulation_over_a_dimension() -> None:
    dose = xr.DataArray([10.0, 20.0, 40.0], dims=("dose",), coords={"dose": ["lo", "mid", "hi"]})
    cmax = xr.DataArray([1.0, 2.1, 3.9], dims=("dose",), coords={"dose": ["lo", "mid", "hi"]})
    (line,) = curve_lines("c", (), dose, cmax)
    np.testing.assert_array_equal(line.x, [10.0, 20.0, 40.0])
    np.testing.assert_array_equal(line.y, [1.0, 2.1, 3.9])
    points = curve_lines("c", ("dose",), dose, cmax)
    assert len(points) == 3 and float(points[2].y) == 3.9


def test_a_dimension_neither_axis_nor_over_raises() -> None:
    y = _scan(("dose", [0, 1]))
    with pytest.raises(ValueError, match=r"curve 'c'.*name \['dose'\] in over=.*Data\(sel=\.\.\.\)"):
        curve_lines("c", (), _time(), y)


def test_over_names_a_dimension_the_data_has_not() -> None:
    with pytest.raises(ValueError, match=r"curve 'c' draws a line per point of \['nope'\]"):
        curve_lines("c", ("nope",), _time(), _scan(("dose", [0, 1])))


def test_the_padding_of_a_ragged_result_is_dropped_per_line() -> None:
    t = xr.DataArray([[0.0, 1.0, np.nan], [0.0, 1.0, 2.0]], dims=("dose", "_point"))
    y = xr.DataArray([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]], dims=("dose", "_point"))
    first, second = curve_lines("c", ("dose",), t, y)
    assert first.x.tolist() == [0.0, 1.0] and first.y.tolist() == [1.0, 2.0]
    assert second.x.tolist() == [0.0, 1.0, 2.0]


def test_band_quantiles_ignore_nan() -> None:
    values = np.array([[1.0, 2.0, 3.0], [2.0, np.nan, 4.0], [3.0, 6.0, 5.0], [4.0, 8.0, 6.0]])
    y = xr.DataArray(values, dims=("draw", "time"), coords={"time": TIME})
    (band,) = band_lines("b", (), "draw", (0.25, 0.75), _time(), y)
    np.testing.assert_allclose(band.low, np.nanquantile(values, 0.25, axis=0))
    np.testing.assert_allclose(band.high, np.nanquantile(values, 0.75, axis=0))
    np.testing.assert_allclose(band.median, np.nanmedian(values, axis=0))


def test_a_band_per_point_of_another_dimension() -> None:
    values = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    y = xr.DataArray(values, dims=("dose", "draw", "time"), coords={"dose": ["lo", "hi"], "time": TIME})
    bands = band_lines("b", ("dose",), "draw", (0.05, 0.95), _time(), y)
    assert [band.point for band in bands] == [("lo",), ("hi",)]
    np.testing.assert_allclose(bands[1].median, np.median(values[1], axis=0))


def test_a_band_of_a_ragged_result_raises() -> None:
    y = xr.DataArray(np.ones((4, 3)), dims=("draw", "_point"))
    t = xr.DataArray(np.ones((4, 3)), dims=("draw", "_point"))
    with pytest.raises(ValueError, match=r"band 'b'.*common grid"):
        band_lines("b", (), "draw", (0.05, 0.95), t, y)


def test_a_band_across_a_dimension_the_data_has_not_raises() -> None:
    with pytest.raises(ValueError, match=r"band 'b' reduces the dimension 'draw'"):
        band_lines("b", (), "draw", (0.05, 0.95), _time(), _scan(("dose", [0, 1])))


def test_point_colors() -> None:
    shades = point_colors(3, "#ff0000")
    assert shades[-1] == "#ff0000" and len(set(shades)) == 3
    assert sum(to_rgb(shades[0])) > sum(to_rgb(shades[1]))  # lighter first
    viridis = point_colors(3, None)
    assert viridis[0] == to_hex(colormaps["viridis"](0.0)) and len(set(viridis)) == 3
    assert point_colors(1, "#00ff00") == ["#00ff00"]


def test_point_linestyles() -> None:
    assert point_linestyles(3) == list(LINE_STYLES[:3])
    with pytest.raises(ValueError, match="at most 4"):
        point_linestyles(5)


def test_point_labels_name_the_values_of_a_changed_target() -> None:
    dose = Dimension("dose", values={"PODOSE": Q([50.0, 100.0], "mg")})
    assert point_labels("dose", [0, 1], dose, {}) == ["PODOSE = 50 mg", "PODOSE = 100 mg"]
    target, values, unit = point_values(dose, {}) or ("", np.array([]), "")
    assert target == "PODOSE" and unit == "mg" and values.tolist() == [50.0, 100.0]
    plain = Dimension("ke", values={"ke": np.array([0.1, 0.2])})
    assert point_labels("ke", [0, 1], plain, {"ke": "1/hr"}) == ["ke = 0.1 1/hr", "ke = 0.2 1/hr"]
    two = Dimension("both", values={"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}, labels=["x", "y"])
    assert point_labels("both", ["x", "y"], two, {}) == ["both = x", "both = y"]
    assert point_labels("d", ["lo", "hi"], None, {}) == ["d = lo", "d = hi"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/plot/test_points.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbmlsim.plot.points'`.

- [ ] **Step 3: Implement `src/sbmlsim/plot/points.py`**

```python
"""The lines of a curve over the points of the dimensions of a scan.

A curve names the dimensions of its data it draws one line per point of
(`over`); after broadcasting by name its data must leave at most one more
dimension, the axis a line is drawn along: the time of a timecourse, the
points of a ragged result, the rows of a dataset or the dimension of the
values a scan sets (a value per simulation over a dimension). The lines are in
the order of the labels of the `over` dimensions, the last one fastest. Their
colours follow the first `over` dimension, their line styles the second; a
legend entry names its point by the value and unit of the target its
dimension changes, else by its label. A band reduces a dimension of draws to
two quantiles and the median, computed when the figure is drawn; it needs a
common grid of times.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from matplotlib import colormaps
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap, to_hex, to_rgb

from sbmlsim.data import ROW
from sbmlsim.plot.padding import without_padding
from sbmlsim.result.scan import POINT, TIME
from sbmlsim.simulation import Dimension
from sbmlsim.units import Quantity

#: the line styles of the points of a second `over` dimension
LINE_STYLES: tuple[str, ...] = ("-", "--", ":", "-.")

#: the number of points from which a dimension gets a colour bar instead of
#: legend entries
COLORBAR_FROM = 11

#: the colour map of the points of a curve whose style sets no colour
COLORMAP = "viridis"

#: the part of the colour map the points use: its light yellow end is hard to
#: see on white
_COLORMAP_END = 0.85

#: how much white the lightest shade of a colour has
_LIGHTEST = 0.65


@dataclass(frozen=True)
class Line:
    """One line of a curve, see `curve_lines`.

    Attributes:
        point: the label of every `over` dimension, in their order.
        index: the position of the point along every `over` dimension.
        x: the x values along the axis, without the padding.
        y: the y values.
        xerr: the x errors, `None` without.
        yerr: the y errors, `None` without.
    """

    point: tuple[Any, ...]
    index: tuple[int, ...]
    x: Any
    y: Any
    xerr: Any = None
    yerr: Any = None


@dataclass(frozen=True)
class BandLine:
    """One band, see `band_lines`.

    Attributes:
        point: the label of every `over` dimension, in their order.
        index: the position of the point along every `over` dimension.
        x: the x values along the axis.
        low: the lower quantile.
        median: the median.
        high: the upper quantile.
    """

    point: tuple[Any, ...]
    index: tuple[int, ...]
    x: Any
    low: Any
    median: Any
    high: Any


def curve_lines(
    sid: str,
    over: Sequence[str],
    x: xr.DataArray,
    y: xr.DataArray,
    xerr: xr.DataArray | None = None,
    yerr: xr.DataArray | None = None,
) -> list[Line]:
    """Split the data of a curve into one line per point of its `over` dimensions.

    Args:
        sid: the id of the curve, for the errors.
        over: the dimensions with a line per point, in the order of the lines.
        x: the x data.
        y: the y data.
        xerr: the x errors, `None` without.
        yerr: the y errors, `None` without.

    Returns:
        The lines, without the padding of a ragged result.

    Raises:
        ValueError: if two arrays have different coordinates of a dimension, an
            `over` dimension is not one of the data, or more than one other
            dimension is left.
    """
    full, dims = _broadcast(sid, "curve", [x, y, xerr, yerr])
    missing = [d for d in over if d not in dims]
    if missing:
        raise ValueError(
            f"The curve '{sid}' draws a line per point of {missing}, which its data "
            f"has not: {dims}."
        )
    axis = [d for d in dims if d not in over]
    if len(axis) > 1:
        scan = [d for d in axis if d not in (TIME, POINT, ROW)] or axis
        raise ValueError(
            f"The curve '{sid}' has the dimensions {dims}; a curve draws a line along "
            f"one of them: name {scan} in over= or select one label with "
            f"Data(sel=...)."
        )
    ordered = [None if a is None else a.transpose(*over, *axis) for a in full]
    labels = _labels(ordered, over)
    lines: list[Line] = []
    for index in itertools.product(*(range(len(labels[d])) for d in over)):
        position = dict(zip(over, index, strict=True))
        values = [
            None if a is None else np.asarray(a.isel(position).values) for a in ordered
        ]
        lx, ly, lxerr, lyerr = without_padding(*values)
        lines.append(
            Line(
                point=tuple(labels[d][k] for d, k in position.items()),
                index=tuple(index),
                x=lx,
                y=ly,
                xerr=lxerr,
                yerr=lyerr,
            )
        )
    return lines


def band_lines(
    sid: str,
    over: Sequence[str],
    across: str,
    quantiles: tuple[float, float],
    x: xr.DataArray,
    y: xr.DataArray,
) -> list[BandLine]:
    """Reduce the dimension `across` of the data of a band to quantiles and the median.

    Args:
        sid: the id of the band, for the errors.
        over: the dimensions with a band per point.
        across: the dimension of the draws, which is reduced.
        quantiles: the lower and the upper quantile, in `[0, 1]`.
        x: the x data.
        y: the y data.

    Returns:
        One band per point of `over`.

    Raises:
        ValueError: if the data is ragged, has not the dimension `across`, or
            leaves more than one dimension besides `over` and `across`.
    """
    if POINT in y.dims or POINT in x.dims:
        raise ValueError(
            f"The band '{sid}' reduces '{across}' of a ragged result, whose "
            f"simulations keep their own time points; run the scan on a common grid "
            f"(a simulation with steps or times)."
        )
    if across not in y.dims:
        raise ValueError(
            f"The band '{sid}' reduces the dimension '{across}', which its data has "
            f"not: {[str(d) for d in y.dims]}."
        )
    low = y.quantile(quantiles[0], dim=across, skipna=True).drop_vars("quantile")
    high = y.quantile(quantiles[1], dim=across, skipna=True).drop_vars("quantile")
    median = y.median(dim=across, skipna=True)
    lows = curve_lines(sid, over, x, low)
    medians = curve_lines(sid, over, x, median)
    highs = curve_lines(sid, over, x, high)
    return [
        BandLine(point=lo.point, index=lo.index, x=lo.x, low=lo.y, median=me.y, high=hi.y)
        for lo, me, hi in zip(lows, medians, highs, strict=True)
    ]


def point_colors(n: int, color: str | None) -> list[str]:
    """Get the colours of the points of a dimension.

    Args:
        n: the number of points.
        color: the colour of the style of the curve, `None` without.

    Returns:
        Shades of the colour from light to the colour itself, or colours of the
        colour map `COLORMAP`, as hex strings; one point keeps the colour.
    """
    if color is not None:
        if n == 1:
            return [to_hex(color)]
        return [_shade(color, f) for f in np.linspace(_LIGHTEST, 0.0, n)]
    return [to_hex(colormaps[COLORMAP](f)) for f in np.linspace(0.0, _COLORMAP_END, n)]


def point_colormap(color: str | None) -> Colormap:
    """Get the colour map of the colour bar of a dimension, see `point_colors`."""
    if color is not None:
        return LinearSegmentedColormap.from_list(
            "shades", [_shade(color, _LIGHTEST), to_hex(color)]
        )
    return ListedColormap(colormaps[COLORMAP](np.linspace(0.0, _COLORMAP_END, 256)))


def point_linestyles(n: int) -> list[str]:
    """Get the line styles of the points of a second `over` dimension.

    Raises:
        ValueError: for more than `len(LINE_STYLES)` points.
    """
    if n > len(LINE_STYLES):
        raise ValueError(
            f"A second dimension of a curve has at most {len(LINE_STYLES)} points, "
            f"one per line style, not {n}."
        )
    return list(LINE_STYLES[:n])


def point_values(
    dimension: Dimension | None, units: Mapping[str, str]
) -> tuple[str, np.ndarray, str] | None:
    """Get the target, the values and the unit of a dimension which changes one target.

    Args:
        dimension: the dimension of the scan, `None` if it is not known.
        units: the units of the symbols of the model, for values without a unit.

    Returns:
        The target, its values and their unit, `None` for a dimension which
        changes no or several targets.
    """
    if dimension is None or len(dimension.values) != 1:
        return None
    ((target, values),) = dimension.values.items()
    if isinstance(values, Quantity):
        return target, np.asarray(values.magnitude, dtype=float), f"{values.units:~P}"
    return target, np.asarray(values, dtype=float), units.get(target, "") or ""


def point_labels(
    dim: str,
    labels: Sequence[Any],
    dimension: Dimension | None,
    units: Mapping[str, str],
) -> list[str]:
    """Get the legend label of every point of a dimension.

    Args:
        dim: the id of the dimension.
        labels: the labels of its points.
        dimension: the dimension of the scan, `None` if it is not known.
        units: the units of the symbols of the model, see `point_values`.

    Returns:
        `<target> = <value> <unit>` for a dimension which changes one target,
        else `<dim> = <label>`.
    """
    found = point_values(dimension, units)
    if found is None:
        return [f"{dim} = {label}" for label in labels]
    target, values, unit = found
    return [f"{target} = {value:g} {unit}".rstrip() for value in values]


def _broadcast(
    sid: str, what: str, arrays: Sequence[xr.DataArray | None]
) -> tuple[list[xr.DataArray | None], list[str]]:
    """Broadcast the arrays of a curve by name.

    Raises:
        ValueError: if two arrays have different coordinates of a dimension.
    """
    given = [a for a in arrays if a is not None]
    try:
        aligned = xr.align(*given, join="exact")
    except ValueError as err:
        raise ValueError(
            f"The data of the {what} '{sid}' has different coordinates of a "
            f"dimension: {err}"
        ) from err
    broadcast = iter(xr.broadcast(*aligned))
    full = [None if a is None else next(broadcast) for a in arrays]
    template = next(a for a in full if a is not None)
    return full, [str(d) for d in template.dims]


def _labels(
    arrays: Sequence[xr.DataArray | None], dims: Sequence[str]
) -> dict[str, list[Any]]:
    """Get the labels of dimensions, their positions for a dimension without labels."""
    template = next(a for a in arrays if a is not None)
    return {
        d: template[d].values.tolist()
        if d in template.coords
        else list(range(template.sizes[d]))
        for d in dims
    }


def _shade(color: str, white: float) -> str:
    """Mix a colour with a part of white, as a hex string."""
    rgb = np.asarray(to_rgb(color))
    return to_hex(rgb + (1.0 - rgb) * white)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/plot/test_points.py`
Expected: PASS.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/plot/points.py tests/plot/test_points.py
git commit -m "The lines, colours, labels and bands of a curve over the points of a scan" -m "plot/points.py splits the broadcast data of a curve into one line per point of the dimensions it names, in the order of their labels, reduces a dimension of draws to two quantiles and the median, and gives the colours (shades of the colour of the curve or viridis), the line styles of a second dimension and the legend labels by the value and unit of the changed target."
```

---

### Task 2: `over` and `Band` in the figure model, checked at `initialize()`

**Files:**
- Modify: `src/sbmlsim/plot/plotting.py` (`Curve(over=)`, `Band`, `Plot.bands`, `Plot.curve(over=)`, `Plot.add_data(over=)`, `Plot.band`, `to_dict`, `__copy__`)
- Modify: `src/sbmlsim/plot/__init__.py` (export `Band`)
- Modify: `src/sbmlsim/experiment/experiment.py` (`_figure_data` reads bands; `_scan_dims`, `_check_figures`, called from `initialize()`)
- Create: `tests/plot/scan_experiment.py` (the experiment the tests of Tasks 2 to 4 draw)
- Test: `tests/plot/test_scan_figure_model.py`, `tests/experiment/test_experiment_run.py` (the test of a curve without `sel`)

**Interfaces:**
- Consumes: `point_linestyles` and `LINE_STYLES` (Task 1), `SimulationExperiment._index_kind`, `_coordinates`, `_labels` (phase 1), `Scan.simulations`, `Simulation.steps`, `Simulation.times`.
- Produces:
  - `Curve(..., over: str | Sequence[str] | None = None)`, attribute `over: tuple[str, ...]`; `Plot.curve(..., over=None)`; `Plot.add_data(..., over=None)`.
  - `class Band(AbstractCurve)` with `__init__(self, x: Data, y: Data, across: str, quantiles: tuple[float, float] = (0.05, 0.95), median: bool = True, over: str | Sequence[str] | None = None, sid: str | None = None, name: str | None = None, color: str | None = None, alpha: float = 0.3, order: int | None = None, yaxis_position: YAxisPosition | None = None)`, attributes `y`, `across`, `quantiles`, `median`, `over: tuple[str, ...]`, `color`, `alpha`; `to_dict`.
  - `Plot.bands: list[Band]`, `Plot.add_band(band)`, `Plot.band(x, y, across, quantiles=(0.05, 0.95), median=True, over=None, name=None, color=None, alpha=0.3, yaxis_position=None) -> Band`.
  - The experiment raises at `initialize()` for: a dimension of a scan of a curve's or band's task data which is neither its axis, in `over`/`across`, nor selected by a single label; an `over` or `across` dimension its data has not; more than `len(LINE_STYLES)` points of a second `over` dimension; a band of a ragged scan. Function and dataset data are checked when they are drawn.

- [ ] **Step 1: Write the test experiment and the failing tests**

Create `tests/plot/scan_experiment.py` (a module the tests of Tasks 2 to 4 import; it is no test file):

```python
"""A dosed one-compartment model over scans, for the figures over scan points."""

import numpy as np

from sbmlsim import Q
from sbmlsim.data import Data
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import PK, Change, Dimension, Observable, Scan, Simulation
from sbmlsim.task import Task
from tests.simulator.models import sbml_pk


def dosed(steps: int | None = 48) -> Simulation:
    return Simulation(end=24, steps=steps, changes=[Change(0, {"PODOSE": Q(100, "mg")})])


def doses(n: int = 3) -> Dimension:
    values = [50.0, 100.0, 200.0] if n == 3 else np.linspace(10.0, 120.0, n).tolist()
    return Dimension("dose", values={"PODOSE": Q(values, "mg")})


class ScanFigures(SimulationExperiment):
    """Doses, many doses, doses times elimination rates, draws and a ragged scan."""

    def models(self) -> dict:
        return {"m": AbstractModel(source=sbml_pk())}

    def simulations(self) -> dict:
        return {
            "doses": Scan(dosed(), [doses()]),
            "many": Scan(dosed(), [doses(12)]),
            "grid2": Scan(dosed(), [doses(), Dimension("ke", values={"ke": np.array([0.1, 0.3])})]),
            "draws": Scan(dosed(), [Dimension("draw", values={"ke": np.linspace(0.1, 0.4, 20)})]),
            "dose_draws": Scan(dosed(), [doses(), Dimension("draw", values={"ke": np.linspace(0.1, 0.4, 8)})]),
            "grid5": Scan(dosed(), [doses(), Dimension("ke", values={"ke": np.linspace(0.1, 0.5, 5)})]),
            "ragged": Scan(dosed(steps=None), [doses()]),
        }

    def observables(self) -> dict[str, Observable]:
        return {"pk": PK("pk", "[C]", dose="PODOSE", route="oral")}

    def tasks(self) -> dict:
        return {f"task_{key}": Task(model="m", simulation=key) for key in self._simulations}

    def data(self) -> dict:
        self.add_selections_data(["time", "[C]"])
        return {
            "cmax_doses": Data("pk.cmax", task="task_doses"),
            "cmax_many": Data("pk.cmax", task="task_many"),
        }
```

Create `tests/plot/test_scan_figure_model.py`:

```python
"""Curves over scan points and bands in the figure model, checked at initialize."""

from pathlib import Path

import pytest

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.plot import Axis, Band, Curve, Figure
from sbmlsim.simulator import Simulator
from tests.plot.scan_experiment import ScanFigures


def _runner(experiment_class: type[SimulationExperiment]) -> ExperimentRunner:
    return ExperimentRunner(
        experiment_classes=[experiment_class],
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
    )


def _with_figure(draw) -> type[SimulationExperiment]:
    """An experiment of the scans with one plot which `draw(plot)` fills."""

    class WithFigure(ScanFigures):
        def figures(self) -> dict:
            figure = Figure(experiment=self, sid="fig", num_rows=1, num_cols=1)
            plot = figure.create_plots(xaxis=Axis("time"), yaxis=Axis("C"))[0]
            draw(plot)
            return {"fig": figure}

    return WithFigure


def test_a_curve_and_a_band_know_their_dimensions() -> None:
    curve = Curve(x=Data("time", task="t"), y=Data("[C]", task="t"), over="dose")
    assert curve.over == ("dose",) and curve.to_dict()["over"] == ["dose"]
    assert Curve(x=Data("time", task="t"), y=Data("[C]", task="t")).over == ()
    band = Band(Data("time", task="t"), Data("[C]", task="t"), across="draw", over=["dose"])
    d = band.to_dict()
    assert d["across"] == "draw" and d["quantiles"] == [0.05, 0.95] and d["over"] == ["dose"]
    with pytest.raises(ValueError, match="quantiles"):
        Band(Data("time", task="t"), Data("[C]", task="t"), across="draw", quantiles=(0.9, 0.1))


def test_plot_band_and_curve_over() -> None:
    def draw(plot) -> None:
        plot.curve(x=Data("time", task="task_doses"), y=Data("[C]", task="task_doses"), over="dose")
        band = plot.band(Data("time", task="task_draws"), Data("[C]", task="task_draws"), across="draw")
        assert band.sid == f"{plot.sid}_band0" and plot.bands == [band]

    experiment = _runner(_with_figure(draw)).experiments["WithFigure"]
    plot = experiment._figures["fig"].get_plots()[0]
    assert plot.curves[0].over == ("dose",) and len(plot.bands) == 1


def test_a_scan_dimension_which_is_not_named_raises_at_initialize() -> None:
    def draw(plot) -> None:
        plot.curve(x=Data("time", task="task_doses"), y=Data("[C]", task="task_doses"))

    with pytest.raises(ValueError, match=r"y of curve '.*' has the dimension 'dose'; name it with over='dose'"):
        _runner(_with_figure(draw))


def test_over_a_dimension_the_data_has_not_raises_at_initialize() -> None:
    def draw(plot) -> None:
        plot.curve(x=Data("time", task="task_doses"), y=Data("[C]", task="task_doses"), over="nope")

    with pytest.raises(ValueError, match=r"draws a line per point of \['nope'\]"):
        _runner(_with_figure(draw))


def test_a_value_per_simulation_over_its_dimension_passes() -> None:
    def draw(plot) -> None:
        plot.curve(x=Data("dose.PODOSE", task="task_doses"), y=Data("pk.cmax", task="task_doses"))

    _runner(_with_figure(draw))


def test_more_than_four_points_of_a_second_dimension_raise() -> None:
    def draw(plot) -> None:
        plot.curve(x=Data("time", task="task_grid2"), y=Data("[C]", task="task_grid2"), over=("dose", "ke"))

    _runner(_with_figure(draw))  # two points of ke: fine

    def draw_five(plot) -> None:
        plot.curve(x=Data("time", task="task_grid5"), y=Data("[C]", task="task_grid5"), over=("dose", "ke"))

    with pytest.raises(ValueError, match="at most 4"):
        _runner(_with_figure(draw_five))


def test_a_dimension_named_twice_and_a_band_over_two_dimensions_raise() -> None:
    with pytest.raises(ValueError, match="twice"):
        Curve(x=Data("time", task="t"), y=Data("[C]", task="t"), over=("dose", "dose"))
    with pytest.raises(ValueError, match="one dimension"):
        Band(Data("time", task="t"), Data("[C]", task="t"), across="draw", over=("dose", "ke"))


def test_a_band_needs_its_dimension_and_names_the_others() -> None:
    def no_across(plot) -> None:
        plot.band(Data("time", task="task_doses"), Data("[C]", task="task_doses"), across="draw")

    with pytest.raises(ValueError, match="reduces the dimension 'draw'"):
        _runner(_with_figure(no_across))

    def unnamed(plot) -> None:
        plot.band(Data("time", task="task_dose_draws"), Data("[C]", task="task_dose_draws"), across="draw")

    with pytest.raises(ValueError, match="name it with over='dose'"):
        _runner(_with_figure(unnamed))

    def named(plot) -> None:
        plot.band(Data("time", task="task_dose_draws"), Data("[C]", task="task_dose_draws"), across="draw", over="dose")

    _runner(_with_figure(named))


def test_a_band_of_a_ragged_scan_raises_at_initialize() -> None:
    def draw(plot) -> None:
        plot.band(Data("time", task="task_ragged"), Data("[C]", task="task_ragged"), across="dose")

    with pytest.raises(ValueError, match="common grid"):
        _runner(_with_figure(draw))


def test_the_data_of_a_band_counts_for_the_selections() -> None:
    class BandOnly(ScanFigures):
        def data(self) -> dict:
            return {}

        def figures(self) -> dict:
            figure = Figure(experiment=self, sid="fig", num_rows=1, num_cols=1)
            plot = figure.create_plots(xaxis=Axis("time"), yaxis=Axis("C"))[0]
            plot.band(Data("time", task="task_draws"), Data("[C]", task="task_draws"), across="draw")
            return {"fig": figure}

    runner = _runner(BandOnly)
    experiment = runner.experiments["BandOnly"]
    experiment.run(runner.simulator)
    assert "[C]" in experiment.results["task_draws"].ds.data_vars
```

In `tests/experiment/test_experiment_run.py`, the test `test_a_curve_of_a_scan_without_a_selection_raises` now expects the error at initialize: `with pytest.raises(ValueError, match=r"name it with over='d'"): _runner(ScanFigureExperiment)`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/plot/test_scan_figure_model.py`
Expected: FAIL with `ImportError: cannot import name 'Band'`.

- [ ] **Step 3: Implement the figure model**

In `src/sbmlsim/plot/plotting.py`:

- A helper next to the curve classes:

```python
def _dimensions(over: str | Sequence[str] | None, sid: str | None) -> tuple[str, ...]:
    """Get the dimensions a curve or band draws one line per point of.

    Raises:
        ValueError: if a dimension is named twice.
    """
    names = (over,) if isinstance(over, str) else tuple(over or ())
    if len(set(names)) != len(names):
        raise ValueError(f"The curve '{sid}' names a dimension twice in over: {names}.")
    return names
```

- `Curve.__init__` gains `over: str | Sequence[str] | None = None` (after `yaxis_position`, before `**kwargs`), documented ("the dimensions of a scan with one line per point, see `sbmlsim.plot.points`"), sets `self.over = _dimensions(over, sid)`; `Curve.to_dict` adds `"over": list(self.over)`; `__repr__`/`__str__` show it.
- `Plot.curve` gains `over: str | Sequence[str] | None = None` and passes it; `Plot.add_data` gains `over` and passes it to `self.curve`.
- The class `Band` after `ShadedArea`:

```python
class Band(AbstractCurve):
    """The median and a quantile band of data over a dimension of draws.

    The dimension `across` of y is reduced to the two quantiles, ignoring
    `NaN`, and the median, when the figure is drawn, see
    `sbmlsim.plot.points.band_lines`; `over` gives one band per point of
    other dimensions, in the colours of a curve over them.
    """

    def __init__(
        self,
        x: Data,
        y: Data,
        across: str,
        quantiles: tuple[float, float] = (0.05, 0.95),
        median: bool = True,
        over: str | Sequence[str] | None = None,
        sid: str | None = None,
        name: str | None = None,
        color: str | None = None,
        alpha: float = 0.3,
        order: int | None = None,
        yaxis_position: YAxisPosition | None = None,
    ):
        """Initialize a band.

        Args:
            x: x data.
            y: y data, with the dimension `across`.
            across: the dimension of the draws, which is reduced.
            quantiles: the lower and the upper quantile, `0 <= low < high <= 1`.
            median: draw the median as a line.
            over: the dimensions with one band per point.
            sid: identifier of the band.
            name: label of the band, the name of y by default.
            color: colour of the band, the first colour of the colour cycle
                by default.
            alpha: opacity of the area.
            order: order of the band in the plot.
            yaxis_position: position of the y axis of the band.

        Raises:
            ValueError: if the quantiles are not increasing within `[0, 1]`,
                `across` is in `over`, or `over` names more than one dimension.
        """
        super().__init__(sid=sid, name=name if name else y.name, x=x, order=order, yaxis_position=yaxis_position)
        low, high = (float(q) for q in quantiles)
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(f"The quantiles of a band are increasing within [0, 1], not {quantiles}.")
        self.x: Data = x
        self.y: Data = y
        self.across: str = across
        self.quantiles: tuple[float, float] = (low, high)
        self.median: bool = median
        self.over: tuple[str, ...] = _dimensions(over, sid)
        if across in self.over:
            raise ValueError(f"The band '{sid}' reduces '{across}', which is also in over.")
        if len(self.over) > 1:
            raise ValueError(f"A band is drawn per point of one dimension, not of {self.over}.")
        self.color: str | None = color
        self.alpha: float = alpha

    def __repr__(self) -> str:
        """Get representation string."""
        return f"Band(sid={self.sid} name={self.name} across={self.across} over={self.over})"

    def to_dict(self) -> dict[str, Any]:
        """Convert the band to a dictionary."""
        return {
            "sid": self.sid,
            "name": self.name,
            "x": self.x.sid,
            "y": self.y.sid,
            "across": self.across,
            "quantiles": list(self.quantiles),
            "median": self.median,
            "over": list(self.over),
            "color": self.color,
            "alpha": self.alpha,
            "yaxis_position": self.yaxis_position,
            "order": self.order,
        }
```

- `Plot`: a `bands` list next to `curves` and `areas` (an `__init__` parameter `bands: list[Band] | None = None`, the property and setter like `areas`, `add_band(band)` which sets `sid = f"{self.sid}_band{len(self.bands)}"` when missing and the order through `_set_order`, which now also looks at the bands), `__copy__` and `to_dict` (`"bands": self.bands`), and

```python
    def band(
        self,
        x: Data,
        y: Data,
        across: str,
        quantiles: tuple[float, float] = (0.05, 0.95),
        median: bool = True,
        over: str | Sequence[str] | None = None,
        name: str | None = None,
        color: str | None = None,
        alpha: float = 0.3,
        yaxis_position: YAxisPosition | None = None,
    ) -> Band:
        """Create a band of the median and the quantiles of y over `across` and add it.

        Returns:
            The band.
        """
        band = Band(x=x, y=y, across=across, quantiles=quantiles, median=median, over=over, name=name, color=color, alpha=alpha, yaxis_position=yaxis_position)
        self.add_band(band)
        return band
```

  Export `Band` from `src/sbmlsim/plot/__init__.py`.

- [ ] **Step 4: Implement the checks at `initialize()`**

In `src/sbmlsim/experiment/experiment.py`:

- `_figure_data` also yields `band.x` and `band.y` of every band of a plot.
- New methods, and `initialize()` calls `self._check_figures()` after `self._check_task_data()`:

```python
    def _scan_dims(self, d: Data) -> set[str] | None:
        """Get the dimensions of the scan of task data after its `sel`.

        A variable or an observable of a scan has every dimension of the scan,
        a coordinate its own dimension, the time none on a common grid and
        every dimension of a ragged scan; a dimension selected by one label is
        gone. Dataset and function data are not known before they are drawn.

        Returns:
            The dimensions, `None` for dataset and function data.
        """
        if not d.is_task():
            return None
        simulation = self._simulations[self._tasks[str(d.task_id)].simulation_id]
        if not isinstance(simulation, Scan):
            return set()
        dims = [dimension.id for dimension in simulation.dimensions]
        kind = self._index_kind(d)
        if kind == "coordinate":
            head = d.selection.partition(".")[0]
            found = (
                {head}
                if head in dims
                else {
                    dimension.id
                    for dimension in simulation.dimensions
                    if d.selection in dimension.coordinates
                }
            )
        elif kind == "time":
            found = set(dims) if _ragged(simulation) else set()
        else:
            found = set(dims)
        single = {
            dim
            for dim, label in d.sel.items()
            if not isinstance(label, list | tuple | np.ndarray)
        }
        return found - single

    def _check_figures(self) -> None:
        """Check the curves and bands of the figures before anything is simulated.

        Every dimension of a scan which the task data of a curve has must be
        its axis (the one dimension of x which is not in `over`), in `over`,
        or selected by one label; a band reduces `across` and needs a common
        grid. Dataset and function data are checked when they are drawn.

        Raises:
            ValueError: for a dimension which is not named, an `over` or
                `across` dimension the data has not, more points of a second
                `over` dimension than line styles, or a band of a ragged scan.
        """
        for key, figure in self._figures.items():
            for plot in figure.get_plots():
                for curve in plot.curves:
                    self._check_lines(key, curve.sid, curve.over, None, curve.x, [curve.y, curve.xerr, curve.yerr])
                for band in plot.bands:
                    self._check_lines(key, band.sid, band.over, band.across, band.x, [band.y])

    def _check_lines(
        self,
        figure: str,
        sid: str | None,
        over: tuple[str, ...],
        across: str | None,
        x: Data,
        others: Sequence[Data | None],
    ) -> None:
        """Check the dimensions of a curve or band, see `_check_figures`."""
        what = "band" if across is not None else "curve"
        xs = self._scan_dims(x)
        ys = [self._scan_dims(d) for d in others if d is not None]
        if xs is None or any(dims is None for dims in ys):
            return
        found = set(xs).union(*[dims for dims in ys if dims is not None])
        missing = [dim for dim in over if dim not in found]
        if missing:
            raise ValueError(
                f"The {what} '{sid}' of the figure '{figure}' draws a line per point of "
                f"{missing}, which its data has not: {sorted(found)}."
            )
        if across is not None:
            task = self._tasks[str(others[0].task_id)] if others[0] is not None else None
            simulation = self._simulations[task.simulation_id] if task else None
            if isinstance(simulation, Scan) and _ragged(simulation):
                raise ValueError(
                    f"The band '{sid}' of the figure '{figure}' reduces '{across}' of a "
                    f"ragged scan, whose simulations keep their own time points; run "
                    f"the scan on a common grid (a simulation with steps or times)."
                )
            if across not in found:
                raise ValueError(
                    f"The band '{sid}' of the figure '{figure}' reduces the dimension "
                    f"'{across}', which its data has not: {sorted(found)}."
                )
        axis = xs - set(over) - {across}
        if len(axis) > 1:
            raise ValueError(
                f"x of the {what} '{sid}' of the figure '{figure}' has the dimensions "
                f"{sorted(axis)} of the scan; name them in over= or select a label "
                f"with Data(sel=...)."
            )
        for dim in sorted(found - set(over) - axis - {across}):
            raise ValueError(
                f"y of {what} '{sid}' of the figure '{figure}' has the dimension "
                f"'{dim}'; name it with over='{dim}' or select a label with "
                f"Data(sel=...)."
            )
        if len(over) == 2:
            task_data = next(d for d in [x, *others] if d is not None and d.is_task())
            labels = _labels(self._simulations[self._tasks[str(task_data.task_id)].simulation_id])
            known = labels.get(over[1])
            if known is not None:
                point_linestyles(len(known))
```

  and the module function

```python
def _ragged(simulation: Scan) -> bool:
    """Check whether the result of a scan keeps the time points of every simulation.

    A scan with dimensions whose simulations output the steps of the
    integrator (neither `steps` nor `times`) is ragged.
    """
    return bool(simulation.dimensions) and any(
        s.steps is None and s.times is None for s in simulation.simulations
    )
```

  Import `point_linestyles` from `sbmlsim.plot.points` (watch for an import cycle: `sbmlsim.plot.points` imports `sbmlsim.data`, `sbmlsim.simulation` and `sbmlsim.result`, not the experiment). The message of the y check uses "y of curve '<sid>'" exactly as the spec quotes it; the wording "of the figure '<figure>'" follows it.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -n 0 tests/plot tests/experiment`
Expected: PASS.

- [ ] **Step 6: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q` and `uv run pytest -q tests/examples/test_example_scripts.py` (the examples draw figures; an example whose curve has an unnamed scan dimension now fails at initialize: name it with `over` or `sel` and report it).

```bash
git add src/sbmlsim/plot/plotting.py src/sbmlsim/plot/__init__.py src/sbmlsim/experiment/experiment.py tests/plot/scan_experiment.py tests/plot/test_scan_figure_model.py tests/experiment/test_experiment_run.py
git commit -m "A curve names the scan dimensions it draws per point and a band reduces a dimension of draws" -m "Curve and Plot.curve take over=, Band and Plot.band describe the median and a quantile band over a dimension, and the experiment checks at initialize from the definitions alone that every dimension of a scan a curve or band reads is its axis, in over or across, or selected; a band of a ragged scan raises."
```

---

### Task 3: Matplotlib draws curves over scan points and bands

**Files:**
- Modify: `src/sbmlsim/plot/serialization_matplotlib.py`
- Modify: `src/sbmlsim/plot/points.py` (`PointStyle`, `point_styles`, shared with Task 4)
- Modify: `src/sbmlsim/experiment/experiment.py` (`scan_dimension`, `model_units`)
- Test: `tests/plot/test_scan_figures_matplotlib.py`, `tests/plot/test_points.py` (`point_styles`)

**Interfaces:**
- Consumes: `curve_lines`, `band_lines`, `point_colors`, `point_colormap`, `point_linestyles`, `point_labels`, `point_values`, `COLORBAR_FROM` (Task 1); `Curve.over`, `Band`, `Plot.bands` (Task 2); `tests/plot/scan_experiment.ScanFigures`.
- Produces:
  - `PointStyle` and `point_styles(lines, over, color, dimensions, units) -> PointStyle` in `sbmlsim.plot.points` (the colours, line styles, labels and colour bar of the lines of a curve or band over scan points).
  - `SimulationExperiment.scan_dimension(task: str, dim: str) -> Dimension | None` (the dimension of the scan of a task by its id) and `SimulationExperiment.model_units(task: str) -> Mapping[str, str]` (the units of the symbols of the model of a task, `{}` for a model which is not loaded).
  - The matplotlib figure of a curve with `over`: one line per point; colours per `point_colors` of the first `over` dimension (the colour of the curve's style if it sets one); line styles per `point_linestyles` of the second; legend labels `"<curve name>, <point label>"` (the point label alone without a name) for a first dimension of fewer than `COLORBAR_FROM` points, else a colour bar of the dimension (label `"<target> [<unit>]"` or the dimension id) and no legend entries of the lines; with two dimensions the second gets legend entries of its line styles in grey. A band draws `fill_between` of the quantiles in its colour with `alpha` and the median as a line of width 2; legend labels `"<name> <low>-<high> %"` for the area and `"<name> median"` for the line.

- [ ] **Step 1: Write the failing tests**

Create `tests/plot/test_scan_figures_matplotlib.py`:

```python
"""Matplotlib draws curves over the points of a scan and bands."""

from pathlib import Path

import numpy as np
import pytest
from matplotlib.colors import to_hex

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.plot import Axis, Figure
from sbmlsim.plot.points import point_colors
from sbmlsim.plot.serialization_matplotlib import MatplotlibFigureSerializer
from sbmlsim.simulator import Simulator
from tests.plot.scan_experiment import ScanFigures


@pytest.fixture(scope="module")
def experiment() -> SimulationExperiment:
    runner = ExperimentRunner(
        experiment_classes=[ScanFigures], simulator=Simulator(), base_path=Path("."), data_path=Path(".")
    )
    experiment = runner.experiments["ScanFigures"]
    experiment.run(runner.simulator)
    return experiment


def _axes(experiment: SimulationExperiment, draw):
    figure = Figure(experiment=experiment, sid="fig", num_rows=1, num_cols=1)
    plot = figure.create_plots(xaxis=Axis("time", unit="hr"), yaxis=Axis("C", unit="mg/l"), legend=True)[0]
    draw(plot)
    fig = MatplotlibFigureSerializer.to_figure(experiment, figure)
    return fig, fig.axes[0]


def _labels(ax) -> list[str]:
    legend = ax.get_legend()
    return [] if legend is None else [t.get_text() for t in legend.get_texts()]


def test_one_line_per_dose_in_viridis_with_labels(experiment: SimulationExperiment) -> None:
    fig, ax = _axes(experiment, lambda p: p.curve(
        x=Data("time", task="task_doses"), y=Data("[C]", task="task_doses"), over="dose", label="C"))
    lines = [line for line in ax.get_lines() if len(line.get_xdata())]
    assert len(lines) == 3
    assert [to_hex(line.get_color()) for line in lines] == point_colors(3, None)
    assert _labels(ax) == ["C, PODOSE = 50 mg", "C, PODOSE = 100 mg", "C, PODOSE = 200 mg"]
    assert len(fig.axes) == 1  # no colour bar


def test_shades_of_the_colour_of_the_curve(experiment: SimulationExperiment) -> None:
    _, ax = _axes(experiment, lambda p: p.curve(
        x=Data("time", task="task_doses"), y=Data("[C]", task="task_doses"), over="dose", color="tab:red"))
    colors = [to_hex(line.get_color()) for line in ax.get_lines() if len(line.get_xdata())]
    assert colors == point_colors(3, "tab:red")


def test_eleven_points_and_more_get_a_colour_bar(experiment: SimulationExperiment) -> None:
    fig, ax = _axes(experiment, lambda p: p.curve(
        x=Data("time", task="task_many"), y=Data("[C]", task="task_many"), over="dose", label="C"))
    assert len([line for line in ax.get_lines() if len(line.get_xdata())]) == 12
    assert _labels(ax) == []
    assert len(fig.axes) == 2 and fig.axes[1].get_ylabel() == "PODOSE [mg]"


def test_two_dimensions_colour_and_line_style(experiment: SimulationExperiment) -> None:
    _, ax = _axes(experiment, lambda p: p.curve(
        x=Data("time", task="task_grid2"), y=Data("[C]", task="task_grid2"), over=("dose", "ke"), label="C"))
    lines = [line for line in ax.get_lines() if len(line.get_xdata())]
    assert len(lines) == 6
    assert {line.get_linestyle() for line in lines} == {"-", "--"}
    assert _labels(ax)[:3] == ["C, PODOSE = 50 mg", "C, PODOSE = 100 mg", "C, PODOSE = 200 mg"]
    unit = experiment.model_units("task_grid2").get("ke", "")
    assert _labels(ax)[3:] == [f"ke = 0.1 {unit}".rstrip(), f"ke = 0.3 {unit}".rstrip()]


def test_two_curves_over_one_dimension_keep_their_names_in_the_legend(experiment: SimulationExperiment) -> None:
    def draw(plot) -> None:
        plot.curve(x=Data("time", task="task_doses"), y=Data("[C]", task="task_doses"), over="dose", label="C")
        plot.curve(x=Data("time", task="task_doses"), y=Data("[C]", task="task_doses"), over="dose", label="C again", color="tab:red")

    _, ax = _axes(experiment, draw)
    labels = _labels(ax)
    assert labels[0].startswith("C, ") and labels[3].startswith("C again, ")


def test_a_value_per_simulation_over_its_dimension(experiment: SimulationExperiment) -> None:
    _, ax = _axes(experiment, lambda p: p.curve(
        x=Data("dose.PODOSE", task="task_doses"), y=Data("pk.cmax", task="task_doses"), label="cmax"))
    (line,) = [line for line in ax.get_lines() if len(line.get_xdata())]
    np.testing.assert_allclose(line.get_xdata(), [50.0, 100.0, 200.0])
    assert np.all(np.diff(line.get_ydata()) > 0)


def test_a_band_with_its_median(experiment: SimulationExperiment) -> None:
    _, ax = _axes(experiment, lambda p: p.band(
        Data("time", task="task_draws"), Data("[C]", task="task_draws"), across="draw", name="C"))
    assert len(ax.collections) == 1
    (median,) = [line for line in ax.get_lines() if len(line.get_xdata())]
    y = Data("[C]", task="task_draws").get_data(experiment, to_units="mg/l")
    np.testing.assert_allclose(median.get_ydata(), np.nanmedian(y.values, axis=0))
    assert _labels(ax) == ["C median", "C 5-95 %"] or _labels(ax) == ["C 5-95 %", "C median"]


def test_a_band_per_dose(experiment: SimulationExperiment) -> None:
    _, ax = _axes(experiment, lambda p: p.band(
        Data("time", task="task_dose_draws"), Data("[C]", task="task_dose_draws"), across="draw", over="dose", name="C"))
    assert len(ax.collections) == 3


def test_a_bar_curve_over_a_dimension_raises(experiment: SimulationExperiment) -> None:
    from sbmlsim.plot.plotting import CurveType

    with pytest.raises(ValueError, match="bar"):
        _axes(experiment, lambda p: p.curve(
            x=Data("time", task="task_doses"), y=Data("[C]", task="task_doses"), over="dose", type=CurveType.BAR))
```

The tests count only lines with data (`len(line.get_xdata())`), because the legend entries of a second dimension are empty lines (`ax.plot([], [])`). Adapt the expected legend order of a band to the order the implementation adds the artists, but keep both labels.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/plot/test_scan_figures_matplotlib.py`
Expected: FAIL (`over` is ignored: `line_values` raises for the dimension `dose`, and bands are not drawn).

- [ ] **Step 3: Implement**

`src/sbmlsim/experiment/experiment.py`:

```python
    def scan_dimension(self, task: str, dim: str) -> Dimension | None:
        """Get a dimension of the scan of a task by its id, `None` if it has none of it."""
        simulation = self._simulations[self._tasks[task].simulation_id]
        if not isinstance(simulation, Scan):
            return None
        return next((d for d in simulation.dimensions if d.id == dim), None)

    def model_units(self, task: str) -> Mapping[str, str]:
        """Get the units of the symbols of the model of a task, `{}` for a model which is not loaded."""
        model = self._models.get(self._tasks[task].model_id)
        uinfo = getattr(model, "uinfo", None)
        return dict(uinfo) if uinfo is not None else {}
```

(Check how `UnitsInformation` converts to a mapping and adapt `dict(uinfo)`.)

`src/sbmlsim/plot/serialization_matplotlib.py`: import from `sbmlsim.plot.points` and `matplotlib.cm.ScalarMappable`, `matplotlib.colors.Normalize`; `abstract_curves` includes `plot.bands`; the drawing of a `Curve` becomes:

```python
                if isinstance(abstract_curve, Curve):
                    curve: Curve = abstract_curve
                    x = curve.x.get_data(experiment=experiment, to_units=xunit)
                    y = curve.y.get_data(experiment=experiment, to_units=yunit)
                    xerr = None if curve.xerr is None else curve.xerr.get_data(experiment=experiment, to_units=xunit)
                    yerr = None if curve.yerr is None else curve.yerr.get_data(experiment=experiment, to_units=yunit)
                    sid = curve.sid or curve.name or ""
                    lines = curve_lines(sid, curve.over, x, y, xerr, yerr)
                    kwargs = cls._curve_kwargs(curve)
                    if curve.over and curve.type != CurveType.POINTS:
                        raise ValueError(
                            f"The curve '{sid}' is a bar curve, which draws no line "
                            f"per point of {list(curve.over)}; draw points or select a label."
                        )
                    if not curve.over:
                        (line,) = lines
                        cls._draw_curve(ax, curve, line, kwargs, curve.name or "_nolegend_", stacks)
                        continue
                    cls._draw_points(fig, ax, experiment, curve.name, curve.over, curve.y, lines, kwargs, cls._style_color(curve.style))
```

and the helpers (methods of the serializer):

- `_curve_kwargs(curve)`: the existing kwargs of the style (`to_mpl_points_kwargs` / `to_mpl_bar_kwargs`).
- `_style_color(style)`: the colour of the line of the resolved style, `None` if it sets none.
- `_draw_curve(ax, curve, line, kwargs, label, stacks)`: the existing code of the curve types (plot, errorbar, bar, barh, stacked bars) on `line.x`, `line.y`, `line.xerr`, `line.yerr`; the four stack variables move into a small dict `stacks` of the plot.
- `_draw_points` and `_colorbar`, which both serializers share through two functions of `sbmlsim.plot.points` (add them there, with tests in `tests/plot/test_points.py`):

```python
@dataclass(frozen=True)
class PointStyle:
    """How the lines of a curve over scan points look, see `point_styles`.

    Attributes:
        colors: the colour of every point of the first dimension.
        linestyles: the line style of every point of the second, `None` with one.
        labels: the legend label of every point of every dimension.
        colorbar: whether the first dimension gets a colour bar.
        values: the values of the colour bar, the positions without a target.
        title: the label of the colour bar, `<target> [<unit>]` or the dimension.
    """

    colors: list[str]
    linestyles: list[str] | None
    labels: list[list[str]]
    colorbar: bool
    values: np.ndarray
    title: str


def point_styles(
    lines: Sequence[Line | BandLine],
    over: Sequence[str],
    color: str | None,
    dimensions: Sequence[Dimension | None],
    units: Mapping[str, str],
) -> PointStyle:
    """Get the colours, line styles, labels and colour bar of the lines of a curve.

    Raises:
        ValueError: for more points of a second dimension than line styles.
    """
    labels_of = [list(dict.fromkeys(line.point[k] for line in lines)) for k in range(len(over))]
    labels = [point_labels(d, labels_of[k], dimensions[k], units) for k, d in enumerate(over)]
    n = len(labels_of[0])
    found = point_values(dimensions[0], units)
    if found is None:
        values, title = np.arange(n, dtype=float), over[0]
    else:
        values, title = found[1], f"{found[0]} [{found[2]}]" if found[2] else found[0]
    return PointStyle(
        colors=point_colors(n, color),
        linestyles=point_linestyles(len(labels_of[1])) if len(over) == 2 else None,
        labels=labels,
        colorbar=n >= COLORBAR_FROM,
        values=values,
        title=title,
    )
```

  In the matplotlib serializer:

```python
    @classmethod
    def _draw_points(
        cls,
        fig: FigureMPL,
        ax: AxesMPL,
        experiment: Any,
        curve: Curve,
        lines: list[Line],
        kwargs: dict[str, Any],
    ) -> None:
        """Draw the lines of a curve over scan points, with legend or colour bar."""
        task = curve.y.task_id or curve.x.task_id
        dimensions = [experiment.scan_dimension(task, d) if task else None for d in curve.over]
        units = experiment.model_units(task) if task else {}
        styles = point_styles(lines, curve.over, cls._style_color(curve.style), dimensions, units)
        prefix = f"{curve.name}, " if curve.name else ""
        labelled = len(curve.over) == 1 and not styles.colorbar
        for line in lines:
            kw = dict(kwargs)
            kw["color"] = styles.colors[line.index[0]]
            if "markerfacecolor" in kw:
                kw["markerfacecolor"] = kw["color"]
            if styles.linestyles is not None:
                kw["linestyle"] = styles.linestyles[line.index[1]]
            label = f"{prefix}{styles.labels[0][line.index[0]]}" if labelled else "_nolegend_"
            cls._draw_line(ax, line, kw, label)
        if styles.linestyles is not None:
            if not styles.colorbar:
                for c, text in zip(styles.colors, styles.labels[0], strict=True):
                    ax.plot([], [], color=c, label=f"{prefix}{text}")
            for ls, text in zip(styles.linestyles, styles.labels[1], strict=True):
                ax.plot([], [], color="0.4", linestyle=ls, label=text)
        if styles.colorbar:
            cls._colorbar(fig, ax, styles, cls._style_color(curve.style))

    @classmethod
    def _colorbar(cls, fig: FigureMPL, ax: AxesMPL, styles: PointStyle, color: str | None) -> None:
        """Draw the colour bar of the first dimension of a curve over scan points."""
        norm = Normalize(float(np.min(styles.values)), float(np.max(styles.values)))
        mappable = ScalarMappable(norm=norm, cmap=point_colormap(color))
        fig.colorbar(mappable, ax=ax, label=styles.title)
```

  `_draw_line(ax, line, kwargs, label)` is the POINTS branch of today (`ax.plot`, or `ax.errorbar` with errors), which `_draw_curve` also uses.

- The drawing of a `Band`:

```python
                elif isinstance(abstract_curve, Band):
                    band: Band = abstract_curve
                    x = band.x.get_data(experiment=experiment, to_units=xunit)
                    y = band.y.get_data(experiment=experiment, to_units=yunit)
                    sid = band.sid or band.name or ""
                    bands = band_lines(sid, band.over, band.across, band.quantiles, x, y)
                    cls._draw_bands(fig, ax, experiment, band, bands)
```

```python
    @classmethod
    def _draw_bands(
        cls, fig: FigureMPL, ax: AxesMPL, experiment: Any, band: Band, bands: list[BandLine]
    ) -> None:
        """Draw the quantile areas and medians of a band, one per point of `over`."""
        low, high = (f"{100 * q:g}" for q in band.quantiles)
        styles = None
        if band.over:
            task = band.y.task_id
            dimension = experiment.scan_dimension(task, band.over[0]) if task else None
            units = experiment.model_units(task) if task else {}
            styles = point_styles(bands, band.over, band.color, [dimension], units)
        for b in bands:
            color = styles.colors[b.index[0]] if styles else (band.color or "C0")
            name = f"{band.name}, {styles.labels[0][b.index[0]]}" if styles else band.name
            show = styles is None or not styles.colorbar
            ax.fill_between(
                b.x, b.low, b.high, color=color, alpha=band.alpha, linewidth=0,
                label=f"{name} {low}-{high} %" if show else "_nolegend_",
            )
            if band.median:
                ax.plot(b.x, b.median, color=color, linewidth=2.0, label=f"{name} median" if show else "_nolegend_")
        if styles is not None and styles.colorbar:
            cls._colorbar(fig, ax, styles, band.color)
```

Add to `tests/plot/test_points.py` a test of `point_styles`: three lines over `dose` with a `Dimension("dose", values={"PODOSE": Q([50.0, 100.0, 200.0], "mg")})` give `colors == point_colors(3, None)`, `linestyles is None`, `labels[0] == ["PODOSE = 50 mg", "PODOSE = 100 mg", "PODOSE = 200 mg"]`, `colorbar is False`, `title == "PODOSE [mg]"`; twelve points give `colorbar is True`; two dimensions give `linestyles == ["-", "--"]` for two points of the second.

- [ ] **Step 4: Run the tests to verify they pass, render and look**

Run: `uv run pytest -q -n 0 tests/plot`
Expected: PASS.

Render the figures of the tests into a scratch directory (a short script under `/tmp/claude-1000/` which builds the experiment of `tests/plot/scan_experiment.py`, draws the eight cases and saves them as png) and look at every one with the Read tool: colours distinguishable, legend not covering the lines (use `legend_position="outside"` of the figure if it does), the colour bar labelled with value and unit, the band translucent with its median on top.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/plot/serialization_matplotlib.py src/sbmlsim/experiment/experiment.py tests/plot/test_scan_figures_matplotlib.py
git commit -m "Matplotlib draws a curve per point of the scan dimensions it names and bands of draws" -m "A curve over one dimension draws its lines in shades of its colour or in viridis with the value and unit of the changed target in the legend, from eleven points with a colour bar instead; a second dimension sets the line style. A band draws the area between two quantiles and the median."
```

---

### Task 4: Plotly draws the same

**Files:**
- Modify: `src/sbmlsim/plot/serialization_plotly.py`
- Test: `tests/plot/test_scan_figures_plotly.py`

**Interfaces:**
- Consumes: Task 1, Task 2, `SimulationExperiment.scan_dimension` and `model_units` (Task 3).
- Produces: the plotly figure of a curve with `over`: one `Scatter` per line with the colour of `point_colors`, `line.dash` per line style of the second dimension (`"solid"`, `"dash"`, `"dot"`, `"dashdot"` for `LINE_STYLES`), `name` the label of the matplotlib legend, `legendgroup` the curve sid, `showlegend` as in matplotlib (no entries of the lines from `COLORBAR_FROM` points); a colour bar from `COLORBAR_FROM` points as an invisible `Scatter` (`x=[None]`, `y=[None]`, `marker={"color": values, "colorscale": <sampled point_colormap>, "showscale": True, "colorbar": {"title": label}}`, `showlegend=False`); a band as two traces of the quantiles (`fill="tonexty"` on the upper, `fillcolor` the colour with the alpha as `rgba`) and a median line.

- [ ] **Step 1: Write the failing tests**

Create `tests/plot/test_scan_figures_plotly.py`:

```python
"""Plotly draws curves over the points of a scan and bands like matplotlib."""

from pathlib import Path

import pytest

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.plot import Axis, Figure
from sbmlsim.plot.points import point_colors
from sbmlsim.simulator import Simulator
from tests.plot.scan_experiment import ScanFigures

pytest.importorskip("plotly")

from sbmlsim.plot.serialization_plotly import PlotlyFigureSerializer  # noqa: E402


@pytest.fixture(scope="module")
def experiment() -> SimulationExperiment:
    runner = ExperimentRunner(
        experiment_classes=[ScanFigures], simulator=Simulator(), base_path=Path("."), data_path=Path(".")
    )
    experiment = runner.experiments["ScanFigures"]
    experiment.run(runner.simulator)
    return experiment


def _traces(experiment: SimulationExperiment, draw) -> list:
    figure = Figure(experiment=experiment, sid="fig", num_rows=1, num_cols=1)
    plot = figure.create_plots(xaxis=Axis("time", unit="hr"), yaxis=Axis("C", unit="mg/l"), legend=True)[0]
    draw(plot)
    return list(PlotlyFigureSerializer.to_figure(experiment, figure).data)


def test_one_trace_per_dose(experiment: SimulationExperiment) -> None:
    traces = _traces(experiment, lambda p: p.curve(
        x=Data("time", task="task_doses"), y=Data("[C]", task="task_doses"), over="dose", label="C"))
    assert [t.name for t in traces] == ["C, PODOSE = 50 mg", "C, PODOSE = 100 mg", "C, PODOSE = 200 mg"]
    assert [t.line.color for t in traces] == point_colors(3, None)


def test_eleven_points_and_more_get_a_colour_bar(experiment: SimulationExperiment) -> None:
    traces = _traces(experiment, lambda p: p.curve(
        x=Data("time", task="task_many"), y=Data("[C]", task="task_many"), over="dose", label="C"))
    lines = [t for t in traces if t.x is not None and t.x[0] is not None]
    assert len(lines) == 12 and not any(t.showlegend for t in lines)
    (bar,) = [t for t in traces if t not in lines]
    assert bar.marker.showscale and bar.marker.colorbar.title.text == "PODOSE [mg]"


def test_two_dimensions_dash(experiment: SimulationExperiment) -> None:
    traces = _traces(experiment, lambda p: p.curve(
        x=Data("time", task="task_grid2"), y=Data("[C]", task="task_grid2"), over=("dose", "ke"), label="C"))
    assert {t.line.dash for t in traces if t.x is not None and len(t.x)} >= {"solid", "dash"}


def test_a_band(experiment: SimulationExperiment) -> None:
    traces = _traces(experiment, lambda p: p.band(
        Data("time", task="task_draws"), Data("[C]", task="task_draws"), across="draw", name="C"))
    assert [t.fill for t in traces].count("tonexty") == 1
    assert any(t.name == "C median" for t in traces)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -n 0 tests/plot/test_scan_figures_plotly.py`
Expected: FAIL.

- [ ] **Step 3: Implement**

`_add_subplot` iterates `plot.curves + plot.areas + plot.bands` and adds every trace of `cls._traces(experiment, abstract_curve, xunit, yunit) -> list[Any]` (the former `_trace`; an area and a curve without `over` give one trace as before). The new parts:

```python
#: the dash of plotly of every line style of `sbmlsim.plot.points.LINE_STYLES`
DASH_BY_LINESTYLE = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}


def _rgba(color: str, alpha: float) -> str:
    """Get a colour with an opacity as the rgba string of plotly."""
    r, g, b = (round(255 * c) for c in to_rgb(color))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _colorscale(colormap: Colormap) -> list[list[Any]]:
    """Sample a colour map of matplotlib into a colour scale of plotly."""
    return [[float(f), to_hex(colormap(float(f)))] for f in np.linspace(0.0, 1.0, 11)]


def _colorbar_trace(styles: PointStyle, color: str | None) -> Any:
    """Get the invisible trace which shows the colour bar of a dimension."""
    import plotly.graph_objects as go

    return go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        showlegend=False,
        hoverinfo="skip",
        marker={
            "color": [float(np.min(styles.values)), float(np.max(styles.values))],
            "colorscale": _colorscale(point_colormap(color)),
            "showscale": True,
            "colorbar": {"title": {"text": styles.title}},
        },
    )
```

  A curve with `over`: `lines = curve_lines(...)`, `styles = point_styles(lines, curve.over, <colour of the style>, dimensions, units)` (as in Task 3), and per line a `go.Scatter(x=line.x, y=line.y, name=..., legendgroup=curve.sid, showlegend=<as in matplotlib>, mode=_mode(style), line={**_line_options(style), "color": styles.colors[line.index[0]], **({"dash": DASH_BY_LINESTYLE[styles.linestyles[line.index[1]]]} if styles.linestyles else {})}, marker={**_marker_options(style), "color": styles.colors[line.index[0]]}, error_x=..., error_y=...)`; the names are the labels of Task 3 (`"<curve name>, <label of the first dimension>"`, with a second dimension `"<curve name>, <first>, <second>"` and `showlegend=True` for every line, since plotly has no proxy entries); from `COLORBAR_FROM` points `showlegend=False` for the lines and `_colorbar_trace(styles, colour)` is added. A band: per `BandLine` a lower trace (`y=b.low`, `line={"width": 0}`, `showlegend=False`, `hoverinfo="skip"`), an upper trace (`y=b.high`, `fill="tonexty"`, `fillcolor=_rgba(colour, band.alpha)`, `line={"width": 0}`, `name="<name> <low>-<high> %"`) and, with `median`, a line trace (`name="<name> median"`, `line={"color": colour, "width": 2}`); colours and names per point as in Task 3. A bar curve with `over` raises the error of Task 3 before the check of the curve type.

- [ ] **Step 4: Run the tests to verify they pass, render and look**

Run: `uv run pytest -q -n 0 tests/plot`
Write the html of the eight cases into a scratch directory and open one in a headless browser screenshot if `chrome-devtools-axi` is available, else inspect the trace structure; record what was checked.

- [ ] **Step 5: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`

```bash
git add src/sbmlsim/plot/serialization_plotly.py tests/plot/test_scan_figures_plotly.py
git commit -m "Plotly draws curves over the points of a scan and bands like matplotlib" -m "A curve with over gives one trace per point with the colours, dashes and legend names of the matplotlib figure and a colour bar from eleven points; a band gives a filled pair of quantile traces and the median."
```

---

### Task 5: The figures of the examples

**Files:**
- Modify: `examples/demo/demo.py` (a second figure over `dim_init`)
- Modify: `examples/glucose/experiments/dose_response.py` (observables over the dose dimension instead of `figures_mpl`)
- Modify: `examples/hctz_fitting/experiments/studies/patel1984.py` (a dose scan with a PK observable and a figure of cmax and AUC over the dose)
- Modify: `examples/experiment_scans.py` (figures: curves per dose, cmax over dose, a band over LHS draws)
- Modify: `examples/README.md` (the rows of the changed examples)

**Interfaces:**
- Consumes: Tasks 1 to 4.

- [ ] **Step 1: The demo**

In `examples/demo/demo.py`, keep `fig1` as it is and add `fig2` with two panels, `[e__A]` and `[c__A]` over time with `over="dim_init"` and `sel={"dim_sens": "reference"}` on both data, `legend=True`; eleven initial values give a colour bar labelled `[e__A] [mM]`. Return `{"fig1": fig1, "fig2": fig2}`.

- [ ] **Step 2: The glucose dose response**

In `examples/glucose/experiments/dose_response.py`, replace `figures_mpl` and its `DataSet` workaround by:

- `observables()`: `{f"{sid}_0": Formula(f"{sid}_0", f"at({sid}, 0)") for sid in SELECTIONS}` (the hormones at the start of every simulation, which are assignment rules of the glucose);
- `data()`: registers nothing for the figures (the figure data counts), keep the selections the experiment still reads;
- `figures()`: a `Figure` of 2 x 2 panels with the same axes, limits and labels as the old matplotlib figure (x `Axis("glucose", unit="mM")` from 2 to 20, y per panel as in `panels`), the simulation as `plot.curve(x=Data("dim1.[glc_ext]", task="task_glc_scan"), y=Data(f"{sid}_0", task="task_glc_scan"), color="black", linewidth=2, label="simulation")` and the data with `plot.add_data(dataset=label, xid="glc", yid="mean", yid_se="mean_se", ...)` as before.

Remove the imports which become unused (`pandas`, `matplotlib.pyplot`, `add_data` of the matplotlib helpers if no longer used). Render the figure before and after (develop vs the branch) from scratch directories and compare: the curves and data points must agree.

- [ ] **Step 3: HCTZ, Patel 1984**

In `examples/hctz_fitting/experiments/studies/patel1984.py`:

- `simulations()`: add `"hctz_doses": Scan(Simulation(time_unit="hr", end=50, steps=500, preinit_changes=self.default_changes(), changes=[Change(0, {"PODOSE_hctz": Q(25, "mg")})]), [Dimension("dose", values={"PODOSE_hctz": Q(self.doses[1:], "mg")})])` (the doses without the zero dose, whose PK parameters are undefined); the type of the return becomes `dict[str, Simulation | Scan]`;
- `observables()`: `{"hctz": PK("hctz", "[Cve_hctz]", dose="PODOSE_hctz", route="oral")}`;
- `figures()`: a figure `Fig_pk` of two panels, cmax and AUC (`hctz.cmax`, `hctz.auc_inf_obs`) over `Data("dose.PODOSE_hctz", task="task_hctz_doses")`, with axes named and with units (`Axis("dose", unit="mg")`, the units of the PK parameters converted to `mM` and `mM*hr`), markers on the line.

The task `task_hctz_doses` has no fit mapping, so the fit problems built from the study are unchanged: run `uv run pytest -q tests/fit -x` and compare the cost of the HCTZ problem before and after (the tests of the fit check it).

- [ ] **Step 4: `examples/experiment_scans.py`**

Add to `MidazolamDoses`:

- a second simulation `"draws"`: `Scan(simulation, [sampling.lhs({"LI__MIDIM_Vmax": sampling.LogNormal(cv=0.3), "Ka_abs_mid": sampling.LogNormal(cv=0.3)}, 40, seed=1, model=self._models["model"], id="draw")])` and its task `task_draws`;
- `figures()`: `Fig1` with three panels: `mid` over time with `over="dose"`; `pk.cmax` over `Data("dose.PODOSE_mid", task="task_doses")`; a band of `mid` over the draws (`plot.band(Data("time", task="task_draws"), Data("mid", task="task_draws"), across="draw", name="mid")`).

The docstring of the module names the figures. Keep the printed output of phase 1.

- [ ] **Step 5: Render and look**

Run every changed example from a scratch directory (`cd $(mktemp -d -p /tmp/claude-1000) && PYTHONPATH=/home/mkoenig/git/sbmlsim uv run --project /home/mkoenig/git/sbmlsim python -W error -m examples.<module>`), convert the svg figures with `uvx --quiet cairosvg in.svg -o out.png`, and look at every figure with the Read tool. Fix what looks off.

- [ ] **Step 6: Lint, types, all tests, commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q && uv run pytest -q tests/examples/test_example_scripts.py`

```bash
git add examples/demo/demo.py examples/glucose/experiments/dose_response.py examples/hctz_fitting/experiments/studies/patel1984.py examples/experiment_scans.py examples/README.md
git commit -m "The examples draw curves over the points of their scans, values over doses and bands" -m "The demo draws the initial values of A as one curve each with a colour bar, the glucose dose response draws the hormones as observables over the dose dimension instead of a matplotlib figure of its own, Patel 1984 draws cmax and AUC of HCTZ over the dose, and the midazolam example draws its doses, cmax over dose and a band over Latin hypercube draws."
```

---

### Task 6: The docs and CLAUDE.md

**Files:**
- Modify: `docs/plotting.md` (sections "Curves over the points of a scan" and "Bands"), `docs/experiments.md` (a sentence), `CLAUDE.md`

- [ ] **Step 1: docs/plotting.md**

After the section "Curves" add:

```markdown
## Curves over the points of a scan

A curve of task data draws one line. When the data has dimensions of a scan, the curve names the ones it draws one line per point of, `plot.curve(x=Data("time", task="task_doses"), y=Data("mid", task="task_doses"), over="dose")`; every other dimension of the scan is selected with `Data(sel=...)`, and a dimension which is neither the axis, in `over` nor selected raises when the experiment is initialized. The lines take shades of the colour of the curve, or the colour map viridis without one, along the labels of the dimension; a second dimension, `over=("dose", "condition")`, sets the line style (solid, dashed, dotted, dash-dot, so at most four points). A legend entry names the point by the value and unit of the target its dimension changes, e.g. `PODOSE_mid = 5 mg`, else by its label; from eleven points a colour bar of the dimension replaces the legend entries. A value per simulation is drawn over the values a dimension sets: `plot.curve(x=Data("dose.PODOSE_mid", task="task_doses"), y=Data("pk.cmax", task="task_doses"))`.

## Bands

`plot.band(x, y, across="draw", quantiles=(0.05, 0.95), median=True)` reduces the dimension `across` of y, e.g. the draws of a Latin hypercube design (see [Sampling and uncertainty](sampling.md)), to two quantiles, ignoring `NaN`, and draws the area between them with the median as a line; `over=` gives one band per point of another dimension. The quantiles are computed when the figure is drawn, so no summary of the result is needed; a band needs a common grid of times, a ragged scan raises. `examples/experiment_scans.py` draws curves per dose, cmax over dose and a band over draws.
```

`docs/experiments.md`, section "Observables and scans": replace "a curve draws one line, so the point of a scan it shows is selected with `Data(sel=...)`" by "a curve draws one line per point of the dimensions it names with `over=`, the others are selected with `Data(sel=...)`, see [Plots and reports](plotting.md)".

- [ ] **Step 2: CLAUDE.md**

The user approved this plan, which includes these edits; change nothing else. In the paragraph of `experiment/`, `data.py`, `task/`, `plot/`, `report/`, replace "A curve draws one line of its broadcast arrays (`plot/padding.line_values`, without the padding) and raises for a dimension of a scan which is not selected." by "A curve names the dimensions of a scan it draws one line per point of (`over=`) and a band (`Plot.band`, `Band`) reduces a dimension of draws to two quantiles and the median; `plot/points.py` splits the broadcast arrays into the lines (`curve_lines`, `band_lines`, without the padding) and gives their colours (shades of the colour of the curve or viridis, a second dimension sets the line style), legend labels (the value and unit of the changed target) and from eleven points a colour bar, which both serializers draw; `initialize()` checks from the definitions that every dimension of a scan a curve or band reads is its axis, in `over`/`across` or selected, and a band of a ragged scan raises." and in `plot/plotting.py is the backend independent figure model (...)` add `Band` to the list of classes.

- [ ] **Step 3: Verify and commit**

Run: `uv run zensical build --clean` (no warnings), `uv run pytest -q -n 0 tests/docs`.

```bash
git add docs/plotting.md docs/experiments.md CLAUDE.md
git commit -m "The docs of curves over the points of a scan and of bands" -m "docs/plotting.md describes over, the colours, legends and colour bars and Plot.band, docs/experiments.md points to it, and CLAUDE.md follows."
```

---

### Task 7: Verification and the pull request

**Files:** none new; the pull request.

- [ ] **Step 1: The whole suite and the checks**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`, `uv run pytest -q tests/examples/test_example_scripts.py`, `uv run zensical build --clean`, and render the figures of the examples once more from a scratch directory.

- [ ] **Step 2: The pull request**

Push and create the pull request with `gh-axi`, base `develop`, title "Figures over scan dimensions: curves per point, values over doses and bands (#249, experiments phase 2)". The body describes, without any agent attribution: `over=` with colours, legends and colour bars, `Plot.band`, the checks at initialize, both backends, the examples, the decisions of this plan, and what phase 3 brings. Wait for the checks and fix any failure; if `develop` moved and the pull request conflicts, merge `develop` into the branch (no force push).
