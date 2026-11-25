"""Tests of the chart builder of the figure contract `figure/1`."""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pint
import pytest
import xarray as xr

from sbmlsim.plot import Chart, ChartAxis
from sbmlsim.plot.chart import BAND_OPACITY, axis_label
from sbmlsim.units import Q

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_spec_is_figure_1() -> None:
    """A chart of one plot with one curve is a spec of `figure/1`."""
    chart = Chart(400, 300, title="decay")
    plot = chart.plot(x=ChartAxis(label="time"), y=ChartAxis(label="A"))
    element = plot.curve([0, 1, 2], [1.0, 0.5, 0.25], label="A")

    spec = chart.spec()
    assert spec["contract"] == "figure/1"
    assert spec["width"] == 400
    assert spec["height"] == 300
    assert spec["title"] == "decay"
    assert spec["grid"] == {"rows": 1, "columns": 1, "wspace": 0.3, "hspace": 0.3}
    assert len(spec["plots"]) == 1
    p = spec["plots"][0]
    assert (p["row"], p["column"]) == (0, 0)
    assert p["x"]["label"] == "time"
    assert "unit" not in p["x"]
    assert "y2" not in p
    (curve,) = p["elements"]
    assert element == curve["id"] == "e0"
    assert curve["kind"] == "curve"
    assert curve["label"] == "A"
    assert curve["x"] == [0.0, 1.0, 2.0]
    assert curve["y"] == [1.0, 0.5, 0.25]
    assert "xerr" not in curve


def test_nan_is_null() -> None:
    """A NaN or an infinite value is a gap, `null` in the JSON."""
    chart = Chart()
    plot = chart.plot()
    plot.curve([0, 1, 2], [1.0, math.nan, math.inf])

    text = chart.to_json()
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text)["plots"][0]["elements"][0]["y"] == [1.0, None, None]


def test_axis_label() -> None:
    """The label of an axis is its name and the short form of its unit."""
    assert axis_label("time", "hr") == "time [h]"
    assert axis_label("time", Q(1, "hr").units) == "time [h]"
    assert axis_label("ratio", "dimensionless") == "ratio"
    assert axis_label("ratio", None) == "ratio"


def test_quantities_set_unit_and_label() -> None:
    """A quantity on an axis without a unit gives the axis its unit."""
    pytest.importorskip("sbmlsim_core")
    chart = Chart()
    plot = chart.plot(x=ChartAxis(label="time"))
    plot.curve(Q([1, 2], "hr"), [3.0, 4.0])

    x = chart.spec()["plots"][0]["x"]
    assert x["label"] == "time"
    assert x["unit"] == "h"
    assert axis_label(x["label"], x["unit"]) == "time [h]"
    assert "time [h]" in chart.svg()


def test_quantity_converted_to_axis_unit() -> None:
    """A quantity on an axis with a unit is converted to that unit."""
    chart = Chart()
    plot = chart.plot(x=ChartAxis(label="time", unit="hr"), y=ChartAxis(unit="mM"))
    plot.curve(Q([30, 90], "min"), Q([500.0, 1000.0], "µM"))
    plot.curve([1.0, 2.0], [0.1, 0.2])

    spec = chart.spec()["plots"][0]
    first, second = spec["elements"]
    assert first["x"] == [0.5, 1.5]
    assert first["y"] == pytest.approx([0.5, 1.0])
    # plain values on an axis with a unit are in that unit
    assert second["x"] == [1.0, 2.0]
    assert spec["y"]["unit"] == "mM"


def test_quantity_of_another_dimension_refused() -> None:
    """A quantity of a dimension the axis cannot take names the axis."""
    chart = Chart()
    plot = chart.plot(x=ChartAxis(unit="hr"))
    with pytest.raises(pint.DimensionalityError, match="axis x"):
        plot.curve(Q([1, 2], "mM"), [1.0, 2.0])

    plot = Chart().plot()
    plot.curve([0, 1], Q([1, 2], "mM"))
    with pytest.raises(pint.DimensionalityError, match="axis y"):
        plot.curve([0, 1], Q([1, 2], "s"))


def test_dataarray_units() -> None:
    """A data array is a quantity of `attrs["units"]`."""
    chart = Chart()
    plot = chart.plot(y=ChartAxis(label="cp", unit="mM"))
    time = xr.DataArray(np.array([0.0, 60.0]), attrs={"units": "minute"})
    cp = xr.DataArray(np.array([1.0, 2.0]), attrs={"units": "µM"})
    plain = xr.DataArray(np.array([3.0, 4.0]))
    plot.curve(time, cp)
    plot.curve(plain, plain)

    spec = chart.spec()["plots"][0]
    assert spec["x"]["unit"] == "min"
    assert spec["elements"][0]["x"] == [0.0, 60.0]
    assert spec["elements"][0]["y"] == pytest.approx([1e-3, 2e-3])
    assert spec["elements"][1]["y"] == [3.0, 4.0]


def test_svg_and_png() -> None:
    """A chart renders as SVG and as PNG."""
    pytest.importorskip("sbmlsim_core")
    chart = Chart()
    chart.plot(x=ChartAxis(label="x"), y=ChartAxis(label="y")).curve(
        [0, 1, 2], [0, 1, 4], marker="circle"
    )

    root = ET.fromstring(chart.svg())
    assert root.tag.endswith("svg")
    assert chart._repr_svg_() == chart.svg()
    assert chart.png().startswith(PNG_SIGNATURE)
    assert len(chart.png(dpi=72.0)) < len(chart.png(dpi=288.0))


def test_save_by_suffix(tmp_path: Path) -> None:
    """A chart is saved as SVG, PNG or JSON by the suffix of its path."""
    pytest.importorskip("sbmlsim_core")
    chart = Chart()
    chart.plot().curve([0, 1], [0, 1])

    svg = chart.save(tmp_path / "chart.svg")
    assert svg == tmp_path / "chart.svg"
    ET.parse(svg)
    png = chart.save(str(tmp_path / "chart.png"), dpi=72.0)
    assert png.read_bytes().startswith(PNG_SIGNATURE)
    spec = chart.save(tmp_path / "chart.json")
    assert json.loads(spec.read_text()) == chart.spec()
    with pytest.raises(ValueError, match=r"\.pdf"):
        chart.save(tmp_path / "chart.pdf")


def test_duplicate_id_refused() -> None:
    """An element id is unique within the chart."""
    chart = Chart(rows=1, columns=2)
    first = chart.plot(0, 0)
    second = chart.plot(0, 1)
    first.curve([0], [0], id="a")
    with pytest.raises(ValueError, match="'a'"):
        second.curve([0], [0], id="a")
    # a default id skips an id which was given
    first.curve([0], [0], id="e1")
    assert first.curve([0], [0]) == "e2"
    assert second.curve([0], [0]) == "e3"


def test_every_kind() -> None:
    """Every kind of element builds and renders."""
    pytest.importorskip("sbmlsim_core")
    chart = Chart(900, 900, rows=4, columns=4)
    plots = iter(chart.plot(i // 4, i % 4) for i in range(16))

    kinds = {
        "curve": next(plots).curve(
            [0, 1, 2], [1, 2, 3], xerr=[0.1] * 3, yerr=[0.2] * 3, marker="square"
        ),
        "band": next(plots).band([0, 1, 2], [0, 1, 2], [2, 3, 4], median=[1, 2, 3]),
        "area": next(plots).area([0, 1, 2], [0, 0, 0], [1, 2, 1]),
        "bar": next(plots).bar([0, 1], [3, 4], err=[0.5, 0.5], bar_width=0.5),
        "box": next(plots).box(
            [0, 1],
            [
                {"low_whisker": 0, "q1": 1, "median": 2, "q3": 3, "high_whisker": 4},
                {
                    "low_whisker": 1,
                    "q1": 2,
                    "median": 3,
                    "q3": 4,
                    "high_whisker": 5,
                    "outliers": [7.0],
                },
            ],
        ),
        "violin": next(plots).violin(
            [0], [{"values": [0, 1, 2], "density": [0.1, 0.5, 0.1]}]
        ),
        "histogram": next(plots).histogram([0, 1, 2, 3], [1, 4, 2]),
        "heatmap": next(plots).heatmap(
            ["a", "b"], ["c", "d"], [[0, 1], [2, 3]], text=[["0", "1"], ["2", "3"]]
        ),
        "text": next(plots).text(0.5, 0.5, "hello", anchor="middle"),
        "rule": next(plots).rule(1.0, orientation="vertical"),
        "polygon": next(plots).polygon([[[0, 0], [1, 0], [0, 1]]]),
    }
    horizontal = next(plots)
    horizontal.bar([0, 1], Q([1, 2], "hr"), horizontal=True)
    colored = next(plots)
    assert colored.colorbar("magma", 0.0, 1.0, label="value") is None

    spec = chart.spec()
    elements = {e["id"]: e for p in spec["plots"] for e in p["elements"]}
    for kind, element in kinds.items():
        assert elements[element]["kind"] == kind
    assert elements[kinds["area"]]["from"] == [0.0, 0.0, 0.0]
    assert elements[kinds["bar"]]["width"] == 0.5
    assert elements[kinds["heatmap"]]["scale"] == {
        "name": "viridis",
        "low": 0.0,
        "high": 3.0,
        "log": False,
    }
    # a horizontal bar has its heights on the x axis
    assert spec["plots"][11]["x"]["unit"] == "h"
    assert spec["plots"][12]["colorbar"] == {
        "scale": {"name": "magma", "low": 0.0, "high": 1.0, "log": False},
        "label": "value",
    }
    ET.fromstring(chart.svg())


def test_grid_and_spans() -> None:
    """Plots are placed in the grid of the chart with their spans."""
    pytest.importorskip("sbmlsim_core")
    chart = Chart(rows=2, columns=3, wspace=0.1, hspace=0.2, legend_scope="row")
    chart.plot(0, 0, column_span=3, title="top", legend_order=["b", "a"])
    chart.plot(
        1, 1, row_span=1, column_span=2, y2=ChartAxis(label="right"), legend=False
    ).curve([0, 1], [1, 2], axis="y2", label="r")

    spec = chart.spec()
    assert spec["grid"] == {"rows": 2, "columns": 3, "wspace": 0.1, "hspace": 0.2}
    assert spec["legend"] == {"scope": "row", "position": "inside"}
    top, bottom = spec["plots"]
    assert top["id"] != bottom["id"]
    assert (top["row"], top["column"], top["column_span"]) == (0, 0, 3)
    assert top["title"] == "top"
    assert top["legend_order"] == ["b", "a"]
    assert (bottom["row"], bottom["column"], bottom["column_span"]) == (1, 1, 2)
    assert bottom["y2"]["label"] == "right"
    assert bottom["legend"] is False
    assert bottom["elements"][0]["axis"] == "y2"
    ET.fromstring(chart.svg())


def test_invalid_spec_message() -> None:
    """A spec the renderer refuses raises its `ValueError` with the path."""
    pytest.importorskip("sbmlsim_core")
    chart = Chart()
    chart.plot().heatmap(["a", "b"], ["c", "d"], [[0, 1]])
    with pytest.raises(ValueError, match=r"plots\[0\]\.elements\[0\]"):
        chart.svg()


def test_dashed_rule() -> None:
    """A rule takes the type and width of its line."""
    chart = Chart()
    chart.plot().rule(1.0, line="dashed", width=1.5, color="#ff0000")

    (rule,) = chart.spec()["plots"][0]["elements"]
    assert rule["style"]["line"] == {"type": "dashed", "width": 1.5}
    assert rule["style"]["color"] == "#ff0000"
    pytest.importorskip("sbmlsim_core")
    assert "stroke-dasharray" in chart.svg()


def test_band_fill_and_median_style() -> None:
    """A band takes the fill and the line of its median."""
    chart = Chart()
    plot = chart.plot()
    plot.band([0, 1], [0, 1], [2, 3], median=[1, 2], line="dotted", width=1.0)
    plot.band([0, 1], [0, 1], [2, 3], fill="#00ff00")
    plot.band([0, 1], [0, 1], [2, 3], fill_opacity=0.5)

    dotted, colored, translucent = chart.spec()["plots"][0]["elements"]
    assert dotted["style"]["line"] == {"type": "dotted", "width": 1.0}
    assert "fill" not in dotted["style"]
    # a fill color keeps the opacity of a band of the renderer
    assert colored["style"]["fill"] == {"color": "#00ff00", "opacity": BAND_OPACITY}
    assert translucent["style"]["fill"] == {"opacity": 0.5}


def test_bar_outline() -> None:
    """A bar takes the line of its outline and its fill."""
    chart = Chart()
    plot = chart.plot()
    plot.bar([0, 1], [1, 2], line="solid", width=1.0, fill="#ffffff", color="#000000")
    plot.bar([0, 1], [1, 2])

    outlined, plain = chart.spec()["plots"][0]["elements"]
    assert outlined["style"] == {
        "color": "#000000",
        "opacity": 1.0,
        "line": {"type": "solid", "width": 1.0},
        "fill": {"color": "#ffffff", "opacity": 1.0},
    }
    assert plain["style"] == {"opacity": 1.0}


def test_marker_edge_width() -> None:
    """The markers of a curve and of the outliers of a box take an edge width."""
    chart = Chart()
    plot = chart.plot()
    plot.curve([0], [0], marker="circle", edge="#000000", edge_width=0.5)
    stats = {"low_whisker": 0, "q1": 1, "median": 2, "q3": 3, "high_whisker": 4}
    plot.box([0], [stats], marker="x", edge_width=2.0)

    curve, box = chart.spec()["plots"][0]["elements"]
    assert curve["style"]["marker"]["edge_width"] == 0.5
    assert curve["style"]["marker"]["edge"] == "#000000"
    assert box["style"]["marker"]["edge_width"] == 2.0


def test_quantity_polygon() -> None:
    """The coordinates of a polygon are values of the axes."""
    chart = Chart()
    plot = chart.plot(x=ChartAxis(unit="hr"))
    plot.polygon([Q([[0, 0], [60, 0], [0, 60]], "min")])
    plot.polygon([(Q([0, 120, 0], "min"), Q([0, 0, 1], "hr"))])

    spec = chart.spec()["plots"][0]
    first, second = spec["elements"]
    assert first["paths"] == [[[0.0, 0.0], [1.0, 0.0], [0.0, 60.0]]]
    assert second["paths"] == [[[0.0, 0.0], [2.0, 0.0], [0.0, 60.0]]]
    assert spec["y"]["unit"] == "min"
    with pytest.raises(pint.DimensionalityError, match="axis y"):
        chart.plots[0].polygon([(Q([0, 1, 0], "hr"), Q([0, 0, 1], "mM"))])


def test_categories() -> None:
    """Names on a category axis are the indices of their categories."""
    chart = Chart()
    given = chart.plot(x=ChartAxis(scale="category", categories=["a", "b", "c"]))
    given.bar(["c", "a"], [1, 2])
    given.text("b", 1.0, "middle")
    with pytest.raises(ValueError, match=r"'d'.*axis x"):
        given.bar(["d"], [1])

    collected = chart.plot(x=ChartAxis(scale="category"))
    collected.box(["x", "y"], [_box(), _box()])
    collected.violin(["y", "z"], [_violin(), _violin()])
    collected.curve(["z", "x"], [1, 2])

    numeric = chart.plot()
    with pytest.raises(ValueError, match="no category axis"):
        numeric.bar(["a"], [1])

    first, second, third = chart.spec()["plots"]
    assert first["x"]["categories"] == ["a", "b", "c"]
    assert first["elements"][0]["x"] == [2.0, 0.0]
    assert first["elements"][1]["x"] == 1.0
    assert second["x"]["categories"] == ["x", "y", "z"]
    assert second["elements"][0]["positions"] == [0.0, 1.0]
    assert second["elements"][1]["positions"] == [1.0, 2.0]
    assert second["elements"][2]["x"] == [2.0, 0.0]
    assert "categories" not in third["x"]


def _box() -> dict[str, float]:
    return {"low_whisker": 0, "q1": 1, "median": 2, "q3": 3, "high_whisker": 4}


def _violin() -> dict[str, list[float]]:
    return {"values": [0, 1, 2], "density": [0.1, 0.5, 0.1]}


def test_not_finite_positions_refused() -> None:
    """A statistic of a box, a text and a rule must be finite numbers."""
    plot = Chart().plot()
    with pytest.raises(ValueError, match=r"stats\[0\]\.median"):
        plot.box([0], [{**_box(), "median": math.nan}])
    with pytest.raises(ValueError, match="x of a text"):
        plot.text(math.inf, 0, "t")
    with pytest.raises(ValueError, match="value of a rule"):
        plot.rule(math.nan)
    # an outlier which is not finite is left out
    plot.box([0], [{**_box(), "outliers": [7.0, math.nan]}])
    assert plot.spec()["elements"][0]["stats"][0]["outliers"] == [7.0]


def test_numpy_scalars_serialize() -> None:
    """Numbers of numpy are written as numbers of JSON."""
    # numpy scalars where the signatures say int and float, as callers pass them
    sizes: dict[str, Any] = {"width": np.float64(400), "height": np.int64(300)}
    grid: dict[str, Any] = {
        "rows": np.int64(1),
        "columns": np.int64(1),
        "wspace": np.float32(0.1),
    }
    chart = Chart(**sizes, **grid, fonts={"tick": np.float64(9)})
    cell: dict[str, Any] = {
        "row": np.int64(0),
        "column": np.int64(0),
        "row_span": np.int64(1),
    }
    chart.plot(**cell).curve([0], [0])
    spec = json.loads(chart.to_json())
    assert spec["fonts"] == {"tick": 9.0}
    assert spec["plots"][0]["row_span"] == 1


def test_frame_and_declared_legends() -> None:
    """An axis takes a frame and a plot its declared legends."""
    chart = Chart()
    chart.plot(y=ChartAxis(frame=(0, 10)), declared_legends=[["a"], ["a", "b"]]).curve(
        [0], [0]
    )

    plot = chart.spec()["plots"][0]
    assert plot["y"]["frame"] == {"low": 0.0, "high": 10.0}
    assert plot["declared_legends"] == [["a"], ["a", "b"]]


def test_dataarray_name_labels_axis() -> None:
    """A data array names an axis without a label."""
    chart = Chart()
    plot = chart.plot(y=ChartAxis(label="given"))
    time = xr.DataArray(np.array([0.0, 1.0]), name="time", attrs={"units": "hr"})
    cp = xr.DataArray(np.array([1.0, 2.0]), name="cp")
    plot.curve(time, cp)

    spec = chart.spec()["plots"][0]
    assert spec["x"]["label"] == "time"
    assert spec["y"]["label"] == "given"


def test_bad_units_name_the_axis() -> None:
    """Units which cannot be read or converted name the axis."""
    plot = Chart().plot()
    bad = xr.DataArray(np.array([1.0]), attrs={"units": "no_such_unit"})
    with pytest.raises(ValueError, match="axis x"):
        plot.curve(bad, [1.0])

    registry = pint.UnitRegistry()
    plot = Chart().plot(x=ChartAxis(unit="hr"))
    plot.curve(registry.Quantity([60.0], "min"), [1.0])
    assert plot.spec()["elements"][0]["x"] == [1.0]
    with pytest.raises(pint.DimensionalityError, match="axis x"):
        plot.curve(registry.Quantity([1.0], "mM"), [1.0])


def test_spec_is_a_copy() -> None:
    """Changing a spec does not change the chart."""
    chart = Chart()
    chart.plot().curve([0, 1], [0, 1])
    spec = chart.spec()
    spec["plots"][0]["elements"][0]["y"][0] = 5.0
    assert chart.spec()["plots"][0]["elements"][0]["y"][0] == 0.0
