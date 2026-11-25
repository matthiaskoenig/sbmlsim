"""The plots of a sensitivity result, charts of the renderer of sbmlsim-core."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sbmlsim.engine import import_core
from sbmlsim.plot.chart import Chart
from sbmlsim.plot.text import text_width
from sbmlsim.sensitivity import local, plot_heatmap, plot_indices, plot_morris, plots
from sbmlsim.sensitivity.plots import box_stats, color_scale, draw_boxes
from sbmlsim.sensitivity.result import PARAMETER, SensitivityResult
from sbmlsim.simulation import Change, Formula, Scan, Simulation, sampling
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml


def _sobol() -> SensitivityResult:
    rng = np.random.default_rng(1)
    data = {}
    for o in ("auc", "cmax"):
        for key in ("S1", "ST", "S1_conf", "ST_conf"):
            data[f"{o}.{key}"] = ((PARAMETER, "dose"), rng.random((3, 2)))
    units = dict.fromkeys(data, "dimensionless")
    return SensitivityResult(
        xr.Dataset(
            data,
            coords={PARAMETER: ["a", "b", "c"], "dose": [0, 1]},
            attrs={"units": units, "method": "sobol"},
        )
    )


def _morris() -> SensitivityResult:
    data = {
        f"y.{key}": ((PARAMETER,), np.array([1.0, 0.5, 0.1]))
        for key in ("mu", "mu_star", "sigma", "mu_star_conf")
    }
    return SensitivityResult(
        xr.Dataset(
            data,
            coords={PARAMETER: ["a", "b", "c"]},
            attrs={"units": dict.fromkeys(data, ""), "method": "morris"},
        )
    )


def _plot(chart: Chart) -> dict[str, Any]:
    (plot,) = chart.spec()["plots"]
    return plot


def _elements(chart: Chart, kind: str) -> list[dict[str, Any]]:
    return [e for e in _plot(chart)["elements"] if e["kind"] == kind]


def _heat(chart: Chart) -> dict[str, Any]:
    (heatmap,) = _elements(chart, "heatmap")
    return heatmap


def _geometry(chart: Chart) -> dict[str, Any]:
    return import_core().figure_geometry(chart.to_json())


def test_plot_heatmap(tmp_path: Path) -> None:
    chart = plot_heatmap(_sobol(), "ST", dose=0, cutoff=None)
    assert isinstance(chart, Chart)
    chart.save(tmp_path / "h.svg")
    chart.save(tmp_path / "h.png", dpi=300)
    assert (tmp_path / "h.svg").read_text(encoding="utf-8").startswith("<svg")
    assert (tmp_path / "h.png").read_bytes()[:4] == b"\x89PNG"


def test_plot_indices_and_morris() -> None:
    chart = plot_indices(_sobol(), "auc", dose=0)
    bars = _elements(chart, "bar")
    assert [b["label"] for b in bars] == ["S1", "ST"]
    assert all(len(b["height"]) == 3 and len(b["err"]) == 3 for b in bars)
    chart.svg()
    chart = plot_morris(_morris(), "y")
    assert [t["text"] for t in _elements(chart, "text")] == ["a", "b", "c"]
    chart.svg()


def test_plot_indices_data() -> None:
    result = _sobol()
    chart = plot_indices(result, "cmax", dose=1)
    s1, st = _elements(chart, "bar")
    np.testing.assert_allclose(s1["height"], result["cmax.S1"].sel(dose=1).values)
    np.testing.assert_allclose(st["err"], result["cmax.ST_conf"].sel(dose=1).values)
    # the pair of a parameter sits around its category
    np.testing.assert_allclose(np.subtract(st["x"], s1["x"]), 0.4)
    plot = _plot(chart)
    assert plot["y"]["label"] == "Sobol index"
    assert plot["x"]["categories"] == ["a", "b", "c"]
    assert (s1["style"]["color"], st["style"]["color"]) == (
        plots.S1_COLOR,
        plots.ST_COLOR,
    )


def test_plot_morris_positions() -> None:
    (curve,) = _elements(plot_morris(_morris(), "y"), "curve")
    np.testing.assert_allclose(curve["x"], [1.0, 0.5, 0.1])
    np.testing.assert_allclose(curve["y"], [1.0, 0.5, 0.1])


def _single() -> SensitivityResult:
    data = {f"o{k}.ST": ((PARAMETER,), np.array([0.5])) for k in range(20)}
    return SensitivityResult(
        xr.Dataset(data, coords={PARAMETER: ["a"]}, attrs={"units": {}})
    )


def test_heatmap_one_parameter_and_size() -> None:
    chart = plot_heatmap(_single(), "ST")
    assert 0 < chart.height <= plots.MAX_HEATMAP_SIZE[1]
    assert chart.width <= plots.MAX_HEATMAP_SIZE[0]
    chart = plot_heatmap(_sobol(), "ST", dose=0)
    assert chart.height < 720


def _normalized() -> SensitivityResult:
    values = np.array([[0.5, 1.0], [np.nan, np.nan], [0.01, 0.02]])
    data = {f"o{k}.normalized": ((PARAMETER,), values[:, k]) for k in range(2)}
    return SensitivityResult(
        xr.Dataset(data, coords={PARAMETER: ["a", "b", "c"]}, attrs={"units": {}})
    )


def test_heatmap_keeps_undefined_rows() -> None:
    chart = plot_heatmap(_normalized(), "normalized")
    heatmap = _heat(chart)
    assert "b" in heatmap["rows"] and "c" not in heatmap["rows"]
    # an undefined cell has no value, which the renderer draws light grey
    b = heatmap["rows"].index("b")
    assert heatmap["values"][b] == [None, None]
    assert heatmap["text"][b] == ["", ""]
    assert not _elements(chart, "polygon")
    chart.svg()


def test_a_cell_below_the_cutoff_is_white_and_has_no_text() -> None:
    values = np.array([[0.5, 0.01], [0.2, 0.3]])
    chart = plot_heatmap(_table(values, index="normalized"), "normalized")
    heatmap = _heat(chart)
    row = heatmap["rows"].index("p0")
    assert heatmap["text"][row] == ["0.50", ""]
    (polygon,) = _elements(chart, "polygon")
    assert polygon["style"]["fill"] == {"color": "#ffffff", "opacity": 1.0}
    # the cell of the second column of its row, rows and columns at 0, 1, ...
    ((xs, ys),) = [tuple(zip(*ring, strict=True)) for ring in polygon["paths"]]
    assert (min(xs), max(xs)) == (0.5, 1.5)
    assert (min(ys), max(ys)) == (row - 0.5, row + 0.5)
    chart.svg()


def test_heatmap_title() -> None:
    chart = plot_heatmap(_sobol(), "ST", dose=0, title="normalized, [S1] = 1")
    assert chart.spec()["title"] == "normalized, [S1] = 1"
    (plot,) = _geometry(chart)["plots"]
    assert plot["rect"]["y"] > 16


def test_heatmap_defaults() -> None:
    def scale(chart: Chart) -> dict[str, Any]:
        heatmap = _heat(chart)
        assert _plot(chart)["colorbar"]["scale"] == heatmap["scale"]
        return heatmap["scale"]

    assert scale(plot_heatmap(_sobol(), "ST", dose=0)) == {
        "name": "viridis",
        "low": 0.0,
        "high": 1.0,
        "log": False,
    }
    assert scale(plot_heatmap(_normalized(), "normalized")) == {
        "name": "coolwarm",
        "low": -2.0,
        "high": 2.0,
        "center": 0.0,
        "log": False,
    }
    chart = plot_heatmap(_sobol(), "ST", dose=0, cmap="Magma", vmax=3)
    assert scale(chart)["name"] == "magma" and scale(chart)["high"] == 3.0
    # the diverging map of matplotlib of sbmlsim 0.8.5 is drawn with coolwarm
    chart = plot_heatmap(_normalized(), "normalized", cmap="seismic")
    assert scale(chart)["name"] == "coolwarm"
    # a center outside of the scale is not used
    chart = plot_heatmap(_normalized(), "normalized", vmin=0.0, vmax=2.0)
    assert "center" not in scale(chart)
    chart.svg()


def test_a_color_map_the_renderer_has_not_is_named() -> None:
    assert color_scale("RdBu") == "rdbu"
    with pytest.raises(ValueError, match=r"'Reds'.*viridis.*seismic"):
        plot_heatmap(_sobol(), "ST", dose=0, cmap="Reds")


def test_unknown_selections() -> None:
    with pytest.raises(ValueError, match=r"Unknown selection 'foo'.*dose"):
        plot_indices(_sobol(), "auc", foo=1)
    with pytest.raises(ValueError, match=r"Unknown label 5.*\[0, 1\]"):
        plot_heatmap(_sobol(), "ST", dose=5)
    with pytest.raises(ValueError, match=r"Unknown observable 'x'.*auc"):
        plot_indices(_sobol(), "x", dose=0)
    with pytest.raises(ValueError, match=r"Unknown observable 'x'"):
        plot_morris(_morris(), "x")
    with pytest.raises(ValueError, match="index 'nope'"):
        plot_heatmap(_sobol(), "nope", dose=0)


def _pixel_x(plot: dict[str, Any], x: float) -> float:
    rect, scale = plot["rect"], plot["x"]
    share = (x - scale["low"]) / (scale["high"] - scale["low"])
    return rect["x"] + share * rect["width"]


def _pixel_y(plot: dict[str, Any], y: float) -> float:
    rect, scale = plot["rect"], plot["y"]
    share = (y - scale["low"]) / (scale["high"] - scale["low"])
    return rect["y"] + (1.0 - share) * rect["height"]


@pytest.mark.parametrize(
    "names", [["a", "b", "c"], ["Vmax_transport_liver", "Km_glucose_uptake", "c"]]
)
def test_plot_morris_labels_inside_frame(names: list[str]) -> None:
    result = SensitivityResult(_morris().ds.assign_coords({PARAMETER: names}))
    chart = plot_morris(result, "y")
    (plot,) = _geometry(chart)["plots"]
    right = plot["rect"]["x"] + plot["rect"]["width"]
    for text in _elements(chart, "text"):
        assert _pixel_x(plot, text["x"]) + text_width(text["text"]) < right
        assert _pixel_y(plot, text["y"]) - 11 > plot["rect"]["y"]


def _result(
    index: str,
    values: np.ndarray,
    parameters: list[str],
    observables: list[str],
    unit: str = "dimensionless",
    method: str = "sobol",
) -> SensitivityResult:
    data = {
        f"{o}.{index}": ((PARAMETER,), values[:, k]) for k, o in enumerate(observables)
    }
    return SensitivityResult(
        xr.Dataset(
            data,
            coords={PARAMETER: parameters},
            attrs={"units": dict.fromkeys(data, unit), "method": method},
        )
    )


def _colorbar(chart: Chart) -> str:
    return _plot(chart)["colorbar"]["label"]


def test_heatmap_cutoff_depends_on_the_index() -> None:
    values = np.array([[0.05, 0.01], [0.002, 0.003], [0.04, 0.0]])
    mu_star = _result("mu_star", values, ["a", "b", "c"], ["o1", "o2"], unit="mM")
    chart = plot_heatmap(mu_star, "mu_star")
    # no cutoff for an index with a unit
    assert sorted(_heat(chart)["rows"]) == ["a", "b", "c"]
    assert _colorbar(chart) == "mu_star [mM]"
    long = _result("mu_star", values, ["a", "b", "c"], ["o1", "o2"], unit="millimolar")
    assert _colorbar(plot_heatmap(long, "mu_star")) == "mu_star [mM]"
    # an explicit cutoff is absolute in the unit of the index
    chart = plot_heatmap(mu_star, "mu_star", cutoff=0.03)
    assert sorted(_heat(chart)["rows"]) == ["a", "c"]
    # 0.1 for a dimensionless index, which no parameter reaches here
    st = _result("ST", values, ["a", "b", "c"], ["o1", "o2"])
    with pytest.raises(ValueError, match=r"cutoff 0\.1.*0\.05"):
        plot_heatmap(st, "ST")
    with pytest.raises(ValueError, match=r"cutoff 1.*0\.05"):
        plot_heatmap(mu_star, "mu_star", cutoff=1.0)
    assert len(_heat(plot_heatmap(st, "ST", cutoff=0))["rows"]) == 3


def test_heatmap_degenerate_rows() -> None:
    """Identical or undefined rows draw without a warning (filterwarnings = error)."""
    identical = _result("normalized", np.ones((3, 2)), ["a", "b", "c"], ["o1", "o2"])
    assert len(_heat(plot_heatmap(identical, "normalized"))["rows"]) == 3
    undefined = _result(
        "normalized", np.full((3, 2), np.nan), ["a", "b", "c"], ["o1", "o2"]
    )
    chart = plot_heatmap(undefined, "normalized")
    assert len(_heat(chart)["rows"]) == 3
    chart.svg()
    chart = plot_heatmap(undefined, "normalized", cutoff=None)
    assert len(_heat(chart)["rows"]) == 3


def _check_heatmap_layout(chart: Chart) -> None:
    """The cells leave room for the parameters left of them and are in the chart."""
    heatmap = _heat(chart)
    (plot,) = _geometry(chart)["plots"]
    rect = plot["rect"]
    widest = max(text_width(r) for r in heatmap["rows"])
    assert rect["x"] >= widest
    assert rect["x"] + rect["width"] <= chart.width
    assert rect["y"] + rect["height"] <= chart.height
    chart.svg()


def test_heatmap_layout_of_short_names() -> None:
    values = np.array([[0.5, -1.2], [2500.0, 0.3]])
    chart = plot_heatmap(
        _result("mu", values, ["k1", "k2"], ["o1", "o2"], unit="mM"), "mu"
    )
    _check_heatmap_layout(chart)
    assert _plot(chart)["x"]["tick_rotation"] == 0.0
    assert _colorbar(chart) == "mu [mM]"
    # the first row is at the top
    assert _plot(chart)["y"]["reverse"] is True


def test_heatmap_layout_of_long_and_many_names() -> None:
    rng = np.random.default_rng(1)
    parameters = [f"Vmax_transport_liver_{k}" for k in range(40)]
    observables = [f"concentration_in_plasma_{k}" for k in range(6)]
    result = _result("ST", rng.random((40, 6)), parameters, observables)
    chart = plot_heatmap(result, "ST", title="ST of the plasma concentrations")
    _check_heatmap_layout(chart)
    assert _plot(chart)["x"]["tick_rotation"] == 45.0
    one = _result("ST", np.array([[0.5]]), ["Vmax_transport_liver"], ["o1"])
    _check_heatmap_layout(plot_heatmap(one, "ST"))


def test_heatmap_names_its_index_and_unit() -> None:
    assert _colorbar(plot_heatmap(_normalized(), "normalized")) == "normalized"
    mixed = _result("raw", np.ones((2, 2)), ["a", "b"], ["o1", "o2"], unit="mM")
    mixed.ds.attrs["units"]["o2.raw"] = "mM/s"
    assert _colorbar(plot_heatmap(mixed, "raw")) == "raw"


def test_plot_indices_labels() -> None:
    plot = _plot(plot_indices(_sobol(), "auc", dose=1))
    assert plot["x"]["tick_rotation"] == 0.0
    assert plot["x"]["label"] == "parameter"
    assert plot["title"] == "auc"


def test_plot_indices_long_names() -> None:
    parameters = [f"Vmax_transport_liver_{k}" for k in range(12)]
    data = {
        f"y.{key}": ((PARAMETER,), np.linspace(0.1, 0.9, 12))
        for key in ("S1", "ST", "S1_conf", "ST_conf")
    }
    result = SensitivityResult(
        xr.Dataset(
            data,
            coords={PARAMETER: parameters},
            attrs={"units": dict.fromkeys(data, "dimensionless"), "method": "fast"},
        )
    )
    chart = plot_indices(result, "y")
    plot = _plot(chart)
    # the first name rotated by 45 degrees would end left of the chart
    assert plot["x"]["tick_rotation"] == 90.0
    assert plot["y"]["label"] == "FAST index"
    (geometry,) = _geometry(chart)["plots"]
    # the rotated names are below the panel, inside the chart
    assert geometry["rect"]["y"] + geometry["rect"]["height"] < chart.height - 100


def test_the_plots_check_the_analysis() -> None:
    with pytest.raises(ValueError, match=r"S1 and ST.*Sobol or FAST.*morris"):
        plot_indices(_morris(), "y")
    with pytest.raises(ValueError, match=r"mu_star and sigma.*Morris.*sobol"):
        plot_morris(_sobol(), "auc", dose=0)


def test_plot_morris_axes_and_undefined_parameters() -> None:
    data = {
        f"y.{key}": ((PARAMETER,), np.array([5e-5, 2e-5, np.nan]))
        for key in ("mu", "mu_star", "sigma", "mu_star_conf")
    }
    result = SensitivityResult(
        xr.Dataset(
            data,
            coords={PARAMETER: ["a", "b", "c"]},
            attrs={"units": dict.fromkeys(data, "mM"), "method": "morris"},
        )
    )
    chart = plot_morris(result, "y")
    plot = _plot(chart)
    assert plot["x"]["min"] == 0 and plot["y"]["min"] == 0
    assert plot["x"]["max"] > 5e-5 and plot["y"]["max"] > 5e-5
    notes = [t["text"] for t in _elements(chart, "text")]
    assert notes[:2] == ["a", "b"]
    assert any("undefined" in n and "c" in n for n in notes)
    assert (plot["x"]["label"], plot["x"]["unit"]) == ("mu_star", "mM")
    chart.svg()
    # a unit in the long form of pint is written in its short form
    long = SensitivityResult(
        result.ds.assign_attrs(units=dict.fromkeys(data, "millimolar / minute"))
    )
    plot = _plot(plot_morris(long, "y"))
    assert plot["x"]["unit"] == "mM/min" and plot["y"]["unit"] == "mM/min"


def _table(
    values: np.ndarray, columns: list[str] | None = None, index: str = "mu_star"
) -> SensitivityResult:
    n_rows, n_cols = values.shape
    names = columns or [f"o{i}" for i in range(n_cols)]
    data = {f"{n}.{index}": ((PARAMETER,), values[:, i]) for i, n in enumerate(names)}
    return SensitivityResult(
        xr.Dataset(
            data,
            coords={PARAMETER: [f"p{i}" for i in range(n_rows)]},
            attrs={"units": dict.fromkeys(data, ""), "method": "morris"},
        )
    )


def _cells(chart: Chart) -> list[str]:
    return [t for row in _heat(chart)["text"] for t in row]


def test_forty_observables_are_rotated() -> None:
    rng = np.random.default_rng(0)
    names = [f"cmax_{i:02d}" for i in range(40)]
    chart = plot_heatmap(_table(rng.random((3, 40)) + 0.5, names), "mu_star")
    # the cells of a chart of the largest width are too narrow for the names
    (plot,) = _geometry(chart)["plots"]
    assert text_width(names[0]) > plot["rect"]["width"] / 40
    assert _plot(chart)["x"]["tick_rotation"] == 45.0
    _check_heatmap_layout(chart)


def test_a_long_first_observable_is_vertical() -> None:
    """A rotated label left of the plot would be cut, it is vertical instead."""
    names = ["the minimum of the change of the body weight over the whole study", "y"]
    chart = plot_heatmap(_table(np.ones((2, 2)), names), "mu_star")
    assert _plot(chart)["x"]["tick_rotation"] == 90.0
    # a rotated label which ends left of the labels of the rows
    parameters = ["k: the rate of the decay", "Vc: central"]
    observables = ["the minimum of the body weight", "y"]
    result = _result("ST", np.ones((2, 2)), parameters, observables)
    chart = plot_heatmap(result, "ST")
    assert _plot(chart)["x"]["tick_rotation"] == 45.0


def test_an_infinite_value_is_clustered_like_nan() -> None:
    values = np.array([[1.0, np.inf], [1.2, 0.0], [np.nan, 0.5]])
    chart = plot_heatmap(_table(values), "mu_star", cutoff=0)
    assert isinstance(chart, Chart)
    chart.svg()


def test_a_zero_is_printed_as_zero() -> None:
    chart = plot_heatmap(_table(np.zeros((2, 2))), "mu_star", cutoff=0)
    assert set(_cells(chart)) == {"0"}
    values = np.array([[1.0, -1e-9], [-0.0, 0.5]])
    texts = _cells(plot_heatmap(_table(values), "mu_star", cutoff=0))
    assert sorted(texts) == ["0", "0", "0.50", "1.00"]
    assert not any("-" in t or t[:2] == "\N{MINUS SIGN}0" for t in texts)


def test_a_parameter_without_indices_is_marked() -> None:
    result = _sobol()
    result.ds["auc.S1"].values[1] = np.nan
    result.ds["auc.ST"].values[1] = np.nan
    chart = plot_indices(result, "auc", dose=0)
    (note,) = _elements(chart, "text")
    assert (note["text"], note["x"]) == ("n/a", 1.0)
    chart.svg()


def test_a_plot_at_the_time_of_a_change_shows_the_state_after_it() -> None:
    """The time of a change selects the state after it, as a time of a grid does.

    The output has the states before and after the reset of `A` at 5, the
    plots raised that `time` had two labels.
    """
    simulator = Simulator(n_workers=1)
    model = simulator.load(
        sbml("model d\n  var A = 0\n  k = 0.2\n  p = 1\n  A' = p - k*A\nend")
    )
    simulation = Simulation(
        end=10, times=[0, 2.5, 5, 7.5, 10], changes=[Change(5, {"A": 1.0})]
    )
    design = sampling.local(["k", "p"], 0.01, model=model)
    scan = simulator.run(model, Scan(simulation, [design]), [Formula("a", "A")])
    result = local(scan)
    assert result["a.normalized"]["time"].values.tolist() == [0, 2.5, 5, 5, 7.5, 10]
    before, after = (
        result["a.normalized"].isel(time=2),
        result["a.normalized"].isel(time=3),
    )
    assert not np.allclose(before.values, after.values)
    at = result.sel(time=5.0)
    np.testing.assert_array_equal(at["a.normalized"].values, after.values)
    stacked = at.index("normalized")
    np.testing.assert_array_equal(stacked.sel(observable="a").values, after.values)
    heatmap = _heat(plot_heatmap(at, "normalized", cutoff=0))
    assert np.asarray(heatmap["values"]).size == 2


def test_the_indices_at_the_time_of_a_change_are_the_ones_after_it() -> None:
    """A plot of a timecourse selects the state after a change at its time."""
    time = [0.0, 5.0, 5.0, 10.0]
    data = {
        f"y.{key}": ((PARAMETER, "time"), np.full((2, 4), 0.01))
        for key in ("S1_conf", "ST_conf")
    }
    for key, scale in (("S1", 1.0), ("ST", 2.0)):
        values = np.array([[0.1, 0.2, 0.3, 0.4], [0.05, 0.1, 0.15, 0.2]]) * scale
        data[f"y.{key}"] = ((PARAMETER, "time"), values)
    result = SensitivityResult(
        xr.Dataset(
            data,
            coords={PARAMETER: ["a", "b"], "time": time},
            attrs={"units": dict.fromkeys(data, ""), "method": "sobol"},
        )
    )

    def heights(chart: Chart) -> list[float]:
        return sorted(h for bar in _elements(chart, "bar") for h in bar["height"])

    after = heights(plot_indices(result.isel(time=2), "y"))
    assert heights(plot_indices(result, "y", time=5.0)) == after
    np.testing.assert_allclose(after, [0.15, 0.3, 0.3, 0.6])


@pytest.mark.parametrize(
    "index, columns",
    [
        (
            [
                "BW0: bodyweight",
                "Emax_SAT_glp1: Emax for GLP-1 agonist on fat loss (rate of fat change)",
            ],
            ["min BW", "min BW change", "min BW ratio"],
        ),
        (
            ["k1", "k2"],
            ["the minimum of the change of the body weight over the whole study", "y"],
        ),
    ],
)
def test_a_heatmap_of_long_labels_has_room_for_them(
    index: list[str], columns: list[str]
) -> None:
    """A heatmap of long labels keeps the size of its cells (#314)."""
    df = pd.DataFrame(
        [[0.86, 5.46, -0.14][: len(columns)], [0.0, 0.82, 0.0][: len(columns)]],
        index=index,
        columns=columns,
    )
    chart = plots._heatmap(
        df,
        label="normalized",
        cutoff=0.05,
        cluster_rows=False,
        title=None,
        cmap="seismic",
        vcenter=0.0,
        vmin=-2.0,
        vmax=2.0,
    )
    _check_heatmap_layout(chart)
    (plot,) = _geometry(chart)["plots"]
    width, height = plots.CELL_SIZE
    height = max(height, plots.MIN_CELLS_HEIGHT / len(index))
    cells = (
        plot["rect"]["width"] / len(columns),
        plot["rect"]["height"] / len(index),
    )
    assert cells == pytest.approx((width, height), rel=0.2)


def test_the_statistics_of_a_box() -> None:
    """The whiskers reach the last value within 1.5 interquartile ranges."""
    stats = box_stats([1.0, 2.0, 3.0, 4.0, 100.0, np.nan])
    assert stats == {
        "low_whisker": 1.0,
        "q1": 2.0,
        "median": 3.0,
        "q3": 4.0,
        "high_whisker": 4.0,
        "outliers": [100.0],
    }
    # the quartiles are the linear interpolation of numpy
    values = np.random.default_rng(0).normal(size=101)
    stats = box_stats(values)
    assert stats is not None
    assert (stats["q1"], stats["median"], stats["q3"]) == tuple(
        np.percentile(values, [25, 50, 75])
    )
    assert stats["low_whisker"] in values and stats["high_whisker"] in values
    assert box_stats([np.nan]) is None and box_stats([]) is None
    single = box_stats([2.0])
    assert single is not None and single["outliers"] == []


def test_draw_boxes() -> None:
    chart = Chart()
    plot = chart.plot()
    assert draw_boxes(plot, []) == []
    rng = np.random.default_rng(0)
    data = [rng.normal(size=50) + k for k in range(3)] + [np.array([])]
    assert len(draw_boxes(plot, data)) == 1
    # the empty sample has no box, the others are at 1, ..., n
    (box,) = plot.spec()["elements"]
    assert box["positions"] == [1.0, 2.0, 3.0]
    chart.svg()
    chart = Chart()
    plot = chart.plot()
    ids = draw_boxes(
        plot,
        data[:3],
        [0.0, 1.0, 2.0],
        colors=["C0", "red", "#00ff00"],
        showfliers=False,
    )
    boxes = plot.spec()["elements"]
    assert len(ids) == len(boxes) == 3
    assert [b["style"]["fill"]["color"] for b in boxes] == [
        "#1f77b4",
        "#ff0000",
        "#00ff00",
    ]
    assert all(s["outliers"] == [] for b in boxes for s in b["stats"])
    chart.svg()
