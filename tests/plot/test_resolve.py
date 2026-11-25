"""The figure model resolves into a chart of the renderer, `figure_chart`."""

import json
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sbmlsim.data import Data
from sbmlsim.engine import import_core
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.plot import Axis, Figure, FigureSettings, Plot
from sbmlsim.plot.plotting import CurveType, ShadedArea, Style, YAxisPosition
from sbmlsim.plot.points import nan_quantile
from sbmlsim.plot.resolve import figure_chart
from sbmlsim.simulator import Simulator
from tests.plot.scan_experiment import ScanFigures

SVG = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def experiment() -> SimulationExperiment:
    runner = ExperimentRunner(
        experiment_classes=[ScanFigures],
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
        on_error="raise",
    )
    experiment = runner.experiments["ScanFigures"]
    experiment.run(runner.simulator)
    return experiment


def _figure(
    experiment: SimulationExperiment,
    draw: Callable[[Plot], object],
    xaxis: Axis | None = None,
    yaxis: Axis | None = None,
    legend: bool = True,
    **settings: Any,
) -> Figure:
    figure = Figure(experiment=experiment, sid="fig", settings=settings or None)
    plot = figure.create_plots(
        xaxis=xaxis or Axis("time", unit="hr"),
        yaxis=yaxis or Axis("C", unit="mg/l"),
        legend=legend,
    )[0]
    draw(plot)
    return figure


def _plot(
    experiment: SimulationExperiment, draw: Callable[[Plot], object], **kwargs: Any
) -> dict:
    """The spec of the one plot of a figure which `draw` fills."""
    chart = figure_chart(experiment, _figure(experiment, draw, **kwargs))
    return chart.spec()["plots"][0]


def _legend(svg: str) -> list[str]:
    """The labels of the legend entries the renderer draws, in their order."""
    root = ET.fromstring(svg)
    return [
        text.text or ""
        for text in root.iter(f"{SVG}text")
        if text.get("class") == "sbmlsim-legend-label"
    ]


def _doses(plot: Plot, task: str = "task_doses", **kwargs: Any) -> None:
    plot.curve(
        x=Data("time", task=task), y=Data("[C]", task=task), over="dose", **kwargs
    )


def test_one_plot_per_subplot(experiment: SimulationExperiment) -> None:
    figure = Figure(experiment=experiment, sid="grid", num_rows=2, num_cols=3)
    plots = [
        Plot(sid=f"p{k}", xaxis=Axis("time"), yaxis=Axis("C"), name=f"plot {k}")
        for k in range(3)
    ]
    figure.add_subplot(plots[0], row=1, col=1, col_span=2)
    figure.add_subplot(plots[1], row=1, col=3, row_span=2)
    figure.add_subplot(plots[2], row=2, col=1)
    spec = figure_chart(experiment, figure).spec()
    assert spec["grid"]["rows"] == 2
    assert spec["grid"]["columns"] == 3
    assert [
        (p["row"], p["column"], p["row_span"], p["column_span"], p["title"])
        for p in spec["plots"]
    ] == [(0, 0, 1, 2, "plot 0"), (0, 2, 2, 1, "plot 1"), (1, 0, 1, 1, "plot 2")]


def test_axis_units_and_labels(experiment: SimulationExperiment) -> None:
    spec = _plot(
        experiment,
        lambda p: p.curve(
            x=Data("time", task="task_doses"),
            y=Data("[C]", task="task_doses"),
            over="dose",
        ),
        xaxis=Axis("time", unit="min"),
        yaxis=Axis("HCTZ urine\n", unit="mg/l", scale="log"),
    )
    assert spec["x"]["label"] == "time"
    assert spec["x"]["unit"] == "min"
    # the data is converted to the unit of the axis: 24 hr are 1440 min
    assert max(spec["elements"][0]["x"]) == pytest.approx(1440.0)
    # a label is one line of text, the renderer has no line breaks
    assert spec["y"]["label"] == "HCTZ urine"
    assert spec["y"]["unit"] == "mg/l"
    assert spec["y"]["scale"] == "log"

    # a unit id of the model is resolved to its unit, in its short form
    spec = _plot(
        experiment,
        lambda p: p.curve(
            x=Data("draw.ke", task="task_draws"), y=Data("draw.ke", task="task_draws")
        ),
        xaxis=Axis("ke", unit="per_hr"),
        yaxis=Axis("ke", unit="1/min"),
    )
    assert spec["x"]["unit"] == "1/h"
    assert spec["elements"][0]["x"] == pytest.approx(np.linspace(0.1, 0.4, 20))
    assert spec["elements"][0]["y"] == pytest.approx(np.linspace(0.1, 0.4, 20) / 60)

    # a name overrides the label and the unit, the data keeps the unit
    spec = _plot(
        experiment,
        lambda p: p.curve(
            x=Data("dose.PODOSE", task="task_doses"), y=Data("dose", task="task_doses")
        ),
        xaxis=Axis(name="Dosis (mg)", unit="mg"),
        yaxis=Axis("position", unit="dimensionless"),
    )
    assert spec["x"]["label"] == "Dosis (mg)"
    assert "unit" not in spec["x"]
    assert spec["y"]["unit"] == "-"


def test_a_plot_without_axes_hides_them(experiment: SimulationExperiment) -> None:
    figure = Figure(experiment=experiment, sid="bare")
    plot = figure.create_plots()[0]
    plot.title_visible = False
    plot.name = "hidden"
    _doses(plot)
    spec = figure_chart(experiment, figure).spec()["plots"][0]
    assert "title" not in spec
    for axis in ("x", "y"):
        assert spec[axis]["label_visible"] is False
        assert spec[axis]["ticks_visible"] is False


def test_axis_bounds_reverse_and_grid(experiment: SimulationExperiment) -> None:
    spec = _plot(
        experiment,
        _doses,
        xaxis=Axis("time", unit="hr", min=1, max=10, reverse=True, grid=True),
        yaxis=Axis(
            "C", unit="mg/l", scale="log", min=0.0, max=5.0, ticks_visible=False
        ),
    )
    assert (spec["x"]["min"], spec["x"]["max"]) == (1.0, 10.0)
    assert spec["x"]["reverse"] is True
    assert spec["x"]["grid"] is True
    # a bound which is not positive is no bound of a log axis
    assert "min" not in spec["y"]
    assert spec["y"]["max"] == 5.0
    assert spec["y"]["ticks_visible"] is False


def test_curve_with_errors(experiment: SimulationExperiment) -> None:
    spec = _plot(
        experiment,
        lambda p: p.curve(
            x=Data("time", task="task_doses", sel={"dose": 1}),
            y=Data("[C]", task="task_doses", sel={"dose": 1}),
            yerr=Data("[C]", task="task_doses", sel={"dose": 1}),
            xerr=Data("time", task="task_doses", sel={"dose": 1}),
            label="data",
            color="tab:red",
            alpha=0.5,
            linestyle="--",
            marker="s",
            markersize=4,
        ),
    )
    (element,) = spec["elements"]
    assert element["kind"] == "curve"
    assert element["label"] == "data"
    assert element["yerr"] == pytest.approx(element["y"])
    assert element["xerr"] == pytest.approx(element["x"])
    style = element["style"]
    assert style["color"] == "#d62728"
    assert style["opacity"] == pytest.approx(0.5, abs=0.01)
    assert style["line"]["type"] == "dashed"
    assert style["marker"]["type"] == "square"
    assert style["marker"]["size"] == 4.0


def test_line_and_marker_types(experiment: SimulationExperiment) -> None:
    pairs = {
        "-": "solid",
        "--": "dashed",
        ":": "dotted",
        "-.": "dashdot",
        "dashdotdotted": "dashdot",
        "none": "none",
    }
    markers = {
        "o": "circle",
        "s": "square",
        "D": "diamond",
        "^": "triangle",
        "v": "triangle",
        "<": "triangle",
        ">": "triangle",
        "+": "plus",
        "x": "x",
        "*": "cross",
        "_": "plus",
        "|": "plus",
        "": "none",
    }

    def draw(plot: Plot) -> None:
        for linestyle in pairs:
            _doses(plot, linestyle=linestyle)
        for marker in markers:
            _doses(plot, marker=marker)

    spec = _plot(experiment, draw)
    styles = [e["style"] for e in spec["elements"]]
    lines = [s["line"]["type"] for s in styles[: 3 * len(pairs) : 3]]
    assert lines == list(pairs.values())
    found = [s["marker"]["type"] for s in styles[3 * len(pairs) :: 3]]
    assert found == list(markers.values())


def test_over_lines_colours_and_legend(experiment: SimulationExperiment) -> None:
    figure = _figure(experiment, lambda p: _doses(p, label="C"))
    chart = figure_chart(experiment, figure)
    spec = chart.spec()["plots"][0]
    elements = spec["elements"]
    assert len(elements) == 3
    assert "colorbar" not in spec
    # the viridis colours at the values 50, 100 and 200 mg of the doses
    colors = import_core().color_scale("viridis", 0.0, 1.0, [0.0, 0.85 / 3, 0.85])
    assert [e["style"]["color"] for e in elements] == list(colors)
    assert len({e["legend"] for e in elements}) == 3
    assert _legend(chart.svg()) == [
        "C, PODOSE = 50 mg",
        "C, PODOSE = 100 mg",
        "C, PODOSE = 200 mg",
    ]


def test_shades_of_the_colour_of_a_curve(experiment: SimulationExperiment) -> None:
    spec = _plot(experiment, lambda p: _doses(p, color="tab:red"))
    colors = [e["style"]["color"] for e in spec["elements"]]
    assert colors[-1] == "#d62728"
    assert len(set(colors)) == 3


def test_two_dimensions_colour_and_line_style(experiment: SimulationExperiment) -> None:
    figure = _figure(
        experiment,
        lambda p: p.curve(
            x=Data("time", task="task_grid2"),
            y=Data("[C]", task="task_grid2"),
            over=("dose", "rate"),
            label="C",
        ),
    )
    chart = figure_chart(experiment, figure)
    elements = chart.spec()["plots"][0]["elements"]
    lines = [e for e in elements if e["x"]]
    assert len(lines) == 6
    assert [e["style"]["line"]["type"] for e in lines] == ["solid", "dashed"] * 3
    assert _legend(chart.svg()) == [
        "C, PODOSE = 50 mg",
        "C, PODOSE = 100 mg",
        "C, PODOSE = 200 mg",
        "ke = 0.1 1/hr",
        "ke = 0.3 1/hr",
    ]


def test_colorbar_from_11_points(experiment: SimulationExperiment) -> None:
    figure = _figure(experiment, lambda p: _doses(p, task="task_many", label="C"))
    chart = figure_chart(experiment, figure)
    spec = chart.spec()["plots"][0]
    elements = spec["elements"]
    assert len(elements) == 12
    # one legend entry of the curve, the lines share its group
    assert len({e["legend"] for e in elements}) == 1
    assert _legend(chart.svg()) == ["C"]
    colorbar = spec["colorbar"]
    assert colorbar["label"] == "PODOSE [mg]"
    scale = colorbar["scale"]
    assert scale["name"] == "viridis"
    assert scale["log"] is False
    assert scale["low"] == pytest.approx(10.0)
    # the colours of the lines are the colours of the bar at their values
    values = np.linspace(10.0, 120.0, 12)
    colors = import_core().color_scale(
        "viridis", scale["low"], scale["high"], values.tolist()
    )
    assert [e["style"]["color"] for e in elements] == list(colors)


def test_a_logarithmic_colorbar_of_geometric_doses(
    experiment: SimulationExperiment,
) -> None:
    spec = _plot(experiment, lambda p: _doses(p, task="task_geom"))
    scale = spec["colorbar"]["scale"]
    assert scale["log"] is True
    colors = import_core().color_scale(
        "viridis",
        scale["low"],
        scale["high"],
        np.geomspace(1, 1000, 12).tolist(),
        log=True,
    )
    assert [e["style"]["color"] for e in spec["elements"]] == list(colors)
    # the lines share the legend entry of the curve
    assert len({e["legend"] for e in spec["elements"]}) == 1
    assert sum("label" in e for e in spec["elements"]) == 1


def test_band_quantiles_and_median(experiment: SimulationExperiment) -> None:
    figure = _figure(
        experiment,
        lambda p: p.band(
            x=Data("time", task="task_draws"),
            y=Data("[C]", task="task_draws"),
            across="draw",
            name="C",
            color="tab:blue",
        ),
    )
    chart = figure_chart(experiment, figure)
    (band,) = chart.spec()["plots"][0]["elements"]
    assert band["kind"] == "band"
    y = Data("[C]", task="task_draws").get_data(experiment, to_units="mg/l")
    values = np.asarray(y.transpose("time", "draw").values)
    assert band["low"] == pytest.approx(nan_quantile(values, 0.05))
    assert band["high"] == pytest.approx(nan_quantile(values, 0.95))
    assert band["median"] == pytest.approx(nan_quantile(values, 0.5))
    assert band["style"]["color"] == "#1f77b4"
    assert band["style"]["fill"]["opacity"] == pytest.approx(0.3)
    assert _legend(chart.svg()) == ["C (5-95 %)"]


def test_a_band_per_dose(experiment: SimulationExperiment) -> None:
    figure = _figure(
        experiment,
        lambda p: p.band(
            x=Data("time", task="task_dose_draws"),
            y=Data("[C]", task="task_dose_draws"),
            across="draw",
            over="dose",
            median=False,
            name="C",
        ),
    )
    chart = figure_chart(experiment, figure)
    bands = chart.spec()["plots"][0]["elements"]
    assert len(bands) == 3
    assert all("median" not in b for b in bands)
    assert _legend(chart.svg()) == [
        "C, PODOSE = 50 mg (5-95 %)",
        "C, PODOSE = 100 mg (5-95 %)",
        "C, PODOSE = 200 mg (5-95 %)",
    ]


def test_shaded_area(experiment: SimulationExperiment) -> None:
    def draw(plot: Plot) -> None:
        x = Data("time", task="task_doses", sel={"dose": 0})
        plot.add_area(
            ShadedArea(
                x=x,
                yfrom=Data("[C]", task="task_doses", sel={"dose": 0}),
                yto=Data("[C]", task="task_doses", sel={"dose": 2}),
                label="range",
                style=Style.from_mpl_kwargs(color="tab:green", alpha=0.4),
            )
        )

    figure = _figure(experiment, draw)
    chart = figure_chart(experiment, figure)
    (area,) = chart.spec()["plots"][0]["elements"]
    assert area["kind"] == "area"
    assert len(area["from"]) == len(area["to"]) == len(area["x"])
    assert area["style"]["fill"]["color"] == "#2ca02c"
    assert area["style"]["fill"]["opacity"] == pytest.approx(0.4, abs=0.01)
    # the alpha of the colour is the opacity of the fill, once
    assert area["style"]["opacity"] == 1.0
    assert _legend(chart.svg()) == ["range"]


def test_right_axis(experiment: SimulationExperiment) -> None:
    def draw(plot: Plot) -> None:
        plot.set_yaxis_right("C", unit="g/l")
        _doses(plot, label="C")
        plot.curve(
            x=Data("time", task="task_doses", sel={"dose": 0}),
            y=Data("[C]", task="task_doses", sel={"dose": 0}),
            label="right",
            yaxis_position=YAxisPosition.RIGHT,
        )

    spec = _plot(experiment, draw)
    assert spec["y2"]["label"] == "C"
    assert spec["y2"]["unit"] == "g/l"
    right = [e for e in spec["elements"] if e["axis"] == "y2"]
    assert [e["label"] for e in right] == ["right"]
    # the concentration in mg/l in the unit g/l of the axis
    assert max(right[0]["y"]) < 1.0

    def wrong(plot: Plot) -> None:
        plot.curve(
            x=Data("time", task="task_doses", sel={"dose": 0}),
            y=Data("[C]", task="task_doses", sel={"dose": 0}),
            yaxis_position=YAxisPosition.RIGHT,
        )

    with pytest.raises(ValueError, match="right y axis"):
        _plot(experiment, wrong)


def test_log_axis_with_zero_data(experiment: SimulationExperiment) -> None:
    figure = _figure(experiment, _doses, yaxis=Axis("C", unit="mg/l", scale="log"))
    chart = figure_chart(experiment, figure)
    # the first value of every timecourse is 0, which a log axis cannot show
    assert chart.spec()["plots"][0]["elements"][0]["y"][0] == 0.0
    ET.fromstring(chart.svg())


def test_ragged_scan_nan_padding_renders(experiment: SimulationExperiment) -> None:
    def draw(plot: Plot) -> None:
        _doses(plot, task="task_ragged", label="C")
        # 0 / 0 at the start of the simulation is a gap
        c = Data("[C]", task="task_doses", sel={"dose": 0})
        plot.curve(
            x=Data("time", task="task_doses", sel={"dose": 0}),
            y=Data("gap", function="c / c * c", variables={"c": c}),
        )
        plot.band(
            x=Data("time", task="task_ragged"),
            y=Data("[C]", task="task_ragged"),
            across="dose",
            name="band",
        )

    figure = _figure(experiment, draw)
    chart = figure_chart(experiment, figure)
    text = chart.to_json()
    assert "NaN" not in text
    assert "null" in text
    json.loads(text)
    ET.fromstring(chart.svg())


def test_settings_map(experiment: SimulationExperiment) -> None:
    spec = figure_chart(experiment, _figure(experiment, _doses)).spec()
    defaults = FigureSettings()
    assert (spec["width"], spec["height"]) == (504.0, 360.0)
    assert spec["fonts"] == {
        "title": defaults.fig_titlesize,
        "label": defaults.axes_labelsize,
        "tick": defaults.xtick_labelsize,
        "legend": defaults.legend_fontsize,
    }
    assert spec["grid"]["wspace"] == defaults.fig_subplots_wspace
    assert spec["legend"] == {"scope": "plot", "position": "inside"}

    figure = _figure(
        experiment,
        _doses,
        panel_width=3.0,
        legend_position="outside",
        legend_fontsize=9,
        fig_subplots_hspace=0.5,
    )
    spec = figure_chart(experiment, figure, {"legend_scope": "row"}).spec()
    assert spec["width"] == 216.0
    assert spec["fonts"]["legend"] == 9.0
    assert spec["grid"]["hspace"] == 0.5
    assert spec["legend"] == {"scope": "row", "position": "outside"}


def test_the_plot_legend_and_the_title(experiment: SimulationExperiment) -> None:
    figure = _figure(experiment, lambda p: _doses(p, label="C"), legend=False)
    figure.name = "Doses"
    spec = figure_chart(experiment, figure).spec()
    assert spec["title"] == "Doses"
    assert spec["plots"][0]["legend"] is False


def test_equal_entries_of_a_row_share_their_group(
    experiment: SimulationExperiment,
) -> None:
    figure = Figure(experiment=experiment, sid="row", num_cols=2)
    for plot in figure.create_plots(xaxis=Axis("time", unit="hr"), yaxis=Axis("C")):
        plot.curve(
            x=Data("time", task="task_doses", sel={"dose": 0}),
            y=Data("[C]", task="task_doses", sel={"dose": 0}),
            label="simulation",
            color="black",
        )
    chart = figure_chart(experiment, figure, {"legend_scope": "row"})
    groups = [p["elements"][0]["legend"] for p in chart.spec()["plots"]]
    assert groups[0] == groups[1]
    assert _legend(chart.svg()) == ["simulation"]


def test_bars(experiment: SimulationExperiment) -> None:
    def draw(plot: Plot) -> None:
        for kind in (CurveType.BAR, CurveType.HORIZONTALBAR):
            plot.curve(
                x=Data("dose.PODOSE", task="task_doses"),
                y=Data("pk.cmax", task="task_doses"),
                type=kind,
                color="tab:orange",
            )

    spec = _plot(experiment, draw, xaxis=Axis("dose", unit="mg"))
    vertical, horizontal = spec["elements"]
    assert vertical["kind"] == horizontal["kind"] == "bar"
    assert vertical["horizontal"] is False
    assert horizontal["horizontal"] is True
    assert vertical["x"] == pytest.approx([50.0, 100.0, 200.0])
    assert vertical["style"]["color"] == "#ff7f0e"


@pytest.mark.parametrize("kind", [CurveType.BARSTACKED, CurveType.HORIZONTALBARSTACKED])
def test_stacked_bar_refused(experiment: SimulationExperiment, kind: CurveType) -> None:
    def draw(plot: Plot) -> None:
        plot.curve(
            x=Data("dose.PODOSE", task="task_doses"),
            y=Data("pk.cmax", task="task_doses"),
            type=kind,
        )

    with pytest.raises(ValueError, match=kind.name):
        _plot(experiment, draw)


@pytest.fixture(scope="module")
def hctz_figures() -> list[tuple[SimulationExperiment, Figure]]:
    from examples.hctz_fitting.project import PROJECT

    runner = PROJECT.experiment_runner(
        PROJECT.experiment_classes(),
        simulator=PROJECT.simulator.simulator(1),
        on_error="raise",
    )
    figures = []
    for experiment in runner.experiments.values():
        experiment.run(runner.simulator)
        figures += [(experiment, figure) for figure in experiment._figures.values()]
    return figures


def test_every_hctz_figure_renders(
    hctz_figures: list[tuple[SimulationExperiment, Figure]],
) -> None:
    assert len(hctz_figures) == 8
    for experiment, figure in hctz_figures:
        chart = figure_chart(experiment, figure)
        assert "NaN" not in chart.to_json()
        root = ET.fromstring(chart.svg())
        assert root.tag == f"{SVG}svg"
        assert len(chart.spec()["plots"]) == len(figure.subplots)


def test_a_colorbar_of_a_dimension_of_labels(experiment: SimulationExperiment) -> None:
    spec = _plot(
        experiment,
        lambda p: p.curve(
            x=Data("time", task="task_cond"),
            y=Data("[C]", task="task_cond"),
            over="cond",
        ),
    )
    colorbar = spec["colorbar"]
    assert colorbar["label"] == "cond"
    scale = colorbar["scale"]
    assert scale["low"] == 0.0
    colors = import_core().color_scale(
        "viridis", scale["low"], scale["high"], list(range(12))
    )
    assert [e["style"]["color"] for e in spec["elements"]] == list(colors)


def test_a_curve_of_its_own_colour_gets_no_colorbar(
    experiment: SimulationExperiment,
) -> None:
    spec = _plot(experiment, lambda p: _doses(p, task="task_many", color="tab:red"))
    assert "colorbar" not in spec
    assert len(spec["elements"]) == 12
