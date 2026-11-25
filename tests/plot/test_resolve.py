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


def _bar_colors(colorbar: dict, values: Any) -> list[str]:
    """The colours of a colour bar at values, with the part of its map."""
    scale = colorbar["scale"]
    return list(
        import_core().color_scale(
            scale["name"],
            scale["low"],
            scale["high"],
            list(values),
            log=scale["log"],
            start=scale.get("start", 0.0),
            end=scale.get("end", 1.0),
        )
    )


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
        yaxis=Axis("HCTZ\nurine\n", unit="mg/l", scale="log"),
    )
    assert spec["x"]["label"] == "time"
    assert spec["x"]["unit"] == "min"
    # the data is converted to the unit of the axis: 24 hr are 1440 min
    assert max(spec["elements"][0]["x"]) == pytest.approx(1440.0)
    # a line break inside a label is kept, the one at its end is not
    assert spec["y"]["label"] == "HCTZ\nurine"
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
        assert "tick_marks_visible" not in spec[axis]


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
    # matplotlib kept the marks of the ticks whose labels it hid
    assert spec["y"]["tick_marks_visible"] is True
    assert "tick_marks_visible" not in spec["x"]


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
    # the bar spans the data and no value which no line has, in the part of
    # the map the lines take
    assert scale["low"] == pytest.approx(10.0)
    assert scale["high"] == pytest.approx(120.0)
    assert "start" not in scale
    assert scale["end"] == 0.85
    assert "categorical" not in colorbar
    # the colours of the lines are the colours of the bar at their values
    values = np.linspace(10.0, 120.0, 12)
    assert [e["style"]["color"] for e in elements] == _bar_colors(colorbar, values)
    # the swatch of the shared entry is the middle colour of the bar
    (group,) = {e["legend"] for e in elements}
    (middle,) = import_core().color_scale("viridis", 0.0, 1.0, [0.425])
    assert spec["legend_swatches"][group]["color"] == middle
    svg = chart.svg()
    assert "139" not in svg


def test_a_logarithmic_colorbar_of_geometric_doses(
    experiment: SimulationExperiment,
) -> None:
    spec = _plot(experiment, lambda p: _doses(p, task="task_geom"))
    scale = spec["colorbar"]["scale"]
    assert scale["log"] is True
    assert (scale["low"], scale["high"]) == pytest.approx((1.0, 1000.0))
    colors = _bar_colors(spec["colorbar"], np.geomspace(1, 1000, 12))
    assert [e["style"]["color"] for e in spec["elements"]] == colors
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
    geometry = import_core().figure_geometry(chart.to_json())
    assert geometry["plots"][0]["y"]["low"] > 0.0


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
        "tick_y": defaults.ytick_labelsize,
        "legend": defaults.legend_fontsize,
        "title_weight": "bold",
        "label_weight": "bold",
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
    # a band per point, labelled by its label
    assert colorbar["categorical"] is True
    assert colorbar["tick_labels"] == [str(10 * k) for k in range(12)]
    scale = colorbar["scale"]
    assert (scale["low"], scale["high"]) == (0.0, 11.0)
    colors = _bar_colors(colorbar, range(12))
    assert [e["style"]["color"] for e in spec["elements"]] == colors


def test_a_curve_of_its_own_colour_shares_the_middle_shade(
    experiment: SimulationExperiment,
) -> None:
    spec = _plot(
        experiment, lambda p: _doses(p, task="task_many", color="tab:red", label="C")
    )
    # no colour map of the renderer has the shades of a colour
    assert "colorbar" not in spec
    assert len(spec["elements"]) == 12
    colors = [e["style"]["color"] for e in spec["elements"]]
    (swatch,) = spec["legend_swatches"].values()
    assert swatch["color"] not in ("#d62728", colors[0])


def test_every_colour_bar_of_a_plot_is_kept(experiment: SimulationExperiment) -> None:
    def draw(plot: Plot) -> None:
        _doses(plot, task="task_many", label="linear")
        _doses(plot, task="task_many", label="again", linestyle="--")
        _doses(plot, task="task_geom", label="geometric")

    chart, spec = _chart_of(experiment, draw)
    assert "colorbar" not in spec
    # the same bar of two curves is shown once
    bars = spec["colorbars"]
    assert [b["scale"]["log"] for b in bars] == [False, True]
    assert all(b["label"] == "PODOSE [mg]" for b in bars)
    chart.svg()


def test_the_ticks_of_x_and_y_take_their_own_sizes(
    experiment: SimulationExperiment,
) -> None:
    figure = _figure(experiment, _doses, xtick_labelsize=12, ytick_labelsize=17)
    fonts = figure_chart(experiment, figure).spec()["fonts"]
    assert (fonts["tick"], fonts["tick_y"]) == (12.0, 17.0)


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ({}, {"title_weight": "bold", "label_weight": "bold"}),
        (
            {
                "fig_titleweight": "normal",
                "axes_titleweight": "normal",
                "axes_labelweight": "normal",
            },
            # written, since the title of the renderer is bold without
            {"title_weight": "normal", "label_weight": "normal"},
        ),
        (
            {"fig_titleweight": "normal", "axes_labelweight": 700},
            {"title_weight": "bold", "label_weight": "bold"},
        ),
        ({"axes_titleweight": "light", "fig_titleweight": "semibold"}, None),
    ],
)
def test_the_font_weights(
    experiment: SimulationExperiment, weights: dict, expected: dict | None
) -> None:
    figure = _figure(experiment, _doses, **weights)
    fonts = figure_chart(experiment, figure).spec()["fonts"]
    found = {k: v for k, v in fonts.items() if k.endswith("_weight")}
    if expected is not None:
        assert found == expected
    else:
        assert found == {"title_weight": "bold", "label_weight": "bold"}


def test_the_right_axis_draws_no_grid(experiment: SimulationExperiment) -> None:
    def draw(plot: Plot) -> None:
        plot.set_yaxis_right("C", unit="g/l", grid=True)
        plot.curve(
            x=Data("time", task="task_doses", sel={"dose": 0}),
            y=Data("[C]", task="task_doses", sel={"dose": 0}),
            yaxis_position=YAxisPosition.RIGHT,
        )

    spec = _plot(experiment, draw, yaxis=Axis("C", unit="mg/l", grid=True))
    assert spec["y"]["grid"] is True
    assert spec["y2"]["grid"] is False


# ---------------------------------------------------------------------------
# ported from the tests of the serializers of matplotlib and plotly (what is
# drawn, not how a library draws it)
# ---------------------------------------------------------------------------
def _chart_of(experiment: SimulationExperiment, draw: Callable[[Plot], object]) -> Any:
    chart = figure_chart(experiment, _figure(experiment, draw))
    return chart, chart.spec()["plots"][0]


def test_two_curves_over_one_dimension_keep_their_names_in_the_legend(
    experiment: SimulationExperiment,
) -> None:
    def draw(plot: Plot) -> None:
        _doses(plot, label="C")
        _doses(plot, label="C again", color="tab:red")

    chart, _ = _chart_of(experiment, draw)
    labels = _legend(chart.svg())
    assert labels[0].startswith("C, ") and labels[3].startswith("C again, ")


def test_a_value_per_simulation_over_its_dimension(
    experiment: SimulationExperiment,
) -> None:
    spec = _plot(
        experiment,
        lambda p: p.curve(
            x=Data("dose.PODOSE", task="task_doses"),
            y=Data("pk.cmax", task="task_doses"),
            label="cmax",
        ),
        xaxis=Axis("PODOSE", unit="mg"),
        yaxis=Axis("cmax", unit="mg/l"),
    )
    (line,) = spec["elements"]
    assert line["x"] == pytest.approx([50.0, 100.0, 200.0])
    assert np.all(np.diff(line["y"]) > 0)


def test_a_band_on_the_right_axis(experiment: SimulationExperiment) -> None:
    def draw(plot: Plot) -> None:
        plot.set_yaxis_right("C right", unit="mg/l")
        plot.band(
            Data("time", task="task_draws"),
            Data("[C]", task="task_draws"),
            across="draw",
            name="C",
            yaxis_position=YAxisPosition.RIGHT,
        )

    (band,) = _plot(experiment, draw)["elements"]
    assert (band["kind"], band["axis"]) == ("band", "y2")


def test_selected_points_are_labelled_by_their_labels(
    experiment: SimulationExperiment,
) -> None:
    def draw(plot: Plot) -> None:
        plot.curve(
            x=Data("time", task="task_doses", sel={"dose": [2, 1]}),
            y=Data("[C]", task="task_doses", sel={"dose": [2, 1]}),
            over="dose",
            label="C",
        )

    chart, spec = _chart_of(experiment, draw)
    assert _legend(chart.svg()) == ["C, PODOSE = 200 mg", "C, PODOSE = 100 mg"]
    high, low = spec["elements"]
    assert np.nanmax(np.asarray(high["y"], dtype=float)) > np.nanmax(
        np.asarray(low["y"], dtype=float)
    )
    # the two points are the two ends of the colours
    ends = import_core().color_scale("viridis", 0.0, 1.0, [0.85, 0.0])
    assert [high["style"]["color"], low["style"]["color"]] == list(ends)


def test_a_selected_second_dimension_draws_with_its_own_labels(
    experiment: SimulationExperiment,
) -> None:
    def draw(plot: Plot) -> None:
        plot.curve(
            x=Data("time", task="task_grid5", sel={"rate": [3, 1]}),
            y=Data("[C]", task="task_grid5", sel={"rate": [3, 1]}),
            over=("dose", "rate"),
            label="C",
        )

    chart, spec = _chart_of(experiment, draw)
    assert len([e for e in spec["elements"] if e["x"]]) == 6
    unit = experiment.model_units("task_grid5").get("ke", "")
    assert _legend(chart.svg())[3:] == [
        f"ke = 0.4 {unit}".rstrip(),
        f"ke = 0.2 {unit}".rstrip(),
    ]


def test_decreasing_doses_keep_line_and_bar_consistent(
    experiment: SimulationExperiment,
) -> None:
    spec = _plot(experiment, lambda p: _doses(p, task="task_decreasing"))
    scale = spec["colorbar"]["scale"]
    assert scale["log"] is False
    assert (scale["low"], scale["high"]) == pytest.approx((10.0, 120.0))
    colors = _bar_colors(spec["colorbar"], np.linspace(120.0, 10.0, 12))
    assert [e["style"]["color"] for e in spec["elements"]] == colors


def test_a_named_band_with_a_colour_bar_keeps_its_legend_entry(
    experiment: SimulationExperiment,
) -> None:
    chart, spec = _chart_of(
        experiment,
        lambda p: p.band(
            Data("time", task="task_many_draws"),
            Data("[C]", task="task_many_draws"),
            across="draw",
            over="dose",
            name="C",
        ),
    )
    assert _legend(chart.svg()) == ["C (5-95 %)"]
    assert spec["colorbar"]["label"] == "PODOSE [mg]"


def _two_bands(plot: Plot, quantiles: tuple[float, float] = (0.25, 0.75)) -> None:
    for name, q in [("a", (0.05, 0.95)), ("b", quantiles)]:
        plot.band(
            Data("time", task="task_draws"),
            Data("[C]", task="task_draws"),
            across="draw",
            name=name,
            quantiles=q,
        )


def test_two_bands_have_an_entry_each(experiment: SimulationExperiment) -> None:
    chart, _ = _chart_of(experiment, _two_bands)
    assert _legend(chart.svg()) == ["a (5-95 %)", "b (25-75 %)"]
    chart, _ = _chart_of(experiment, lambda p: _two_bands(p, quantiles=(0.05, 0.95)))
    assert _legend(chart.svg()) == ["a (5-95 %)", "b (5-95 %)"]


@pytest.mark.parametrize("median", [True, False], ids=["median", "no median"])
def test_the_legend_has_one_entry_per_band_with_its_area_in_its_colour(
    experiment: SimulationExperiment, median: bool
) -> None:
    """A band has one legend entry, its area in its colour, with its quantiles (#302)."""

    def draw(plot: Plot) -> None:
        for name, color, quantiles in [
            ("a", "black", (0.0, 1.0)),
            ("b", "blue", (0.05, 0.95)),
        ]:
            plot.band(
                Data("time", task="task_draws"),
                Data("[C]", task="task_draws"),
                across="draw",
                quantiles=quantiles,
                median=median,
                name=name,
                color=color,
            )
        plot.curve(
            x=Data("time", task="task_doses", sel={"dose": 0}),
            y=Data("[C]", task="task_doses", sel={"dose": 0}),
            label="sim",
            color="black",
        )

    chart, spec = _chart_of(experiment, draw)
    assert _legend(chart.svg()) == ["a (0-100 %)", "b (5-95 %)", "sim"]
    a, b, _ = spec["elements"]
    assert ("median" in a, "median" in b) == (median, median)
    assert (a["style"]["color"], b["style"]["color"]) == ("#000000", "#0000ff")
    assert a["style"]["fill"]["opacity"] == pytest.approx(0.3)


def test_the_median_of_a_band_without_a_name(experiment: SimulationExperiment) -> None:
    """A band without a name has the entry of its range."""

    def draw(plot: Plot) -> None:
        band = plot.band(
            Data("time", task="task_draws"),
            Data("[C]", task="task_draws"),
            across="draw",
        )
        band.name = None

    chart, spec = _chart_of(experiment, draw)
    (band,) = spec["elements"]
    assert "median" in band
    assert _legend(chart.svg()) == ["5-95 %"]


def test_the_bands_of_two_panels_keep_their_own_legend_groups(
    experiment: SimulationExperiment,
) -> None:
    figure = Figure(experiment=experiment, sid="fig", num_rows=1, num_cols=2)
    plots = figure.create_plots(
        xaxis=Axis("time", unit="hr"), yaxis=Axis("C", unit="mg/l"), legend=True
    )
    for plot, name in zip(plots, ("a", "b"), strict=True):
        plot.band(
            Data("time", task="task_draws"),
            Data("[C]", task="task_draws"),
            across="draw",
            name=name,
        )
    chart = figure_chart(experiment, figure)
    groups = [p["elements"][0]["legend"] for p in chart.spec()["plots"]]
    assert groups[0] != groups[1]
    assert _legend(chart.svg()) == ["a (5-95 %)", "b (5-95 %)"]


def test_the_legend_entries_of_two_dimensions_carry_the_markers(
    experiment: SimulationExperiment,
) -> None:
    spec = _plot(
        experiment,
        lambda p: p.curve(
            x=Data("time", task="task_grid2"),
            y=Data("[C]", task="task_grid2"),
            over=("dose", "rate"),
            label="C",
            marker="o",
        ),
    )
    assert {e["style"]["marker"]["type"] for e in spec["elements"]} == {"circle"}
