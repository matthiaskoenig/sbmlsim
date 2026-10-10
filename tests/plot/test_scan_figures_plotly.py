"""Plotly draws curves over the points of a scan and bands like matplotlib."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib.colors import LogNorm, to_hex

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.plot import Axis, Figure, Plot
from sbmlsim.plot.plotting import YAxisPosition
from sbmlsim.plot.points import point_colormap
from sbmlsim.simulator import Simulator
from tests.plot.scan_experiment import ScanFigures

pytest.importorskip("plotly")

from sbmlsim.plot.serialization_plotly import PlotlyFigureSerializer


@pytest.fixture(scope="module")
def experiment() -> SimulationExperiment:
    runner = ExperimentRunner(
        experiment_classes=[ScanFigures],
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
    )
    experiment = runner.experiments["ScanFigures"]
    experiment.run(runner.simulator)
    return experiment


def _traces(experiment: SimulationExperiment, draw: Callable[[Plot], Any]) -> list[Any]:
    figure = Figure(experiment=experiment, sid="fig", num_rows=1, num_cols=1)
    plot = figure.create_plots(
        xaxis=Axis("time", unit="hr"), yaxis=Axis("C", unit="mg/l"), legend=True
    )[0]
    draw(plot)
    return list(PlotlyFigureSerializer.to_figure(experiment, figure).data)


def _curve(plot: Plot, task: str, over: Any, **kwargs: Any) -> None:
    """A curve of the concentration over `over`, labelled C unless given."""
    sel = kwargs.pop("sel", {})
    options: dict[str, Any] = {"label": "C", **kwargs}
    plot.curve(
        x=Data("time", task=task, sel=sel),
        y=Data("[C]", task=task, sel=sel),
        over=over,
        **options,
    )


def test_one_trace_per_dose(experiment: SimulationExperiment) -> None:
    traces = _traces(experiment, lambda p: _curve(p, "task_doses", "dose"))
    assert [t.name for t in traces] == [
        "C, PODOSE = 50 mg",
        "C, PODOSE = 100 mg",
        "C, PODOSE = 200 mg",
    ]
    cmap = point_colormap(None)
    # the colour map at the values 50, 100 and 200 mg
    assert [t.line.color for t in traces] == [
        to_hex(cmap(f)) for f in (0.0, 1 / 3, 1.0)
    ]
    assert all(t.showlegend for t in traces)


def test_eleven_points_and_more_get_a_colour_bar(
    experiment: SimulationExperiment,
) -> None:
    traces = _traces(experiment, lambda p: _curve(p, "task_many", "dose"))
    lines = [t for t in traces if t.x is not None and t.x[0] is not None]
    assert len(lines) == 12 and not any(t.showlegend for t in lines)
    (bar,) = [t for t in traces if t.marker.showscale]
    assert bar.marker.colorbar.title.text == "PODOSE [mg]"
    assert not bar.showlegend


def test_two_dimensions_legend_like_matplotlib(
    experiment: SimulationExperiment,
) -> None:
    traces = _traces(experiment, lambda p: _curve(p, "task_grid2", ("dose", "rate")))
    lines = [t for t in traces if t.x[0] is not None]
    assert len(lines) == 6 and not any(t.showlegend for t in lines)
    assert [t.line.dash for t in lines] == ["solid", "dash"] * 3
    entries = [t for t in traces if t.showlegend]
    assert [t.name for t in entries] == [
        "C, PODOSE = 50 mg",
        "C, PODOSE = 100 mg",
        "C, PODOSE = 200 mg",
        "ke = 0.1 1/hr",
        "ke = 0.3 1/hr",
    ]
    assert [t.line.dash for t in entries[3:]] == ["solid", "dash"]


def test_many_doses_and_two_dimensions_have_dash_entries_and_a_bar(
    experiment: SimulationExperiment,
) -> None:
    traces = _traces(experiment, lambda p: _curve(p, "task_many2", ("dose", "rate")))
    assert [t.name for t in traces if t.showlegend] == [
        "C",
        "ke = 0.1 1/hr",
        "ke = 0.3 1/hr",
    ]
    assert sum(1 for t in traces if t.marker.showscale) == 1


def test_a_colour_bar_per_panel(experiment: SimulationExperiment) -> None:
    figure = Figure(experiment=experiment, sid="fig", num_rows=1, num_cols=2)
    for plot in figure.create_plots(
        xaxis=Axis("time", unit="hr"), yaxis=Axis("C", unit="mg/l")
    ):
        _curve(plot, "task_many", "dose")
    fig = PlotlyFigureSerializer.to_figure(experiment, figure)
    bars = [t.marker.colorbar for t in fig.data if t.marker.showscale]
    assert len(bars) == 2 and bars[0].x != bars[1].x
    for bar, col in zip(bars, (1, 2), strict=True):
        end = fig.get_subplot(1, col).xaxis.domain[1]
        assert end < bar.x < 1


def test_a_band(experiment: SimulationExperiment) -> None:
    traces = _traces(
        experiment,
        lambda p: p.band(
            Data("time", task="task_draws"),
            Data("[C]", task="task_draws"),
            across="draw",
            name="C",
        ),
    )
    assert [t.fill for t in traces].count("tonexty") == 1
    (median,) = [t for t in traces if t.name == "C median" and t.showlegend]
    assert median.showlegend
    (upper,) = [t for t in traces if t.fill == "tonexty"]
    assert not upper.showlegend and upper.name == "C median"
    (rng,) = [t for t in traces if t.name == "5-95 %"]
    assert rng.showlegend and rng.x == (None,)
    assert sum(1 for t in traces if t.showlegend) == 2


def test_bands_over_doses(experiment: SimulationExperiment) -> None:
    traces = _traces(
        experiment,
        lambda p: p.band(
            Data("time", task="task_dose_draws"),
            Data("[C]", task="task_dose_draws"),
            across="draw",
            over="dose",
            name="C",
        ),
    )
    assert [t.fill for t in traces].count("tonexty") == 3
    names = [t.name for t in traces if t.showlegend]
    assert names[:3] == [
        "C, PODOSE = 50 mg",
        "C, PODOSE = 100 mg",
        "C, PODOSE = 200 mg",
    ]


def test_a_band_on_the_right_axis(experiment: SimulationExperiment) -> None:
    figure = Figure(experiment=experiment, sid="fig", num_rows=1, num_cols=1)
    plot = figure.create_plots(
        xaxis=Axis("time", unit="hr"), yaxis=Axis("C", unit="mg/l"), legend=True
    )[0]
    plot.yaxis_right = Axis("C right", unit="mg/l")
    plot.band(
        Data("time", task="task_draws"),
        Data("[C]", task="task_draws"),
        across="draw",
        name="C",
        yaxis_position=YAxisPosition.RIGHT,
    )
    fig = PlotlyFigureSerializer.to_figure(experiment, figure)
    drawn = [t for t in fig.data if t.x is not None and t.x[0] is not None]
    assert len(drawn) == 3 and {t.yaxis for t in drawn} == {"y2"}


def test_a_band_without_median_names_the_upper_line(
    experiment: SimulationExperiment,
) -> None:
    traces = _traces(
        experiment,
        lambda p: p.band(
            Data("time", task="task_dose_draws"),
            Data("[C]", task="task_dose_draws"),
            across="draw",
            over="dose",
            name="C",
            median=False,
        ),
    )
    assert [t.name for t in traces if t.showlegend] == [
        "C, PODOSE = 50 mg",
        "C, PODOSE = 100 mg",
        "C, PODOSE = 200 mg",
        "5-95 %",
    ]
    assert all(t.fill == "tonexty" for t in traces if t.showlegend and t.fill)


def _bars(fig: Any) -> list[Any]:
    return [t.marker.colorbar for t in fig.data if t.marker.showscale]


def _plot_width(fig: Any) -> float:
    """The width of the plot area in pixels, the unit of the paper coordinates."""
    margin = fig.layout.margin
    return fig.layout.width - margin.l - margin.r


def _lines(traces: list[Any]) -> list[Any]:
    return [t for t in traces if t.x is not None and t.x[0] is not None]


def test_selected_points_are_named_by_their_labels(
    experiment: SimulationExperiment,
) -> None:
    traces = _traces(
        experiment, lambda p: _curve(p, "task_doses", "dose", sel={"dose": [2, 1]})
    )
    assert [t.name for t in traces] == ["C, PODOSE = 200 mg", "C, PODOSE = 100 mg"]
    cmap = point_colormap(None)
    assert [t.line.color for t in traces] == [to_hex(cmap(1.0)), to_hex(cmap(0.0))]


def test_a_selected_second_dimension_draws_with_its_own_labels(
    experiment: SimulationExperiment,
) -> None:
    traces = _traces(
        experiment,
        lambda p: _curve(p, "task_grid5", ("dose", "rate"), sel={"rate": [3, 1]}),
    )
    assert len(_lines(traces)) == 6
    entries = [t.name for t in traces if t.showlegend]
    assert entries[3:] == ["ke = 0.4 1/hr", "ke = 0.2 1/hr"]


def test_geometric_doses_get_a_logarithmic_colour_bar(
    experiment: SimulationExperiment,
) -> None:
    traces = _traces(experiment, lambda p: _curve(p, "task_geom", "dose"))
    (bar,) = [t for t in traces if t.marker.showscale]
    # the bar spans the decades of the doses on a logarithmic axis
    assert list(bar.marker.color) == pytest.approx([0.0, 3.0])
    ticks = dict(
        zip(bar.marker.colorbar.ticktext, bar.marker.colorbar.tickvals, strict=True)
    )
    assert {"1", "10", "100", "1000"} <= set(ticks)
    assert ticks["100"] == pytest.approx(2.0)
    values = np.geomspace(1, 1000, 12)
    norm, cmap = LogNorm(1.0, 1000.0), point_colormap(None)
    assert [t.line.color for t in _lines(traces)] == [
        to_hex(cmap(float(norm(v)))) for v in values
    ]
    assert all(c == to_hex(cmap(f)) for f, c in bar.marker.colorscale)


def test_decreasing_doses_keep_line_and_bar_consistent(
    experiment: SimulationExperiment,
) -> None:
    traces = _traces(experiment, lambda p: _curve(p, "task_decreasing", "dose"))
    (bar,) = [t for t in traces if t.marker.showscale]
    assert list(bar.marker.color) == pytest.approx([10.0, 120.0])
    cmap = point_colormap(None)
    colors = [t.line.color for t in _lines(traces)]
    assert colors[0] == to_hex(cmap(1.0)) and colors[-1] == to_hex(cmap(0.0))
    assert all(c == to_hex(cmap(f)) for f, c in bar.marker.colorscale)


def test_a_dimension_of_labels_shows_its_labels_on_the_colour_bar(
    experiment: SimulationExperiment,
) -> None:
    traces = _traces(experiment, lambda p: _curve(p, "task_cond", "cond"))
    (bar,) = [t for t in traces if t.marker.showscale]
    colorbar = bar.marker.colorbar
    assert colorbar.title.text == "cond"
    assert list(colorbar.ticktext) == [str(10 * k) for k in range(12)]
    assert list(colorbar.tickvals) == list(range(12))
    # one step of the scale per point, in the colour of its line
    steps = [c for _, c in bar.marker.colorscale][::2]
    assert steps == [t.line.color for t in _lines(traces)]


def test_a_named_curve_with_a_colour_bar_keeps_its_legend_entry(
    experiment: SimulationExperiment,
) -> None:
    traces = _traces(experiment, lambda p: _curve(p, "task_many", "dose"))
    (entry,) = [t for t in traces if t.showlegend]
    assert entry.name == "C"
    assert entry.line.color == to_hex(point_colormap(None)(0.5))


def _two_bars(plot: Plot, right: bool = False) -> None:
    """Two curves with a colour bar each, the second on the right axis if `right`."""
    _curve(plot, "task_many", "dose")
    _curve(
        plot,
        "task_geom",
        "dose",
        label="C geom",
        color="tab:red",
        yaxis_position=YAxisPosition.RIGHT if right else None,
    )


def test_two_colour_bars_of_a_panel_sit_side_by_side(
    experiment: SimulationExperiment,
) -> None:
    figure = Figure(experiment=experiment, sid="fig", num_rows=1, num_cols=1)
    plot = figure.create_plots(
        xaxis=Axis("time", unit="hr"), yaxis=Axis("C", unit="mg/l"), legend=True
    )[0]
    _two_bars(plot)
    fig = PlotlyFigureSerializer.to_figure(experiment, figure)
    first, second = _bars(fig)
    end = fig.get_subplot(1, 1).xaxis.domain[1]
    assert end < first.x < second.x
    # the first bar keeps the room of its ticks and its title
    assert (second.x - first.x) * _plot_width(fig) >= 80
    assert first.title.side == second.title.side == "right"


def test_a_colour_bar_is_right_of_the_title_of_a_right_axis(
    experiment: SimulationExperiment,
) -> None:
    figure = Figure(experiment=experiment, sid="fig", num_rows=1, num_cols=2)
    plots = figure.create_plots(
        xaxis=Axis("time", unit="hr"), yaxis=Axis("C", unit="mg/l"), legend=True
    )
    plots[0].yaxis_right = Axis("C right", unit="mg/l")
    _two_bars(plots[0], right=True)
    _curve(plots[1], "task_doses", "dose")
    fig = PlotlyFigureSerializer.to_figure(experiment, figure)
    first, second = _bars(fig)
    end = fig.get_subplot(1, 1).xaxis.domain[1]
    # the tick labels and the title of the right axis are right of the panel
    assert (first.x - end) * _plot_width(fig) >= 60
    assert (second.x - first.x) * _plot_width(fig) >= 80
    # both bars and their titles are left of the next panel
    start = fig.get_subplot(1, 2).xaxis.domain[0]
    assert (start - second.x) * _plot_width(fig) >= 80


def test_the_legend_is_right_of_every_panel_and_colour_bar(
    experiment: SimulationExperiment,
) -> None:
    figure = Figure(experiment=experiment, sid="fig", num_rows=1, num_cols=2)
    plots = figure.create_plots(
        xaxis=Axis("time", unit="hr"), yaxis=Axis("C", unit="mg/l"), legend=True
    )
    _curve(plots[0], "task_many", "dose")
    _curve(plots[1], "task_doses", "dose")
    fig = PlotlyFigureSerializer.to_figure(experiment, figure)
    (bar,) = _bars(fig)
    ends = [fig.get_subplot(1, col).xaxis.domain[1] for col in (1, 2)]
    legend = fig.layout.legend
    assert legend.xanchor == "left"
    assert legend.x > max([*ends, bar.x])
    # the bar of the first panel is inside its cell, left of the second panel
    assert bar.x < fig.get_subplot(1, 2).xaxis.domain[0]


def _two_bands(plot: Plot, quantiles: tuple[float, float] = (0.25, 0.75)) -> None:
    plot.band(
        Data("time", task="task_draws"),
        Data("[C]", task="task_draws"),
        across="draw",
        name="a",
    )
    plot.band(
        Data("time", task="task_draws"),
        Data("[C]", task="task_draws"),
        across="draw",
        name="b",
        quantiles=quantiles,
    )


def test_bands_without_a_colour_take_the_colour_cycle(
    experiment: SimulationExperiment,
) -> None:
    traces = _traces(experiment, _two_bands)
    medians = {
        t.name: t.line.color
        for t in traces
        if t.showlegend and t.name.endswith("median")
    }
    assert medians == {"a median": to_hex("C0"), "b median": to_hex("C1")}
    assert [t.name for t in traces if t.showlegend] == [
        "a median",
        "5-95 %",
        "b median",
        "25-75 %",
    ]


def test_the_range_entry_appears_once_per_range(
    experiment: SimulationExperiment,
) -> None:
    traces = _traces(experiment, lambda p: _two_bands(p, quantiles=(0.05, 0.95)))
    assert [t.name for t in traces if t.showlegend] == [
        "a median",
        "5-95 %",
        "b median",
    ]


def test_the_legend_shows_an_entry_of_several_panels_once(
    experiment: SimulationExperiment,
) -> None:
    figure = Figure(experiment=experiment, sid="fig", num_rows=1, num_cols=2)
    plots = figure.create_plots(
        xaxis=Axis("time", unit="hr"), yaxis=Axis("C", unit="mg/l"), legend=True
    )
    for plot in plots:
        _curve(plot, "task_doses", (), sel={"dose": 0}, label="simulation", color="k")
    _curve(plots[1], "task_doses", (), sel={"dose": 1}, label="other", color="k")
    fig = PlotlyFigureSerializer.to_figure(experiment, figure)
    assert [t.name for t in fig.data if t.showlegend] == ["simulation", "other"]
    simulations = [t for t in fig.data if t.name == "simulation"]
    assert len(simulations) == 2
    assert simulations[0].legendgroup == simulations[1].legendgroup is not None
