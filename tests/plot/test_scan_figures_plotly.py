"""Plotly draws curves over the points of a scan and bands like matplotlib."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.plot import Axis, Figure, Plot
from sbmlsim.plot.plotting import YAxisPosition
from sbmlsim.plot.points import point_colors
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


def _curve(plot: Plot, task: str, over: Any) -> None:
    plot.curve(
        x=Data("time", task=task),
        y=Data("[C]", task=task),
        over=over,
        label="C",
    )


def test_one_trace_per_dose(experiment: SimulationExperiment) -> None:
    traces = _traces(experiment, lambda p: _curve(p, "task_doses", "dose"))
    assert [t.name for t in traces] == [
        "C, PODOSE = 50 mg",
        "C, PODOSE = 100 mg",
        "C, PODOSE = 200 mg",
    ]
    assert [t.line.color for t in traces] == point_colors(3, None)
    assert all(t.showlegend for t in traces)


def test_eleven_points_and_more_get_a_colour_bar(
    experiment: SimulationExperiment,
) -> None:
    traces = _traces(experiment, lambda p: _curve(p, "task_many", "dose"))
    lines = [t for t in traces if t.x is not None and t.x[0] is not None]
    assert len(lines) == 12 and not any(t.showlegend for t in lines)
    (bar,) = [t for t in traces if t not in lines]
    assert bar.marker.showscale and bar.marker.colorbar.title.text == "PODOSE [mg]"
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
