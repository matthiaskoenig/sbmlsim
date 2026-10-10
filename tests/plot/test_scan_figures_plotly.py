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


def test_two_dimensions_dash(experiment: SimulationExperiment) -> None:
    traces = _traces(experiment, lambda p: _curve(p, "task_grid2", ("dose", "rate")))
    assert len(traces) == 6
    assert [t.line.dash for t in traces] == ["solid", "dash"] * 3
    assert [t.name for t in traces][:2] == [
        "C, PODOSE = 50 mg, ke = 0.1 1/hr",
        "C, PODOSE = 50 mg, ke = 0.3 1/hr",
    ]
    assert all(t.legendgroup for t in traces)


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
    (median,) = [t for t in traces if t.name == "C median"]
    assert median.showlegend
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
