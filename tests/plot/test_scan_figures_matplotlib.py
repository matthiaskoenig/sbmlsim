"""Matplotlib draws curves over the points of a scan and bands."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.figure import Figure as FigureMPL
from matplotlib.lines import Line2D

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.plot import Axis, Figure, Plot
from sbmlsim.plot.plotting import CurveType
from sbmlsim.plot.points import point_colors
from sbmlsim.plot.serialization_matplotlib import MatplotlibFigureSerializer
from sbmlsim.simulator import Simulator
from tests.plot.scan_experiment import ScanFigures


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


def _axes(
    experiment: SimulationExperiment,
    draw: Callable[[Plot], object],
    xaxis: Axis | None = None,
    yaxis: Axis | None = None,
) -> tuple[FigureMPL, Axes]:
    figure = Figure(experiment=experiment, sid="fig", num_rows=1, num_cols=1)
    plot = figure.create_plots(
        xaxis=xaxis or Axis("time", unit="hr"),
        yaxis=yaxis or Axis("C", unit="mg/l"),
        legend=True,
    )[0]
    draw(plot)
    fig = MatplotlibFigureSerializer.to_figure(experiment, figure)
    return fig, fig.axes[0]


def _labels(ax: Axes) -> list[str]:
    legend = ax.get_legend()
    return [] if legend is None else [t.get_text() for t in legend.get_texts()]


def _data_lines(ax: Axes) -> list[Line2D]:
    return [line for line in ax.get_lines() if np.size(line.get_xdata())]


def test_one_line_per_dose_in_viridis_with_labels(
    experiment: SimulationExperiment,
) -> None:
    fig, ax = _axes(
        experiment,
        lambda p: p.curve(
            x=Data("time", task="task_doses"),
            y=Data("[C]", task="task_doses"),
            over="dose",
            label="C",
        ),
    )
    lines = _data_lines(ax)
    assert len(lines) == 3
    assert [to_hex(line.get_color()) for line in lines] == point_colors(3, None)
    assert _labels(ax) == [
        "C, PODOSE = 50 mg",
        "C, PODOSE = 100 mg",
        "C, PODOSE = 200 mg",
    ]
    assert len(fig.axes) == 1  # no colour bar


def test_shades_of_the_colour_of_the_curve(experiment: SimulationExperiment) -> None:
    _, ax = _axes(
        experiment,
        lambda p: p.curve(
            x=Data("time", task="task_doses"),
            y=Data("[C]", task="task_doses"),
            over="dose",
            color="tab:red",
        ),
    )
    colors = [to_hex(line.get_color()) for line in _data_lines(ax)]
    assert colors == point_colors(3, "tab:red")


def test_eleven_points_and_more_get_a_colour_bar(
    experiment: SimulationExperiment,
) -> None:
    fig, ax = _axes(
        experiment,
        lambda p: p.curve(
            x=Data("time", task="task_many"),
            y=Data("[C]", task="task_many"),
            over="dose",
            label="C",
        ),
    )
    assert len(_data_lines(ax)) == 12
    assert _labels(ax) == []
    assert len(fig.axes) == 2 and fig.axes[1].get_ylabel() == "PODOSE [mg]"


def test_two_dimensions_colour_and_line_style(
    experiment: SimulationExperiment,
) -> None:
    _, ax = _axes(
        experiment,
        lambda p: p.curve(
            x=Data("time", task="task_grid2"),
            y=Data("[C]", task="task_grid2"),
            over=("dose", "rate"),
            label="C",
        ),
    )
    lines = _data_lines(ax)
    assert len(lines) == 6
    assert {line.get_linestyle() for line in lines} == {"-", "--"}
    assert _labels(ax)[:3] == [
        "C, PODOSE = 50 mg",
        "C, PODOSE = 100 mg",
        "C, PODOSE = 200 mg",
    ]
    unit = experiment.model_units("task_grid2").get("ke", "")
    assert _labels(ax)[3:] == [f"ke = 0.1 {unit}".rstrip(), f"ke = 0.3 {unit}".rstrip()]


def test_two_curves_over_one_dimension_keep_their_names_in_the_legend(
    experiment: SimulationExperiment,
) -> None:
    def draw(plot: Plot) -> None:
        plot.curve(
            x=Data("time", task="task_doses"),
            y=Data("[C]", task="task_doses"),
            over="dose",
            label="C",
        )
        plot.curve(
            x=Data("time", task="task_doses"),
            y=Data("[C]", task="task_doses"),
            over="dose",
            label="C again",
            color="tab:red",
        )

    _, ax = _axes(experiment, draw)
    labels = _labels(ax)
    assert labels[0].startswith("C, ") and labels[3].startswith("C again, ")


def test_a_value_per_simulation_over_its_dimension(
    experiment: SimulationExperiment,
) -> None:
    _, ax = _axes(
        experiment,
        lambda p: p.curve(
            x=Data("dose.PODOSE", task="task_doses"),
            y=Data("pk.cmax", task="task_doses"),
            label="cmax",
        ),
        xaxis=Axis("PODOSE", unit="mg"),
        yaxis=Axis("cmax", unit="mg/l"),
    )
    (line,) = _data_lines(ax)
    np.testing.assert_allclose(np.asarray(line.get_xdata()), [50.0, 100.0, 200.0])
    assert np.all(np.diff(line.get_ydata()) > 0)


def test_a_band_with_its_median(experiment: SimulationExperiment) -> None:
    _, ax = _axes(
        experiment,
        lambda p: p.band(
            Data("time", task="task_draws"),
            Data("[C]", task="task_draws"),
            across="draw",
            name="C",
        ),
    )
    assert len(ax.collections) == 1
    (median,) = _data_lines(ax)
    y = Data("[C]", task="task_draws").get_data(experiment, to_units="mg/l")
    np.testing.assert_allclose(
        np.asarray(median.get_ydata()), np.nanmedian(y.values, axis=0)
    )
    assert sorted(_labels(ax)) == ["C 5-95 %", "C median"]


def test_a_band_per_dose(experiment: SimulationExperiment) -> None:
    _, ax = _axes(
        experiment,
        lambda p: p.band(
            Data("time", task="task_dose_draws"),
            Data("[C]", task="task_dose_draws"),
            across="draw",
            over="dose",
            name="C",
        ),
    )
    assert len(ax.collections) == 3


def test_a_bar_curve_over_a_dimension_raises(experiment: SimulationExperiment) -> None:
    with pytest.raises(ValueError, match="bar"):
        _axes(
            experiment,
            lambda p: p.curve(
                x=Data("time", task="task_doses"),
                y=Data("[C]", task="task_doses"),
                over="dose",
                type=CurveType.BAR,
            ),
        )
