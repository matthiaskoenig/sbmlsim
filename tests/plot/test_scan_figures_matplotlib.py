"""Matplotlib draws curves over the points of a scan and bands."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure as FigureMPL
from matplotlib.lines import Line2D

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.plot import Axis, Figure, Plot
from sbmlsim.plot.plotting import CurveType, YAxisPosition
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
    assert len(ax.collections) == 2  # the area and the empty one of the legend
    _low, _high, median = _data_lines(ax)
    y = Data("[C]", task="task_draws").get_data(experiment, to_units="mg/l")
    np.testing.assert_allclose(
        np.asarray(median.get_ydata()), np.nanmedian(y.values, axis=0)
    )
    assert sorted(_labels(ax)) == ["5-95 %", "C median"]


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
    assert len(ax.collections) == 4  # three areas and the one of the legend


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


def _figure(
    experiment: SimulationExperiment,
    draw: Callable[[Plot], object],
    right: bool = False,
    outside: bool = False,
) -> FigureMPL:
    figure = Figure(experiment=experiment, sid="fig", num_rows=1, num_cols=1)
    if outside:
        figure.legend_position = "outside"
    plot = figure.create_plots(
        xaxis=Axis("time", unit="hr"),
        yaxis=Axis("C", unit="mg/l"),
        legend=True,
    )[0]
    if right:
        plot.yaxis_right = Axis("C right", unit="mg/l")
    draw(plot)
    fig = MatplotlibFigureSerializer.to_figure(experiment, figure)
    FigureCanvasAgg(fig).draw()
    return fig


def test_a_band_per_dose_has_an_entry_per_point_and_one_for_the_range(
    experiment: SimulationExperiment,
) -> None:
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
    assert sorted(_labels(ax)) == [
        "5-95 %",
        "C, PODOSE = 100 mg",
        "C, PODOSE = 200 mg",
        "C, PODOSE = 50 mg",
    ]


def test_a_band_without_median_names_the_upper_line(
    experiment: SimulationExperiment,
) -> None:
    _, ax = _axes(
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
    assert sorted(_labels(ax)) == [
        "5-95 %",
        "C, PODOSE = 100 mg",
        "C, PODOSE = 200 mg",
        "C, PODOSE = 50 mg",
    ]
    named = [ln for ln in ax.get_lines() if not str(ln.get_label()).startswith("_")]
    assert len(named) == 3
    for line in named:
        peak = float(np.max(np.asarray(line.get_ydata(), dtype=float)))
        same = [
            o
            for o in ax.get_lines()
            if o.get_color() == line.get_color() and o is not line
        ]
        assert all(
            float(np.max(np.asarray(o.get_ydata(), dtype=float))) <= peak for o in same
        )


def test_a_band_on_the_right_axis(experiment: SimulationExperiment) -> None:
    fig = _figure(
        experiment,
        lambda p: p.band(
            Data("time", task="task_draws"),
            Data("[C]", task="task_draws"),
            across="draw",
            name="C",
            yaxis_position=YAxisPosition.RIGHT,
        ),
        right=True,
    )
    assert len(fig.axes) == 2 and len(fig.axes[1].lines) > 0


def test_the_colour_bar_leaves_the_right_axis_label_alone(
    experiment: SimulationExperiment,
) -> None:
    fig = _figure(
        experiment,
        lambda p: p.curve(
            x=Data("time", task="task_many"),
            y=Data("[C]", task="task_many"),
            over="dose",
            yaxis_position=YAxisPosition.RIGHT,
        ),
        right=True,
    )
    renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()
    ax1, ax2, bar = fig.axes
    assert ax2.yaxis.label.get_window_extent(renderer).x1 <= (
        bar.get_window_extent(renderer).x0
    )
    assert ax1.get_position().width == pytest.approx(ax2.get_position().width)


def test_an_outside_legend_is_right_of_the_colour_bar(
    experiment: SimulationExperiment,
) -> None:
    fig = _figure(
        experiment,
        lambda p: p.curve(
            x=Data("time", task="task_many2"),
            y=Data("[C]", task="task_many2"),
            over=("dose", "rate"),
        ),
        outside=True,
    )
    renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()
    ax, bar = fig.axes
    legend = ax.get_legend()
    assert legend is not None
    tight = bar.get_tightbbox(renderer)
    assert tight is not None
    assert legend.get_window_extent(renderer).x0 >= tight.x1


def test_the_legend_of_bands_covers_no_boundary(
    experiment: SimulationExperiment,
) -> None:
    fig = _figure(
        experiment,
        lambda p: p.band(
            Data("time", task="task_dose_draws"),
            Data("[C]", task="task_dose_draws"),
            across="draw",
            over="dose",
            name="C",
        ),
    )
    renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    box = legend.get_window_extent(renderer)
    for line in ax.get_lines():
        xy = ax.transData.transform(
            np.column_stack([line.get_xdata(), line.get_ydata()])
        )
        assert not any(box.contains(x, y) for x, y in xy)
