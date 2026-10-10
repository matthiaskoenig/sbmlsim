"""Matplotlib draws curves over the points of a scan and bands."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import QuadMesh
from matplotlib.colors import LogNorm, to_hex
from matplotlib.figure import Figure as FigureMPL
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.plot import Axis, Figure, Plot
from sbmlsim.plot.plotting import CurveType, YAxisPosition
from sbmlsim.plot.points import point_colormap
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
    cmap = point_colormap(None)
    # the colour map at the values 50, 100 and 200 mg
    assert [to_hex(line.get_color()) for line in lines] == [
        to_hex(cmap(f)) for f in (0.0, 1 / 3, 1.0)
    ]
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
    shades = point_colormap("tab:red")
    assert colors == [to_hex(shades(f)) for f in (0.0, 1 / 3, 1.0)]


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
    assert _labels(ax) == ["C"]  # the name of the curve, not one entry per point
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


def _renderer(fig: FigureMPL) -> Any:
    return cast(FigureCanvasAgg, fig.canvas).get_renderer()


def _tight(fig: FigureMPL, ax: Axes) -> Bbox:
    """The box of an axes with its ticks and labels, in pixels."""
    box = ax.get_tightbbox(_renderer(fig))
    assert box is not None
    return box


def _bar_colors(bar: Axes, values: Any) -> list[str]:
    """The colours a colour bar draws at values."""
    mesh = next(c for c in bar.collections if isinstance(c, QuadMesh))
    # a fraction of the colour map, or the index of a colour of a discrete norm
    mapped = np.asarray(mesh.norm(np.asarray(values, dtype=float)))
    return [to_hex(mesh.cmap(f.item())) for f in mapped]


def _colors(ax: Axes) -> list[str]:
    return [to_hex(line.get_color()) for line in _data_lines(ax)]


def test_selected_points_are_labelled_by_their_labels(
    experiment: SimulationExperiment,
) -> None:
    _, ax = _axes(
        experiment, lambda p: _curve(p, "task_doses", "dose", sel={"dose": [2, 1]})
    )
    assert _labels(ax) == ["C, PODOSE = 200 mg", "C, PODOSE = 100 mg"]
    high, low = _data_lines(ax)
    peaks = [
        np.nanmax(np.asarray(line.get_ydata(), dtype=float)) for line in (high, low)
    ]
    assert peaks[0] > peaks[1]
    cmap = point_colormap(None)
    assert _colors(ax) == [to_hex(cmap(1.0)), to_hex(cmap(0.0))]


def test_a_selected_second_dimension_draws_with_its_own_labels(
    experiment: SimulationExperiment,
) -> None:
    _, ax = _axes(
        experiment,
        lambda p: _curve(p, "task_grid5", ("dose", "rate"), sel={"rate": [3, 1]}),
    )
    assert len(_data_lines(ax)) == 6
    unit = experiment.model_units("task_grid5").get("ke", "")
    assert _labels(ax)[3:] == [f"ke = 0.4 {unit}".rstrip(), f"ke = 0.2 {unit}".rstrip()]


def test_geometric_doses_get_a_logarithmic_colour_bar(
    experiment: SimulationExperiment,
) -> None:
    fig, ax = _axes(experiment, lambda p: _curve(p, "task_geom", "dose"))
    bar = fig.axes[1]
    assert bar.get_yscale() == "log"
    assert bar.get_ylim() == pytest.approx((1.0, 1000.0))
    values = np.geomspace(1, 1000, 12)
    norm, cmap = LogNorm(1.0, 1000.0), point_colormap(None)
    assert _colors(ax) == [to_hex(cmap(float(norm(v)))) for v in values]
    assert _colors(ax) == _bar_colors(bar, values)


def test_decreasing_doses_keep_line_and_bar_consistent(
    experiment: SimulationExperiment,
) -> None:
    fig, ax = _axes(experiment, lambda p: _curve(p, "task_decreasing", "dose"))
    bar = fig.axes[1]
    assert bar.get_yscale() == "linear"
    assert bar.get_ylim() == pytest.approx((10.0, 120.0))
    assert _colors(ax) == _bar_colors(bar, np.linspace(120, 10, 12))
    assert _colors(ax)[0] == to_hex(point_colormap(None)(1.0))


def test_a_dimension_of_labels_shows_its_labels_on_the_colour_bar(
    experiment: SimulationExperiment,
) -> None:
    fig, ax = _axes(experiment, lambda p: _curve(p, "task_cond", "cond"))
    FigureCanvasAgg(fig).draw()
    bar = fig.axes[1]
    assert bar.get_ylabel() == "cond"
    assert [t.get_text() for t in bar.get_yticklabels()] == [
        str(10 * k) for k in range(12)
    ]
    assert _colors(ax) == _bar_colors(bar, range(12))


def test_a_named_curve_with_a_colour_bar_keeps_its_legend_entry(
    experiment: SimulationExperiment,
) -> None:
    _, ax = _axes(experiment, lambda p: _curve(p, "task_many", "dose"))
    legend = ax.get_legend()
    assert legend is not None
    (handle,) = legend.legend_handles
    assert isinstance(handle, Line2D)
    assert to_hex(handle.get_color()) == to_hex(point_colormap(None)(0.5))


def test_a_named_band_with_a_colour_bar_keeps_its_legend_entry(
    experiment: SimulationExperiment,
) -> None:
    fig, ax = _axes(
        experiment,
        lambda p: p.band(
            Data("time", task="task_many_draws"),
            Data("[C]", task="task_many_draws"),
            across="draw",
            over="dose",
            name="C",
        ),
    )
    assert sorted(_labels(ax)) == ["5-95 %", "C"]
    assert fig.axes[1].get_ylabel() == "PODOSE [mg]"


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
    fig = _figure(experiment, _two_bars)
    ax, first, second = fig.axes
    assert _labels(ax) == ["C", "C geom"]
    assert ax.get_window_extent(_renderer(fig)).x1 < first.get_window_extent().x0
    assert _tight(fig, first).x1 <= second.get_window_extent().x0
    assert _tight(fig, second).x1 <= fig.bbox.x1
    assert first.get_ylabel() == second.get_ylabel() == "PODOSE [mg]"


def test_colour_bars_are_right_of_the_title_of_a_right_axis(
    experiment: SimulationExperiment,
) -> None:
    fig = _figure(experiment, lambda p: _two_bars(p, right=True), right=True)
    ax1, ax2, first, second = fig.axes
    renderer = _renderer(fig)
    axis = ax2.yaxis.get_tightbbox(renderer)
    assert axis is not None and axis.x1 <= first.get_window_extent().x0
    assert _tight(fig, first).x1 <= second.get_window_extent().x0
    assert ax1.get_position().width == pytest.approx(ax2.get_position().width)


def test_an_outside_legend_is_right_of_the_last_colour_bar(
    experiment: SimulationExperiment,
) -> None:
    fig = _figure(experiment, _two_bars, outside=True)
    ax, _first, second = fig.axes
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_window_extent(_renderer(fig)).x0 >= _tight(fig, second).x1


def test_the_colour_bars_of_a_panel_leave_the_next_panel_alone(
    experiment: SimulationExperiment,
) -> None:
    figure = Figure(experiment=experiment, sid="fig", num_rows=1, num_cols=2)
    plots = figure.create_plots(
        xaxis=Axis("time", unit="hr"), yaxis=Axis("C", unit="mg/l"), legend=True
    )
    plots[0].yaxis_right = Axis("C right", unit="mg/l")
    _two_bars(plots[0], right=True)
    _curve(plots[1], "task_doses", "dose")
    fig = MatplotlibFigureSerializer.to_figure(experiment, figure)
    FigureCanvasAgg(fig).draw()
    # the colour bars are placed after every panel was drawn
    _ax1, _ax2, other, first, second = fig.axes
    assert _tight(fig, first).x1 <= second.get_window_extent().x0
    assert _tight(fig, second).x1 <= _tight(fig, other).x0


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
    _, ax = _axes(experiment, _two_bands)
    medians = {
        str(line.get_label()): to_hex(line.get_color())
        for line in ax.get_lines()
        if str(line.get_label()).endswith("median")
    }
    assert medians == {"a median": to_hex("C0"), "b median": to_hex("C1")}
    assert _labels(ax) == ["a median", "5-95 %", "b median", "25-75 %"]


def test_the_range_entry_appears_once_per_range(
    experiment: SimulationExperiment,
) -> None:
    _, ax = _axes(experiment, lambda p: _two_bands(p, quantiles=(0.05, 0.95)))
    assert _labels(ax) == ["a median", "5-95 %", "b median"]


def test_colour_bars_widen_the_figure_and_keep_the_panels(
    experiment: SimulationExperiment,
) -> None:
    figure = Figure(experiment=experiment, sid="fig", num_rows=1, num_cols=2)
    plots = figure.create_plots(
        xaxis=Axis("time", unit="hr"), yaxis=Axis("C", unit="mg/l"), legend=True
    )
    _two_bars(plots[0])
    _curve(plots[1], "task_doses", "dose")
    fig = MatplotlibFigureSerializer.to_figure(experiment, figure)
    left, right, *_bars = fig.axes
    assert fig.get_figwidth() > figure.width
    assert left.get_position().width == pytest.approx(right.get_position().width)


def test_a_panel_with_a_right_axis_has_one_legend_off_the_data_of_both_axes(
    experiment: SimulationExperiment,
) -> None:
    def draw(plot: Plot) -> None:
        _curve(plot, "task_doses", "dose")
        _curve(
            plot,
            "task_doses",
            (),
            sel={"dose": 2},
            label="C right",
            yaxis_position=YAxisPosition.RIGHT,
        )

    fig = _figure(experiment, draw, right=True)
    ax1, ax2 = fig.axes
    assert ax1.get_legend() is None
    legend = ax2.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == [
        "C, PODOSE = 50 mg",
        "C, PODOSE = 100 mg",
        "C, PODOSE = 200 mg",
        "C right",
    ]
    box = legend.get_window_extent(_renderer(fig))
    for ax in (ax1, ax2):
        for line in ax.get_lines():
            if line.get_visible():
                xy = ax.transData.transform(
                    np.column_stack([line.get_xdata(), line.get_ydata()])
                )
                assert not any(box.contains(x, y) for x, y in xy)


def test_the_legend_entries_of_two_dimensions_carry_the_markers(
    experiment: SimulationExperiment,
) -> None:
    _, ax = _axes(
        experiment, lambda p: _curve(p, "task_grid2", ("dose", "rate"), marker="o")
    )
    legend = ax.get_legend()
    assert legend is not None
    handles = [h for h in legend.legend_handles if isinstance(h, Line2D)]
    assert len(handles) == 5
    assert all(h.get_marker() == "o" for h in handles)
