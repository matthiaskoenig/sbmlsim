"""Matplotlib draws curves over the points of a scan and bands."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import QuadMesh
from matplotlib.colors import LogNorm, to_hex, to_rgba
from matplotlib.figure import Figure as FigureMPL
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.plot import Axis, Figure, FigureSettings, Plot
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
        on_error="raise",
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


def _rgb(color: Any) -> str:
    """The colour of an area, an RGBA of one or more rows, as hex without alpha."""
    r, g, b, _ = np.asarray(color, dtype=float).reshape(-1, 4)[0]
    return to_hex((float(r), float(g), float(b)))


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
    assert len(ax.collections) == 1  # the area, which is the legend entry
    _low, _high, median = _data_lines(ax)
    y = Data("[C]", task="task_draws").get_data(experiment, to_units="mg/l")
    np.testing.assert_allclose(
        np.asarray(median.get_ydata()), np.nanmedian(y.values, axis=0)
    )
    assert _labels(ax) == ["C (5-95 %)"]


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
    assert len(ax.collections) == 3  # three areas


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
    figure = Figure(
        experiment=experiment,
        sid="fig",
        num_rows=1,
        num_cols=1,
        settings={"legend_position": "outside"} if outside else None,
    )
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


def test_a_band_per_dose_has_an_entry_per_point(
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
    assert _labels(ax) == [
        "C, PODOSE = 50 mg (5-95 %)",
        "C, PODOSE = 100 mg (5-95 %)",
        "C, PODOSE = 200 mg (5-95 %)",
    ]


def test_a_band_without_median_names_its_area(
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
    assert _labels(ax) == [
        "C, PODOSE = 50 mg (5-95 %)",
        "C, PODOSE = 100 mg (5-95 %)",
        "C, PODOSE = 200 mg (5-95 %)",
    ]
    assert all(str(line.get_label()).startswith("_") for line in ax.get_lines())
    legend = ax.get_legend()
    assert legend is not None
    areas = [_rgb(c.get_facecolor()) for c in ax.collections]
    handles = [h for h in legend.legend_handles if isinstance(h, Patch)]
    assert [_rgb(h.get_facecolor()) for h in handles] == areas


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
    assert _labels(ax) == ["C (5-95 %)"]
    legend = ax.get_legend()
    assert legend is not None
    (handle,) = legend.legend_handles
    assert isinstance(handle, Patch)
    assert to_hex(handle.get_facecolor()) == to_hex(point_colormap(None)(0.5))
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
    areas = {
        str(area.get_label()): _rgb(area.get_facecolor()) for area in ax.collections
    }
    assert areas == {"a (5-95 %)": to_hex("C0"), "b (25-75 %)": to_hex("C1")}
    medians = [
        to_hex(ln.get_color()) for ln in ax.get_lines() if ln.get_linewidth() == 2
    ]
    assert medians == [to_hex("C0"), to_hex("C1")]
    assert _labels(ax) == ["a (5-95 %)", "b (25-75 %)"]


def test_two_bands_of_one_range_have_an_entry_each(
    experiment: SimulationExperiment,
) -> None:
    _, ax = _axes(experiment, lambda p: _two_bands(p, quantiles=(0.05, 0.95)))
    assert _labels(ax) == ["a (5-95 %)", "b (5-95 %)"]


def _bands_of_two_colours(plot: Plot, median: bool) -> None:
    """A black band of 0-100 %, a blue one of 5-95 % and a black curve (#302)."""
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
    _curve(plot, "task_doses", (), sel={"dose": 0}, label="sim", color="black")


@pytest.mark.parametrize("median", [True, False], ids=["median", "no median"])
def test_the_legend_has_one_entry_per_band_with_its_area_in_its_colour(
    experiment: SimulationExperiment, median: bool
) -> None:
    """A band has one legend entry, its area in its colour, with its quantiles (#302)."""
    _, ax = _axes(experiment, lambda p: _bands_of_two_colours(p, median))
    legend = ax.get_legend()
    assert legend is not None
    assert _labels(ax) == ["a (0-100 %)", "b (5-95 %)", "sim"]
    a, b, _ = legend.legend_handles
    assert isinstance(a, Patch) and isinstance(b, Patch)
    assert to_hex(a.get_facecolor(), keep_alpha=True) == to_hex(
        to_rgba("black", 0.3), keep_alpha=True
    )
    assert to_hex(b.get_facecolor(), keep_alpha=True) == to_hex(
        to_rgba("blue", 0.3), keep_alpha=True
    )


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
    assert fig.get_figwidth() > 2 * FigureSettings().panel_width
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


def test_the_combined_legend_covers_no_bar_of_the_left_axis(
    experiment: SimulationExperiment,
) -> None:
    def draw(plot: Plot) -> None:
        plot.curve(
            x=Data("time", task="task_doses", sel={"dose": 2}),
            y=Data("[C]", task="task_doses", sel={"dose": 2}),
            label="bars",
            type=CurveType.BAR,
        )
        _curve(
            plot,
            "task_doses",
            (),
            sel={"dose": 1},
            label="C right",
            yaxis_position=YAxisPosition.RIGHT,
        )

    fig = _figure(experiment, draw, right=True)
    ax1, ax2 = fig.axes
    legend = ax2.get_legend()
    assert legend is not None
    assert ax1.patches
    box = legend.get_window_extent(_renderer(fig))
    for bar in ax1.patches:
        assert not box.overlaps(bar.get_window_extent())
    # the legend is placed off invisible copies of the bars on its own axes
    assert len([p for p in ax2.patches if not p.get_visible()]) == len(ax1.patches)


def _grid(
    experiment: SimulationExperiment,
    legends: set[tuple[int, int]],
    outside: bool = True,
) -> FigureMPL:
    """A figure of two rows and three columns, the legend on the given panels."""
    figure = Figure(
        experiment=experiment,
        sid="grid",
        num_rows=2,
        num_cols=3,
        settings={"legend_position": "outside"} if outside else None,
    )
    plots = figure.create_plots(
        xaxis=Axis("time", unit="hr"), yaxis=Axis("C", unit="mg/l"), legend=True
    )
    for k, plot in enumerate(plots):
        plot.legend = divmod(k, 3) in legends
        _curve(plot, "task_doses", "dose")
    fig = MatplotlibFigureSerializer.to_figure(experiment, figure)
    FigureCanvasAgg(fig).draw()
    return fig


def _panels(fig: FigureMPL) -> list[tuple[float, float]]:
    """The left and the right edge of the panels in inches, row by row."""
    width = fig.get_figwidth()
    return [
        (ax.get_position().x0 * width, ax.get_position().x1 * width) for ax in fig.axes
    ]


def _room(fig: FigureMPL, ax: Axes) -> float:
    """The room an outside legend of a panel takes, the gap and the legend, in inches."""
    legend = ax.get_legend()
    assert legend is not None
    return 0.12 + legend.get_window_extent(_renderer(fig)).width / fig.dpi


def test_an_outside_legend_widens_only_the_gap_right_of_its_column(
    experiment: SimulationExperiment,
) -> None:
    """One legend outside widens the gap right of its column and no other (ruling 67).

    `legend_position="outside"` added the width of a panel to every gap of the
    grid, so a figure of many panels became very wide (the band figures of the
    acetaminophen model of pkdb_models: 72 inches). The panels keep their size.
    """
    plain = _grid(experiment, set(), outside=False)
    fig = _grid(experiment, {(0, 0)})
    room = _room(fig, fig.axes[0])
    assert fig.get_figwidth() == pytest.approx(plain.get_figwidth() + room)
    before, after = _panels(plain), _panels(fig)
    for (a0, a1), (b0, b1) in zip(before, after, strict=True):
        assert b1 - b0 == pytest.approx(a1 - a0)
    # the panels of the first column stay, the others move by the room
    for k, ((a0, _), (b0, _)) in enumerate(zip(before, after, strict=True)):
        assert b0 == pytest.approx(a0 + (room if k % 3 else 0.0))
    legend = fig.axes[0].get_legend()
    assert legend is not None
    extent = legend.get_window_extent(_renderer(fig))
    panel = fig.axes[0].get_window_extent()
    assert extent.x0 == pytest.approx(panel.x1 + 0.12 * fig.dpi, abs=1.0)
    assert extent.y1 == pytest.approx(panel.y1, abs=1.0)
    assert extent.x1 <= _tight(fig, fig.axes[1]).x0


def test_one_legend_per_row_widens_the_figure_by_its_legend(
    experiment: SimulationExperiment,
) -> None:
    """A legend on the last plot of every row is right of the grid, inside the figure."""
    plain = _grid(experiment, set(), outside=False)
    fig = _grid(experiment, {(0, 2), (1, 2)})
    room = max(_room(fig, fig.axes[2]), _room(fig, fig.axes[5]))
    assert fig.get_figwidth() == pytest.approx(plain.get_figwidth() + room)
    for (a0, a1), (b0, b1) in zip(_panels(plain), _panels(fig), strict=True):
        assert (b0, b1) == pytest.approx((a0, a1))
    for ax in (fig.axes[2], fig.axes[5]):
        legend = ax.get_legend()
        assert legend is not None
        extent = legend.get_window_extent(_renderer(fig))
        assert extent.x0 >= ax.get_window_extent().x1
        assert extent.x1 <= fig.bbox.x1
    assert [ax.get_legend() is None for ax in fig.axes] == [
        True,
        True,
        False,
        True,
        True,
        False,
    ]


def test_an_outside_legend_is_right_of_the_right_axis(
    experiment: SimulationExperiment,
) -> None:
    """The legend of a panel with a right y axis and no colour bar leaves the axis alone."""
    fig = _figure(
        experiment,
        lambda p: _curve(p, "task_doses", "dose", yaxis_position=YAxisPosition.RIGHT),
        right=True,
        outside=True,
    )
    ax1, ax2 = fig.axes
    legend = ax1.get_legend()
    assert legend is not None
    axis = ax2.yaxis.get_tightbbox(_renderer(fig))
    assert axis is not None
    extent = legend.get_window_extent(_renderer(fig))
    assert extent.x0 == pytest.approx(axis.x1 + 0.12 * fig.dpi, abs=1.0)
    assert extent.x1 <= fig.bbox.x1
