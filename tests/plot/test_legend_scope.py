"""One legend per row, `FigureSettings.legend_scope` (#327).

The case of the figure templates of pkdb_models: a grid of 2 x 3 plots, every
plot draws the simulation, the plots of the first column also data and the
plots of the second column a sum; the last plot of the second row is empty or
missing.
"""

from pathlib import Path
from typing import Any

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure as FigureMPL
from matplotlib.legend import Legend

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.plot import Axis, Figure, Plot
from sbmlsim.plot.plotting import YAxisPosition
from sbmlsim.plot.serialization_matplotlib import (
    COLORBAR_GAP,
    MatplotlibFigureSerializer,
)
from sbmlsim.simulator import Simulator
from tests.plot.scan_experiment import ScanFigures

ENTRIES = ["simulation", "data", "sum"]


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


def _dose(plot: Plot, dose: int, **kwargs: Any) -> None:
    """A curve of one dose of the scan of doses."""
    plot.curve(
        x=Data("time", task="task_doses", sel={"dose": dose}),
        y=Data("[C]", task="task_doses", sel={"dose": dose}),
        **kwargs,
    )


def _grid(
    experiment: SimulationExperiment,
    scope: str | None = "row",
    missing: bool = False,
    hidden: tuple[tuple[int, int], ...] = (),
    settings: dict[str, Any] | None = None,
) -> Figure:
    """The grid of the case; `hidden` plots have no legend, the last cell is empty or missing."""
    overrides: dict[str, Any] = {"legend_position": "outside", **(settings or {})}
    if scope is not None:
        overrides["legend_scope"] = scope
    figure = Figure(
        experiment=experiment,
        sid="grid",
        name="One legend per row",
        num_rows=2,
        num_cols=3,
        settings=overrides,
    )
    for row in (1, 2):
        for col in (1, 2, 3):
            if (row, col) == (2, 3) and missing:
                continue
            plot = Plot(
                sid=f"p{row}{col}",
                name=f"panel {row} {col}",
                xaxis=Axis("time", unit="hr"),
                yaxis=Axis("C", unit="mg/l"),
            )
            plot.legend = (row, col) not in hidden
            figure.add_subplot(plot, row=row, col=col)
            if (row, col) == (2, 3):
                continue
            _dose(plot, 2, label="simulation", color="black")
            if col == 1:
                _dose(plot, 0, label="data", color="black", marker="s", linestyle="")
            if col == 2:
                _dose(plot, 1, label="sum", color="black", linestyle="--")
    return figure


def _render(experiment: SimulationExperiment, figure: Figure) -> FigureMPL:
    fig = MatplotlibFigureSerializer.to_figure(experiment, figure)
    FigureCanvasAgg(fig).draw()
    return fig


def _texts(legend: Legend) -> list[str]:
    return [text.get_text() for text in legend.get_texts()]


def _tight(ax: Any) -> Any:
    """The box of an axes with its ticks and labels."""
    box = ax.get_tightbbox()
    assert box is not None
    return box


def _panels(fig: FigureMPL) -> list[Any]:
    """The drawn axes of the plots, the axes of an empty cell left out."""
    return [ax for ax in fig.axes if ax.axison]


@pytest.mark.parametrize("missing", [False, True])
def test_a_row_legend_has_the_entries_of_all_plots_of_its_row(
    experiment: SimulationExperiment, missing: bool
) -> None:
    """One legend per row, right of the grid, with the entries of every plot of the row.

    The legend of the last plot of a row lacked the data and the sum of the
    other plots, and a row whose last plot is empty or missing had none.
    """
    fig = _render(experiment, _grid(experiment, missing=missing))
    assert [ax.get_legend() for ax in fig.axes] == [None] * len(fig.axes)
    first, second = sorted(fig.legends, key=lambda lg: -lg.get_window_extent().y1)
    assert _texts(first) == _texts(second) == ENTRIES
    panels = _panels(fig)
    right = max(ax.get_window_extent().x1 for ax in panels)
    for legend, row in [(first, panels[:3]), (second, panels[3:])]:
        extent = legend.get_window_extent()
        # right of the grid, at the top of its row
        assert extent.x0 == pytest.approx(right + COLORBAR_GAP * fig.dpi, abs=1.0)
        assert extent.y1 == pytest.approx(row[0].get_window_extent().y1, abs=1.0)


def test_a_row_legend_widens_the_last_column_only(
    experiment: SimulationExperiment,
) -> None:
    """The panels keep their places and sizes, the figure is wider by the legend."""
    plain = _render(
        experiment,
        _grid(
            experiment,
            scope=None,
            hidden=((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)),
        ),
    )
    fig = _render(experiment, _grid(experiment))
    legend = max(lg.get_window_extent().width for lg in fig.legends) / fig.dpi
    assert fig.get_figwidth() == pytest.approx(
        plain.get_figwidth() + COLORBAR_GAP + legend
    )
    for a, b in zip(_panels(plain), _panels(fig), strict=True):
        assert tuple(b.get_window_extent().bounds) == pytest.approx(
            tuple(a.get_window_extent().bounds), abs=0.5
        )


def test_an_entry_which_looks_otherwise_is_shown_again(
    experiment: SimulationExperiment,
) -> None:
    """Entries are one entry when they read and look the same, not when only the label is."""
    figure = _grid(experiment)
    plot = figure.subplots[0].plot
    _dose(plot, 1, label="simulation", color="tab:red")
    fig = _render(experiment, figure)
    first = max(fig.legends, key=lambda lg: lg.get_window_extent().y1)
    assert _texts(first) == ["simulation", "data", "simulation", "sum"]


def test_a_plot_without_a_legend_has_no_entries_in_the_row_legend(
    experiment: SimulationExperiment,
) -> None:
    """The `legend` of a plot decides whether its entries enter the legend of its row."""
    fig = _render(experiment, _grid(experiment, hidden=((1, 1),)))
    first, second = sorted(fig.legends, key=lambda lg: -lg.get_window_extent().y1)
    assert _texts(first) == ["simulation", "sum"]
    assert _texts(second) == ENTRIES


def test_bands_and_error_bars_are_merged_when_they_look_the_same(
    experiment: SimulationExperiment,
) -> None:
    """The bands and the data with error bars of the plots of a row are shown once.

    The band figures of acetaminophen in pkdb_models: every plot of a row draws
    the band of the sensitivity of a task and the data of a study with error
    bars; data with other error bars look otherwise and stay apart.
    """
    figure = _grid(experiment)
    for k, subplot in enumerate(figure.subplots[:3]):
        plot = subplot.plot
        plot.band(
            x=Data("time", task="task_draws"),
            y=Data("[C]", task="task_draws"),
            across="draw",
            name="sensitivity",
            color="tab:blue",
        )
        data = Data("[C]", task="task_doses", sel={"dose": 0})
        plot.curve(
            x=Data("time", task="task_doses", sel={"dose": 0}),
            y=data,
            yerr=data,
            label="study",
            color="tab:red" if k == 2 else "black",
            marker="o",
            linestyle="",
        )
    fig = _render(experiment, figure)
    first = max(fig.legends, key=lambda lg: lg.get_window_extent().y1)
    assert _texts(first) == [
        "simulation",
        "data",
        "sensitivity (5-95 %)",
        "study",
        "sum",
        "study",
    ]


def test_a_plot_which_spans_rows_has_the_legend_of_the_lower_row_right_of_it(
    experiment: SimulationExperiment,
) -> None:
    """A plot of the last column which spans both rows: both legends are right of it."""
    figure = Figure(
        experiment=experiment,
        sid="span",
        num_rows=2,
        num_cols=2,
        settings={"legend_scope": "row"},
    )
    plots = [
        figure.add_subplot(Plot(sid=sid), row=row, col=col, row_span=span)
        for sid, row, col, span in [("a", 1, 1, 1), ("b", 2, 1, 1), ("c", 1, 2, 2)]
    ]
    _dose(plots[0], 0, label="upper", color="black")
    _dose(plots[1], 1, label="lower", color="tab:red")
    _dose(plots[2], 2, label="span", color="tab:blue")
    fig = _render(experiment, figure)
    first, second = sorted(fig.legends, key=lambda lg: -lg.get_window_extent().y1)
    assert _texts(first) == ["upper", "span"]
    assert _texts(second) == ["lower"]
    a, b, c = _panels(fig)
    for legend, row in [(first, a), (second, b)]:
        extent = legend.get_window_extent()
        assert extent.x0 == pytest.approx(
            c.get_window_extent().x1 + COLORBAR_GAP * fig.dpi, abs=1.0
        )
        assert extent.y1 == pytest.approx(row.get_window_extent().y1, abs=1.0)


def test_a_tall_row_legend_gets_the_room_below_its_row(
    experiment: SimulationExperiment,
) -> None:
    """A row legend taller than its row overlaps no legend and no plot below it."""
    figure = _grid(experiment, settings={"panel_height": 2.0})
    for k in range(10):
        _dose(figure.subplots[0].plot, 1, label=f"task {k}", color=f"C{k}")
    fig = _render(experiment, figure)
    first, second = sorted(fig.legends, key=lambda lg: -lg.get_window_extent().y1)
    assert first.get_window_extent().height > _panels(fig)[0].get_window_extent().height
    assert first.get_window_extent().y0 >= second.get_window_extent().y1
    assert first.get_window_extent().y0 >= max(_tight(ax).y1 for ax in _panels(fig)[3:])


def test_the_row_legends_are_right_of_the_colour_bars_of_the_last_column(
    experiment: SimulationExperiment,
) -> None:
    """Every legend of a row starts right of the colour bar of a plot in the last column.

    The legends of the rows are one column right of the grid: the legend of
    the second row, whose last plot has no colour bar, is right of the colour
    bar of the first row too, and the last column is widened only once.
    """
    figure = _grid(experiment)
    plot = figure.subplots[2].plot
    plot.curve(
        x=Data("time", task="task_many"), y=Data("[C]", task="task_many"), over="dose"
    )
    fig = _render(experiment, figure)
    # the colour bar is the axes placed after the six panels
    bar = fig.axes[-1]
    assert bar.get_window_extent().width < 0.2 * fig.dpi
    first, second = sorted(fig.legends, key=lambda lg: -lg.get_window_extent().y1)
    x0 = first.get_window_extent().x0
    assert x0 >= _tight(bar).x1
    assert second.get_window_extent().x0 == pytest.approx(x0)
    # the figure ends at the widest legend, as cropped by `bbox_inches="tight"`
    legend = max(lg.get_window_extent().x1 for lg in fig.legends)
    assert fig.get_figwidth() * fig.dpi >= legend


def test_the_row_legends_are_right_of_a_right_y_axis_of_the_last_column(
    experiment: SimulationExperiment,
) -> None:
    """The legend of a row is right of the ticks of a right y axis in another row."""
    figure = _grid(experiment)
    plot = figure.subplots[5].plot
    plot.set_yaxis_right("C2", unit="mg/l")
    _dose(plot, 1, label="right", color="tab:green", yaxis_position=YAxisPosition.RIGHT)
    fig = _render(experiment, figure)
    first, second = sorted(fig.legends, key=lambda lg: -lg.get_window_extent().y1)
    right = max(_tight(ax).x1 for ax in fig.axes)
    assert first.get_window_extent().x0 >= right
    assert second.get_window_extent().x0 == pytest.approx(first.get_window_extent().x0)
    assert _texts(second) == [*ENTRIES, "right"]


def test_a_figure_without_a_scope_has_legends_per_plot(
    experiment: SimulationExperiment,
) -> None:
    """The default scope is a legend per plot, as before."""
    fig = _render(experiment, _grid(experiment, scope=None))
    assert fig.legends == []
    texts = [
        _texts(legend) for ax in fig.axes if (legend := ax.get_legend()) is not None
    ]
    # the last plot of the second row is empty and has no legend
    assert texts == [
        ["simulation", "data"],
        ["simulation", "sum"],
        ["simulation"],
        ["simulation", "data"],
        ["simulation", "sum"],
    ]


# ---------------------------------------------------------------------------
# plotly
# ---------------------------------------------------------------------------
def _plotly(experiment: SimulationExperiment, figure: Figure) -> Any:
    pytest.importorskip("plotly")
    from sbmlsim.plot.serialization_plotly import PlotlyFigureSerializer

    return PlotlyFigureSerializer.to_figure(experiment, figure)


def _shown(fig: Any) -> dict[str, list[str]]:
    """The entries of every legend of a plotly figure."""
    legends: dict[str, list[str]] = {}
    traces = sorted(fig.data, key=lambda t: t.legendrank or 0)
    for trace in traces:
        if trace.name and trace.showlegend is not False:
            legends.setdefault(trace.legend or "legend", []).append(trace.name)
    return legends


def test_plotly_has_a_legend_per_row(experiment: SimulationExperiment) -> None:
    """The legend of a row of plotly has the entries of its plots, at the top of the row."""
    fig = _plotly(experiment, _grid(experiment, missing=True, hidden=((2, 1),)))
    assert _shown(fig) == {"legend": ENTRIES, "legend2": ["simulation", "sum"]}
    for row, legend in [(1, fig.layout.legend), (2, fig.layout.legend2)]:
        low, high = fig.get_subplot(row, 1).yaxis.domain
        assert (legend.y, legend.yanchor, legend.xanchor) == (high, "top", "left")
        assert legend.maxheight == pytest.approx(high - low)


def test_plotly_without_a_scope_is_unchanged(experiment: SimulationExperiment) -> None:
    """The one legend of the figure of plotly shows every plot, its `legend` aside."""
    fig = _plotly(experiment, _grid(experiment, scope=None, hidden=((1, 1),)))
    assert _shown(fig) == {"legend": ENTRIES}
    assert {t.legend for t in fig.data} == {None}
