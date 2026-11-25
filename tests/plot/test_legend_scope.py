"""One legend per row, `FigureSettings.legend_scope` (#327).

The case of the figure templates of pkdb_models: a grid of 2 x 3 plots, every
plot draws the simulation, the plots of the first column also data and the
plots of the second column a sum; the last plot of the second row is empty or
missing.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.plot import Axis, Figure, Plot
from sbmlsim.plot.resolve import figure_chart
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


SVG = "{http://www.w3.org/2000/svg}"


def _legends(experiment: SimulationExperiment, figure: Figure) -> list[str]:
    """The labels of the legend entries the renderer draws, in their order."""
    svg = figure_chart(experiment, figure).svg()
    return [
        text.text or ""
        for text in ET.fromstring(svg).iter(f"{SVG}text")
        if text.get("class") == "sbmlsim-legend-label"
    ]


@pytest.mark.parametrize("missing", [False, True])
def test_a_row_legend_has_the_entries_of_all_plots_of_its_row(
    experiment: SimulationExperiment, missing: bool
) -> None:
    """One legend per row with the entries of every plot of the row.

    The legend of the last plot of a row lacked the data and the sum of the
    other plots, and a row whose last plot is empty or missing had none.
    """
    figure = _grid(experiment, missing=missing)
    assert figure_chart(experiment, figure).spec()["legend"]["scope"] == "row"
    assert _legends(experiment, figure) == ENTRIES + ENTRIES


def test_an_entry_which_looks_otherwise_is_shown_again(
    experiment: SimulationExperiment,
) -> None:
    """Entries are one entry when they read and look the same, not when only the label is."""
    figure = _grid(experiment)
    plot = figure.subplots[0].plot
    _dose(plot, 1, label="simulation", color="tab:red")
    assert _legends(experiment, figure)[:4] == [
        "simulation",
        "data",
        "simulation",
        "sum",
    ]


def test_a_plot_without_a_legend_has_no_entries_in_the_row_legend(
    experiment: SimulationExperiment,
) -> None:
    """The `legend` of a plot decides whether its entries enter the legend of its row."""
    figure = _grid(experiment, hidden=((1, 1),))
    assert _legends(experiment, figure) == ["simulation", "sum", *ENTRIES]


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
    assert _legends(experiment, figure)[:6] == (
        ["simulation", "data", "sensitivity (5-95 %)", "study", "sum", "study"]
    )


def test_a_figure_without_a_scope_has_legends_per_plot(
    experiment: SimulationExperiment,
) -> None:
    """The default scope is a legend per plot, as before."""
    figure = _grid(experiment, scope=None)
    spec = figure_chart(experiment, figure).spec()
    assert spec["legend"]["scope"] == "plot"
    # the last plot of the second row is empty and has no legend
    assert _legends(experiment, figure) == [
        "simulation",
        "data",
        "simulation",
        "sum",
        "simulation",
        "simulation",
        "data",
        "simulation",
        "sum",
    ]
