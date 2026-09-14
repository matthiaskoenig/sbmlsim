"""Tests of rendering the figure model with matplotlib."""

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot.plotting import (
    Axis,
    ColorType,
    Curve,
    Figure,
    Line,
    LineType,
    Plot,
    Style,
    SubPlot,
)
from sbmlsim.plot.serialization_matplotlib import (
    MatplotlibFigureSerializer,
)
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.simulator.simulation_serial import SimulatorSerial
from sbmlsim.task import Task


def _axes_of(plot: Plot) -> Axes:
    """Render a single plot and get the axes it was drawn on."""
    figure = Figure(
        experiment=None,
        sid="fig",
        num_rows=1,
        num_cols=1,
        subplots=[SubPlot(plot=plot, row=1, col=1)],
    )
    mpl_figure = MatplotlibFigureSerializer.to_figure(experiment=None, figure=figure)
    axes = mpl_figure.axes[0]
    plt.close(mpl_figure)
    return axes


# ---------------------------------------------------------------------------
# the spines of an axis which is not there
# ---------------------------------------------------------------------------
def test_a_plot_without_an_x_axis_hides_the_horizontal_spines() -> None:
    """The x axis is the bottom and the top spine, not the left and the right.

    The two were swapped, so a plot without an x axis kept the line of the
    axis it was hiding and lost the box of the one it was keeping.
    """
    plot = Plot(sid="p", yaxis=Axis("y", unit="mM"))
    plot.xaxis = None
    axes = _axes_of(plot)

    assert axes.spines["bottom"].get_visible() is False
    assert axes.spines["top"].get_visible() is False
    assert axes.spines["left"].get_visible() is True
    assert axes.spines["right"].get_visible() is True


def test_a_plot_without_a_y_axis_hides_the_vertical_spines() -> None:
    """And the y axis is the left and the right spine."""
    plot = Plot(sid="p", xaxis=Axis("x", unit="s"))
    plot.yaxis = None
    axes = _axes_of(plot)

    assert axes.spines["left"].get_visible() is False
    assert axes.spines["right"].get_visible() is False
    assert axes.spines["bottom"].get_visible() is True
    assert axes.spines["top"].get_visible() is True


# ---------------------------------------------------------------------------
# the reverse flag
# ---------------------------------------------------------------------------
reverse_data = [
    # only a lower bound, which used to leave the axis in its normal direction
    (0.0, None),
    (None, 10.0),
    (0.0, 10.0),
]


@pytest.mark.parametrize("min_, max_", reverse_data)
def test_a_reversed_axis_is_reversed(min_: float | None, max_: float | None) -> None:
    """`reverse` inverts the axis whichever of the bounds are given.

    The limits were swapped rather than the axis inverted, so a `reverse` with
    only one bound set assigned `None` and did nothing at all.
    """
    plot = Plot(
        sid="p",
        xaxis=Axis("x", unit="s", reverse=True, min=min_, max=max_),
        yaxis=Axis("y", unit="mM"),
    )
    left, right = _axes_of(plot).get_xlim()
    assert left > right


def test_an_axis_which_is_not_reversed_keeps_its_direction() -> None:
    """Without the flag the bounds are the limits, in order."""
    plot = Plot(
        sid="p",
        xaxis=Axis("x", unit="s", min=0.0, max=10.0),
        yaxis=Axis("y", unit="mM"),
    )
    assert _axes_of(plot).get_xlim() == (0.0, 10.0)


# ---------------------------------------------------------------------------
# settings of the plot which were not rendered
# ---------------------------------------------------------------------------
def test_the_panel_has_the_facecolor_of_its_plot() -> None:
    """`Plot.facecolor` was stored and never read by a serializer."""
    plot = Plot(
        sid="p",
        xaxis=Axis("x", unit="s"),
        yaxis=Axis("y", unit="mM"),
        facecolor=ColorType.parse_color("red"),
    )
    assert _axes_of(plot).get_facecolor() == (1.0, 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# matplotlib idioms
# ---------------------------------------------------------------------------
def test_hiding_the_ticks_hides_the_labels_and_keeps_the_marks() -> None:
    """`ticks_visible` hides the labels of whatever the locator produces.

    `set_xticklabels([])` is only correct with a fixed locator and leaves the
    tick marks drawn anyway, so `tick_params` is what this asks for.
    """
    plot = Plot(
        sid="p",
        xaxis=Axis("x", unit="s", ticks_visible=False),
        yaxis=Axis("y", unit="mM"),
    )
    axes = _axes_of(plot)
    assert all(not t.get_text() for t in axes.get_xticklabels())
    # the y axis is untouched
    assert any(t.get_text() for t in axes.get_yticklabels())


def test_a_spine_which_is_not_drawn_is_hidden() -> None:
    """A spine of `LineType.NONE` was painted in the colour of the figure.

    That only hides it while the panel has that colour, which `Plot.facecolor`
    decides; an invisible spine is invisible on any background.
    """
    plot = Plot(
        sid="p",
        xaxis=Axis("x", unit="s", style=Style(line=Line(type=LineType.NONE))),
        yaxis=Axis("y", unit="mM"),
    )
    axes = _axes_of(plot)
    assert axes.spines["bottom"].get_visible() is False
    assert axes.spines["top"].get_visible() is False
    assert axes.spines["left"].get_visible() is True


# ---------------------------------------------------------------------------
# a curve is drawn with the call which fits its data
# ---------------------------------------------------------------------------
class _CurveExperiment(SimulationExperiment):
    """An experiment of one timecourse, plotted once with and once without error."""

    def models(self) -> dict:
        return {"m": AbstractModel(source=REPRESSILATOR_SBML)}

    def simulations(self) -> dict:
        return {"s": TimecourseSim([Timecourse(start=0, end=20, steps=20)])}

    def tasks(self) -> dict:
        return {"t": Task(model="m", simulation="s")}

    def data(self) -> dict:
        return {"X": Data(index="[X]", task="t"), "time": Data(index="time", task="t")}

    def figures(self) -> dict:
        return {}


def _curve_plot(with_error: bool) -> Plot:
    """Get a plot of one curve, with or without error data."""
    plot = Plot(
        sid="p",
        xaxis=Axis("time", unit="second"),
        yaxis=Axis("X", unit="dimensionless"),
    )
    plot.curves.append(
        Curve(
            x=Data(index="time", task="t"),
            y=Data(index="[X]", task="t"),
            yerr=Data(index="[X]", task="t") if with_error else None,
            name="X",
        )
    )
    return plot


def _render_curve(with_error: bool) -> Axes:
    """Run the experiment and render a single curve."""
    runner = ExperimentRunner(
        experiment_classes=[_CurveExperiment],
        simulator=SimulatorSerial(),
        base_path=Path("."),
        data_path=Path("."),
    )
    experiment = runner.experiments["_CurveExperiment"]
    experiment.run(runner.simulator, show_figures=False)
    plot = _curve_plot(with_error=with_error)
    figure = Figure(
        experiment=experiment,
        sid="fig",
        num_rows=1,
        num_cols=1,
        subplots=[SubPlot(plot=plot, row=1, col=1)],
    )
    mpl_figure = MatplotlibFigureSerializer.to_figure(
        experiment=experiment, figure=figure
    )
    axes = mpl_figure.axes[0]
    plt.close(mpl_figure)
    return axes


def test_a_curve_without_error_data_is_drawn_as_a_line() -> None:
    """`errorbar` builds the containers of the bars whether or not there are any.

    It is twice the cost of `plot` for the same line, so a curve which has no
    error data is drawn with `plot`; it must keep its label and its style.
    """
    axes = _render_curve(with_error=False)
    assert len(axes.lines) == 1
    assert not axes.containers
    assert axes.get_legend_handles_labels()[1] == ["X"]
    assert axes.lines[0].get_linewidth() == 2.0


def test_a_curve_with_error_data_is_drawn_with_its_error_bars() -> None:
    """And one which has error data still goes through `errorbar`."""
    axes = _render_curve(with_error=True)
    assert len(axes.containers) == 1


# ---------------------------------------------------------------------------
# the figures are not held by the global registry of pyplot
# ---------------------------------------------------------------------------
def test_rendering_does_not_fill_the_pyplot_registry() -> None:
    """A figure of the serializer is not managed by `pyplot`.

    `plt.figure()` keeps every figure it creates until someone closes it, so
    rendering many of them leaks; matplotlib warns from the twentieth on. The
    figure is created directly, which nothing here needed the state machine
    for.
    """
    before = set(plt.get_fignums())
    figures = []
    for index in range(25):
        plot = Plot(
            sid=f"p{index}", xaxis=Axis("x", unit="s"), yaxis=Axis("y", unit="mM")
        )
        figures.append(
            MatplotlibFigureSerializer.to_figure(
                experiment=None,
                figure=Figure(
                    experiment=None,
                    sid=f"f{index}",
                    num_rows=1,
                    num_cols=1,
                    subplots=[SubPlot(plot=plot, row=1, col=1)],
                ),
            )
        )

    assert set(plt.get_fignums()) == before
    assert all(figure.canvas.manager is None for figure in figures)


def test_a_rendered_figure_can_still_be_saved(tmp_path: Path) -> None:
    """`savefig` is what a run does with a figure and needs no manager."""
    plot = Plot(sid="p", xaxis=Axis("x", unit="s"), yaxis=Axis("y", unit="mM"))
    figure = MatplotlibFigureSerializer.to_figure(
        experiment=None,
        figure=Figure(
            experiment=None,
            sid="f",
            num_rows=1,
            num_cols=1,
            subplots=[SubPlot(plot=plot, row=1, col=1)],
        ),
    )
    path = tmp_path / "figure.svg"
    figure.savefig(path, bbox_inches="tight")
    assert path.stat().st_size > 0
