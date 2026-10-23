"""Tests of rendering the figure model with matplotlib."""

from pathlib import Path
from typing import Any

import matplotlib
import pytest
from matplotlib.figure import Figure as FigureMPL

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text

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
    limit_log_axes,
)
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
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
        return {"s": Simulation(end=20, steps=20)}

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
    return _render(_curve_plot(with_error=with_error))


def _render(plot: Plot) -> Axes:
    """Run the experiment and render a plot of its data."""
    runner = ExperimentRunner(
        experiment_classes=[_CurveExperiment],
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
        on_error="raise",
    )
    experiment = runner.experiments["_CurveExperiment"]
    experiment.run(runner.simulator, show_figures=False)
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


@pytest.mark.parametrize("axis", ["x", "y"])
def test_a_bound_of_a_log_axis_keeps_the_other_limit_of_the_log_scale(
    axis: str,
) -> None:
    """The scale of an axis is set before its bounds.

    A bound was set while the axis was still linear, which fixed the other
    limit at the linear autoscale limit, no margin on a log axis: the log
    panel of the demo cut the peak of a curve at its top.
    """

    def plot_of(bound: float | None) -> Plot:
        plot = _curve_plot(with_error=False)
        unit = "second" if axis == "x" else "dimensionless"
        log = Axis("v", unit=unit, scale="log", min=bound)
        if axis == "x":
            plot.xaxis = log
        else:
            plot.yaxis = log
        return plot

    def limits(axes: Axes) -> tuple[float, float]:
        return axes.get_xlim() if axis == "x" else axes.get_ylim()

    free = limits(_render(plot_of(None)))
    bounded = limits(_render(plot_of(1e-3)))
    assert bounded[0] == 1e-3
    assert bounded[1] == pytest.approx(free[1])


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


# ---------------------------------------------------------------------------
# the title of the figure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("height", [3.0, 5.0, 10.0, 20.0])
def test_the_title_of_the_figure_is_above_the_titles_of_the_plots(
    height: float,
) -> None:
    """The title of a figure does not overlap the titles of its plots."""
    figure = Figure(
        experiment=None,
        sid="fig",
        name="Title of the figure",
        num_rows=1,
        num_cols=1,
        width=5,
        height=height,
        subplots=[
            SubPlot(plot=Plot(sid="plot", name="Title of the plot"), row=1, col=1)
        ],
    )
    mpl_figure = MatplotlibFigureSerializer.to_figure(experiment=None, figure=figure)
    renderer = FigureCanvasAgg(mpl_figure).get_renderer()
    title = next(
        text
        for text in mpl_figure.findobj(Text)
        if text.get_text() == mpl_figure.get_suptitle()
    ).get_window_extent(renderer)
    plot_title = mpl_figure.axes[0].title.get_window_extent(renderer)
    plt.close(mpl_figure)
    assert plot_title.y1 <= title.y0


def test_the_gap_below_the_title_of_the_figure_does_not_grow_with_its_height() -> None:
    """The plots of a tall figure begin right below its title (#312).

    The top of the plots was capped at the fraction of matplotlib (0.88), so
    the gap between the titles grew with the height: 3.6 in above the plots
    of a figure of six rows.
    """

    def gap(rows: int) -> float:
        figure = Figure(
            experiment=None,
            sid="fig",
            name="Title of the figure",
            num_rows=rows,
            num_cols=1,
            subplots=[
                SubPlot(plot=Plot(sid=f"p{k}", name="Title of the plot"), row=k, col=1)
                for k in range(1, rows + 1)
            ],
        )
        mpl_figure = MatplotlibFigureSerializer.to_figure(
            experiment=None, figure=figure
        )
        renderer = FigureCanvasAgg(mpl_figure).get_renderer()
        title = next(
            text
            for text in mpl_figure.findobj(Text)
            if text.get_text() == mpl_figure.get_suptitle()
        ).get_window_extent(renderer)
        plot_title = mpl_figure.axes[0].title.get_window_extent(renderer)
        assert plot_title.y1 <= title.y0
        return (title.y0 - plot_title.y1) / mpl_figure.dpi

    gaps = [gap(rows) for rows in (1, 2, 6)]
    assert max(gaps) - min(gaps) < 0.02, gaps
    assert max(gaps) < 0.3, gaps


def _margins(figure: Figure) -> tuple[float, float, float, float]:
    """Get the margins of the panels of a rendered figure, left, right, bottom and top, in inches."""
    mpl_figure = MatplotlibFigureSerializer.to_figure(experiment=None, figure=figure)
    width, height = mpl_figure.get_size_inches()
    boxes = [ax.get_position() for ax in mpl_figure.axes]
    return (
        min(box.x0 for box in boxes) * width,
        (1.0 - max(box.x1 for box in boxes)) * width,
        min(box.y0 for box in boxes) * height,
        (1.0 - max(box.y1 for box in boxes)) * height,
    )


def _grid(rows: int, cols: int, **kwargs: Any) -> Figure:
    """Get a figure with a plot in every cell of its grid."""
    return Figure(
        experiment=None,
        sid="fig",
        num_rows=rows,
        num_cols=cols,
        subplots=[
            SubPlot(plot=Plot(sid=f"p{row}_{col}"), row=row, col=col)
            for row in range(1, rows + 1)
            for col in range(1, cols + 1)
        ],
        **kwargs,
    )


def test_the_margins_of_a_figure_do_not_grow_with_its_number_of_panels() -> None:
    """A figure has the margins of a figure of one panel (#323).

    The margins were the fractions of matplotlib of the size of the figure,
    while the panels have a size in inches: a figure of six rows had 3.3 in
    below its last row, one of six columns 5.25 in left of its first.
    """
    single = _margins(_grid(1, 1))
    for rows, cols in [(6, 1), (1, 6), (3, 3)]:
        assert _margins(_grid(rows, cols)) == pytest.approx(single), (rows, cols)
    # the margins of a figure of an explicit size are the ones of its panels
    assert _margins(_grid(2, 3, width=21.0, height=10.0)) == pytest.approx(single)
    assert _margins(_grid(1, 1, width=3.5, height=2.5)) == pytest.approx(
        [margin / 2 for margin in single]
    )


def test_the_fixture_no_pyplot_fails_on_a_resolution_of_the_backend(
    no_pyplot: None,
) -> None:
    """Reading the backend from `rcParams` resolves it, which `no_pyplot` reports (#317)."""
    with pytest.raises(AssertionError, match="backend of pyplot was resolved"):
        _ = {**matplotlib.rcParams}
    with pytest.raises(AssertionError, match="created with pyplot"):
        plt.subplots()


def test_an_automatic_log_axis_of_a_custom_figure_is_limited() -> None:
    """The tiny first steps of the integrator do not stretch a log axis."""
    figure = FigureMPL()
    free, fixed = figure.subplots(1, 2)
    for ax in (free, fixed):
        ax.plot([0.0, 1e-9, 1.0, 2.0], [0.0, 1e-40, 10.0, 1.0])
        ax.set_yscale("log")
    fixed.set_ylim(1e-50, 100.0)
    limit_log_axes(figure)
    # six decades below the largest value, with the margin of the decades
    assert 1e-6 < free.get_ylim()[0] < 1e-5
    assert fixed.get_ylim() == (1e-50, 100.0)


def _labelled(
    rows: int = 1,
    cols: int = 1,
    ylabel: str = "midazolam",
    right: bool = False,
    **kwargs: Any,
) -> Figure:
    """Get a figure of plots with labelled axes in every cell of its grid."""
    return Figure(
        experiment=None,
        sid="fig",
        num_rows=rows,
        num_cols=cols,
        subplots=[
            SubPlot(
                plot=Plot(
                    sid=f"p{row}_{col}",
                    name=f"plot {row} {col}",
                    xaxis=Axis("time", unit="hr"),
                    yaxis=Axis(ylabel, unit="ng/ml"),
                    yaxis_right=Axis("metabolite", unit="ng/ml") if right else None,
                ),
                row=row,
                col=col,
            )
            for row in range(1, rows + 1)
            for col in range(1, cols + 1)
        ],
        **kwargs,
    )


@pytest.mark.parametrize(
    "figure",
    [
        _labelled(),
        _labelled(2, 3),
        _labelled(ylabel="the concentration of midazolam in the venous plasma"),
        _labelled(right=True),
        _labelled(2, 2, right=True, name="A title"),
        _labelled(width=3.5, height=2.5),
    ],
    ids=["one", "grid", "long label", "right axis", "title", "small"],
)
def test_the_labels_are_inside_a_figure_saved_without_a_tight_box(
    figure: Figure,
) -> None:
    """The margins fit the ticks and the labels of the panels at the edges (#362).

    The margins of matplotlib are sized for its fonts of 10 pt: the label of
    the x axis of the fonts of sbmlsim ended 0.066 in below the figure, which
    a figure saved without `bbox_inches="tight"` cut off.
    """
    mpl_figure = MatplotlibFigureSerializer.to_figure(experiment=None, figure=figure)
    renderer = FigureCanvasAgg(mpl_figure).get_renderer()
    width, height = mpl_figure.get_size_inches()
    dpi = mpl_figure.dpi
    axes = [
        box for ax in mpl_figure.axes if (box := ax.get_tightbbox(renderer)) is not None
    ]
    boxes = list(axes)
    texts = [
        text
        for text in mpl_figure.findobj(Text)
        if mpl_figure.get_suptitle() and text.get_text() == mpl_figure.get_suptitle()
    ]
    boxes += [text.get_window_extent(renderer) for text in texts]
    assert min(box.x0 for box in boxes) / dpi >= 0.05
    assert min(box.y0 for box in boxes) / dpi >= 0.05
    assert max(box.x1 for box in boxes) / dpi <= width - 0.05
    assert max(box.y1 for box in boxes) / dpi <= height - 0.05
    if texts:
        # the plots stay below the title
        title = texts[0].get_window_extent(renderer)
        assert max(box.y1 for box in axes) <= title.y0


def test_a_figure_whose_labels_fit_keeps_its_size() -> None:
    figure = _grid(2, 3)
    assert tuple(
        MatplotlibFigureSerializer.to_figure(
            experiment=None, figure=figure
        ).get_size_inches()
    ) == pytest.approx(figure.size(figure.resolved_settings(None)))


def test_a_label_longer_than_its_axis_is_wrapped() -> None:
    """A label of the y axis longer than the panel reached far above and below it."""
    long = "the concentration of midazolam in the venous plasma"
    figure = _labelled(ylabel=long)
    mpl_figure = MatplotlibFigureSerializer.to_figure(experiment=None, figure=figure)
    renderer = FigureCanvasAgg(mpl_figure).get_renderer()
    ax = mpl_figure.axes[0]
    label = ax.yaxis.label
    assert "\n" in label.get_text()
    assert label.get_text().replace("\n", " ") == f"{long} [ng/ml]"
    assert (
        label.get_window_extent(renderer).height
        <= ax.get_window_extent(renderer).height
    )
    # a label which fits is not wrapped
    assert "\n" not in ax.xaxis.label.get_text()
    # the figure is as tall as one of a short label
    short = MatplotlibFigureSerializer.to_figure(None, _labelled())
    assert mpl_figure.get_figheight() == pytest.approx(short.get_figheight())
