"""Tests of rendering the figure model with matplotlib."""

import matplotlib
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from sbmlsim.plot.plotting import (
    Axis,
    Figure,
    Plot,
    SubPlot,
)
from sbmlsim.plot.serialization_matplotlib import (
    MatplotlibFigureSerializer,
)


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
