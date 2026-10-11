"""A figure draws the jump of a change vertically.

The output of a simulation has the states before and after a change at its
time: the line of a curve has two points at the time of the change, the
state before it and the state after it.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure, Plot
from sbmlsim.plot.serialization_matplotlib import MatplotlibFigureSerializer
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from tests.simulator.models import sbml

#: `X` grows with the rate `k`
LINEAR = "model linear\n  var X = 0\n  k = 1\n  X' = k\nend"
#: the reset of `X` at 2
RESET = Change(2, {"X": 0.0})


class JumpFigures(SimulationExperiment):
    """A reset at 2: the steps of the integrator, a grid and scans."""

    def models(self) -> dict:
        return {"m": AbstractModel(source=sbml(LINEAR))}

    def simulations(self) -> dict:
        return {
            "steps": Simulation(end=4, changes=[RESET]),
            "grid": Simulation(end=4, steps=8, changes=[RESET]),
            # the value of the reset at 2, a grid with the time 2 twice
            "reset": Scan(
                Simulation(end=4, steps=8),
                [Dimension("reset", values={"X": np.array([0.0, 1.0])}, at=2.0)],
            ),
            # the steps of the integrator, ragged and padded
            "rates": Scan(
                Simulation(end=4, changes=[RESET]),
                [Dimension("rate", values={"k": np.array([1.0, 0.5])})],
            ),
        }

    def tasks(self) -> dict:
        return {
            f"task_{key}": Task(model="m", simulation=key) for key in self._simulations
        }

    def data(self) -> dict:
        self.add_selections_data(["time", "X"])
        return {}


@pytest.fixture(scope="module")
def experiment() -> SimulationExperiment:
    runner = ExperimentRunner(
        experiment_classes=[JumpFigures],
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
        on_error="raise",
    )
    experiment = runner.experiments["JumpFigures"]
    experiment.run(runner.simulator)
    return experiment


def _figure(experiment: SimulationExperiment, draw: Callable[[Plot], Any]) -> Figure:
    figure = Figure(experiment=experiment, sid="jumps", num_rows=1, num_cols=1)
    plot = figure.create_plots(
        xaxis=Axis("time", unit="s"), yaxis=Axis("X", unit="dimensionless")
    )[0]
    draw(plot)
    return figure


def _curve(task: str, over: str | None = None) -> Callable[[Plot], Any]:
    def draw(plot: Plot) -> None:
        plot.curve(
            x=Data("time", task=f"task_{task}"),
            y=Data("X", task=f"task_{task}"),
            over=over,
            label=task,
        )

    return draw


#: the task, the dimension a curve draws a line per point of, and the values
#: of X before and after the reset of every line
CASES = [
    ("steps", None, [(2.0, 0.0)]),
    ("grid", None, [(2.0, 0.0)]),
    ("reset", "reset", [(2.0, 0.0), (2.0, 1.0)]),
    ("rates", "rate", [(2.0, 0.0), (1.0, 0.0)]),
]


def _assert_jump(x: Any, y: Any, before: float, after: float) -> None:
    """The line has the time 2 twice, the state before and after the reset."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    assert np.all(np.diff(x) >= 0)
    at = np.flatnonzero(x == 2.0)
    assert at.tolist() == [at[0], at[0] + 1]
    np.testing.assert_allclose(y[at], [before, after], atol=1e-9)
    # the line before the reset ends in the state before it
    previous = at[0] - 1
    slope = (y[at[0]] - y[previous]) / (x[at[0]] - x[previous])
    assert slope == pytest.approx(before / 2.0, rel=1e-6)


@pytest.mark.parametrize(("task", "over", "jumps"), CASES)
def test_matplotlib_draws_the_jump_vertically(
    experiment: SimulationExperiment,
    task: str,
    over: str | None,
    jumps: list[tuple[float, float]],
) -> None:
    figure = _figure(experiment, _curve(task, over))
    fig = MatplotlibFigureSerializer.to_figure(experiment, figure)
    try:
        lines = [
            line
            for line in fig.axes[0].get_lines()
            if isinstance(line, Line2D) and np.size(line.get_xdata()) > 2
        ]
        assert len(lines) == len(jumps)
        for line, (before, after) in zip(lines, jumps, strict=True):
            _assert_jump(line.get_xdata(), line.get_ydata(), before, after)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(("task", "over", "jumps"), CASES)
def test_plotly_draws_the_jump_vertically(
    experiment: SimulationExperiment,
    task: str,
    over: str | None,
    jumps: list[tuple[float, float]],
) -> None:
    pytest.importorskip("plotly")
    from sbmlsim.plot.serialization_plotly import PlotlyFigureSerializer

    figure = _figure(experiment, _curve(task, over))
    traces = [
        trace
        for trace in PlotlyFigureSerializer.to_figure(experiment, figure).data
        if trace.x is not None and len(trace.x) > 2
    ]
    assert len(traces) == len(jumps)
    for trace, (before, after) in zip(traces, jumps, strict=True):
        _assert_jump(trace.x, trace.y, before, after)


def test_the_jump_in_a_png(experiment: SimulationExperiment, tmp_path: Path) -> None:
    """The figure of every task renders."""
    for task, over, _ in CASES:
        figure = _figure(experiment, _curve(task, over))
        fig = MatplotlibFigureSerializer.to_figure(experiment, figure)
        try:
            fig.savefig(tmp_path / f"{task}.png")
        finally:
            plt.close(fig)
        assert (tmp_path / f"{task}.png").stat().st_size > 0
