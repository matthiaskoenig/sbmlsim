"""Tests of a run of experiments which has a failing experiment or figure (#270)."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure as FigureMPL

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.report.experiment_report import ExperimentReport
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task


class _Base(SimulationExperiment):
    """An experiment of the repressilator."""

    def models(self) -> dict:
        return {"m": AbstractModel(source=REPRESSILATOR_SBML)}

    def simulations(self) -> dict:
        return {"sim": Simulation(end=20, steps=20)}

    def tasks(self) -> dict:
        return {"task": Task(model="m", simulation="sim")}

    def data(self) -> dict:
        return {"X": Data(index="[X]", task="task")}

    def figures(self) -> dict:
        figure = _figure("fig_ok")
        figure.experiment = self
        return {"fig_ok": figure}


def _figure(sid: str, style: Any = None) -> Figure:
    """A figure of the selection X, which raises when drawn with a bad style."""
    figure = Figure(experiment=None, sid=sid, num_rows=1, num_cols=1)
    plot = figure.create_plots(xaxis=Axis("time"), yaxis=Axis("X"))[0]
    kwargs: dict[str, Any] = {} if style is None else {"style": style}
    plot.curve(
        x=Data("time", task="task"),
        y=Data("[X]", task="task"),
        label="X",
        **kwargs,
    )
    return figure


class BrokenFigureMpl(_Base):
    """An experiment whose custom figure raises when it is drawn."""

    def figures_mpl(self) -> dict[str, FigureMPL]:
        raise RuntimeError("figure broken")


class GoodExperiment(_Base):
    """An experiment which works."""


class BrokenTask(_Base):
    """An experiment whose evaluation raises."""

    def evaluate_fit_mappings(self) -> None:
        raise RuntimeError("experiment broken")


class TwoFigures(_Base):
    """An experiment with a figure of the model and one which raises."""

    def figures_mpl(self) -> dict[str, FigureMPL]:
        fig = plt.figure()
        return {"fig_mpl": fig}


def _runner(*classes: type[SimulationExperiment]) -> ExperimentRunner:
    return ExperimentRunner(
        experiment_classes=list(classes),
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
    )


def test_a_failing_experiment_does_not_stop_the_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The experiment after a failing one is run and written (#270)."""
    runner = _runner(BrokenTask, GoodExperiment)
    with caplog.at_level(logging.INFO):
        results = runner.run_experiments(output_path=tmp_path)

    assert [r.experiment.sid for r in results] == ["BrokenTask", "GoodExperiment"]
    assert "experiment broken" in (results[0].error or "")
    assert results[1].error is None
    assert (tmp_path / "GoodExperiment" / "GoodExperiment_fig_ok.svg").exists()

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(r.exc_info for r in errors)
    assert "BrokenTask" in caplog.text
    assert any("1 of 2 experiments failed" in r.getMessage() for r in errors)


def test_a_failing_figure_does_not_lose_the_others(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A figure which raises is skipped, the others are written (#270)."""

    class BadCurve(_Base):
        def figures(self) -> dict:
            ok, bad = _figure("fig_ok"), _bad_figure()
            ok.experiment = bad.experiment = self
            return {"fig_ok": ok, "fig_bad": bad}

    runner = _runner(BadCurve)
    with caplog.at_level(logging.INFO):
        (result,) = runner.run_experiments(output_path=tmp_path)

    out = tmp_path / "BadCurve"
    assert result.error is None
    assert set(result.failed_figures) == {"fig_bad"}
    assert (out / "BadCurve_fig_ok.svg").exists()
    assert not (out / "BadCurve_fig_bad.svg").exists()
    assert (out / "BadCurve.json").exists()
    assert "fig_bad" in caplog.text


def _bad_figure() -> Figure:
    return _figure("fig_bad", style="not a style")


def test_a_failing_custom_figure_is_skipped(tmp_path: Path) -> None:
    """`figures_mpl` which raises fails the experiment's figures, not the run."""
    runner = _runner(BrokenFigureMpl, GoodExperiment)
    results = runner.run_experiments(output_path=tmp_path)
    assert (tmp_path / "GoodExperiment" / "GoodExperiment_fig_ok.svg").exists()
    assert (tmp_path / "BrokenFigureMpl" / "BrokenFigureMpl.json").exists()
    assert results[0].failed_figures


def test_run_called_directly_raises() -> None:
    """A user who runs one experiment sees the error."""
    runner = _runner(BrokenTask)
    experiment = runner.experiments["BrokenTask"]
    with pytest.raises(RuntimeError, match="experiment broken"):
        experiment.run(runner.simulator)


def test_the_report_lists_the_failed_experiments(tmp_path: Path) -> None:
    """The report is created and names the failed experiment."""
    runner = _runner(BrokenTask, GoodExperiment)
    results = runner.run_experiments(output_path=tmp_path)
    ExperimentReport(results).create_report(output_path=tmp_path)
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "BrokenTask" in index
    assert "experiment broken" in index
