"""Tests of a run of experiments which has a failing experiment or figure (#270)."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure as FigureMPL

from sbmlsim.data import Data
from sbmlsim.experiment import (
    ExperimentRunError,
    ExperimentRunner,
    SimulationExperiment,
)
from sbmlsim.experiment.runner import run_experiments
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.report.experiment_report import ExperimentReport
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task

ReportType = ExperimentReport.ReportType


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


def test_a_failing_experiment_releases_its_results_and_figures(
    tmp_path: Path,
) -> None:
    """A failing experiment holds no results and leaves no figure open."""
    plt.close("all")
    runner = _runner(BrokenTask)
    (result,) = runner.run_experiments(output_path=tmp_path)
    assert result.failed
    assert result.experiment._results == {}
    assert plt.get_fignums() == []


def test_pyplot_figures_of_a_failing_figures_mpl_are_closed(tmp_path: Path) -> None:
    """The pyplot figures created before the error of `figures_mpl` are closed."""

    class PyplotThenError(_Base):
        def figures_mpl(self) -> dict[str, FigureMPL]:
            plt.figure()
            raise RuntimeError("late")

    plt.close("all")
    runner = _runner(PyplotThenError)
    (result,) = runner.run_experiments(output_path=tmp_path)
    assert "figures_mpl()" in result.failed_figures
    assert plt.get_fignums() == []


def test_a_figure_failing_in_two_passes_is_recorded_once_with_both() -> None:
    """The second failure of a key is appended and does not overwrite."""
    runner = _runner(GoodExperiment)
    experiment = runner.experiments["GoodExperiment"]
    failed: dict[str, str] = {}
    experiment._figure_failed("f", ValueError("a"), "log", failed)
    experiment._figure_failed("f", ValueError("b"), "log", failed)
    assert failed == {"f": "ValueError: a; ValueError: b"}


def test_raise_on_failure_raises_after_every_experiment_ran(tmp_path: Path) -> None:
    """The error names the failures; the experiments after it ran."""
    runner = _runner(BrokenTask, GoodExperiment)
    with pytest.raises(ExperimentRunError, match="experiment broken") as info:
        runner.run_experiments(output_path=tmp_path, raise_on_failure=True)
    assert [r.failed for r in info.value.results] == [True, False]
    assert (tmp_path / "GoodExperiment" / "GoodExperiment_fig_ok.svg").exists()


def test_the_module_function_returns_the_results_and_raises_after_the_report(
    tmp_path: Path,
) -> None:
    """The report is written before `raise_on_failure` raises."""
    results = run_experiments([GoodExperiment], output_path=tmp_path / "a")
    assert [r.failed for r in results] == [False]
    with pytest.raises(ExperimentRunError):
        run_experiments(
            [BrokenTask, GoodExperiment], tmp_path / "b", raise_on_failure=True
        )
    assert (tmp_path / "b" / "index.html").exists()


def test_the_markdown_and_latex_reports_list_failures(tmp_path: Path) -> None:
    """Failed experiments and figures are in every report type."""
    results = _runner(BrokenTask, GoodExperiment).run_experiments(
        output_path=tmp_path, figure_formats=["svg", "png"]
    )
    report = ExperimentReport(results)
    report.create_report(output_path=tmp_path, report_type=ReportType.MARKDOWN)
    assert "The experiment failed:** RuntimeError: experiment broken" in (
        tmp_path / "index.md"
    ).read_text(encoding="utf-8")
    assert "The experiment failed:** RuntimeError: experiment broken" in (
        tmp_path / "BrokenTask" / "BrokenTask.md"
    ).read_text(encoding="utf-8")
    report.create_report(output_path=tmp_path, report_type=ReportType.LATEX)
    assert "% The experiment BrokenTask failed" in (tmp_path / "index.tex").read_text(
        encoding="utf-8"
    )


def test_a_direct_run_can_log_a_failing_figure(tmp_path: Path) -> None:
    """`run(on_error="log")` skips the figure and returns its record."""
    runner = _runner(BrokenFigureMpl)
    experiment = runner.experiments["BrokenFigureMpl"]
    result = experiment.run(runner.simulator, output_path=tmp_path, on_error="log")
    assert "figures_mpl()" in result.failed_figures
    assert not result.error
