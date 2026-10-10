"""Tests of a run of experiments which has a failing experiment or figure (#270)."""

import logging
import pickle
from pathlib import Path
from typing import Any, override

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure as FigureMPL

from sbmlsim.data import Data, DataSet
from sbmlsim.experiment import (
    ExperimentResult,
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
    # the failed experiments are counted and can be filtered
    assert 'class="badge failed">failed <b>1</b>' in index
    assert 'class="chip" data-kind="failed"' in index
    assert 'data-filter="card" data-kind="failed"' in index
    assert 'data-filter="card" data-kind="passed"' in index
    # the page of the experiment says it in its header and overview
    page = (tmp_path / "BrokenTask" / "BrokenTask.html").read_text(encoding="utf-8")
    assert 'class="badge failed">experiment <b>failed</b>' in page
    overview = page.split('<section id="overview">')[1].split("</section>")[0]
    assert "The experiment failed: RuntimeError: experiment broken" in overview


def test_a_report_without_failures_has_no_filter(tmp_path: Path) -> None:
    """The chips are only there when an experiment failed."""
    results = _runner(GoodExperiment).run_experiments(output_path=tmp_path)
    ExperimentReport(results).create_report(output_path=tmp_path)
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'class="chip"' not in index
    assert "data-kind" not in index.split("<main>")[1].split("</main>")[0]
    assert "badge failed" not in index


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


class BrokenFigures(_Base):
    """An experiment whose definition of the figures raises."""

    def figures(self) -> dict:
        raise KeyError("no such figure")


class BadKey(_Base):
    """An experiment with a key which is no SId."""

    def simulations(self) -> dict:
        return {"sim": Simulation(end=20, steps=20), "sim-1": Simulation(end=5)}


class BrokenDataset(_Base):
    """An experiment whose dataset cannot be read."""

    def datasets(self) -> dict:
        raise ValueError("no such table")


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        (BrokenFigures, "no such figure"),
        (BadKey, "sim-1"),
        (BrokenDataset, "no such table"),
    ],
)
def test_an_experiment_which_fails_to_initialize_does_not_stop_the_others(
    broken: type[SimulationExperiment],
    message: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The definition of one experiment raises, the others run (#270)."""
    with caplog.at_level(logging.INFO):
        runner = _runner(broken, GoodExperiment)
        results = runner.run_experiments(output_path=tmp_path)

    name = broken.__name__
    assert [r.experiment.sid for r in results] == [name, "GoodExperiment"]
    assert message in (results[0].error or "")
    assert not results[1].failed
    assert (tmp_path / "GoodExperiment" / "GoodExperiment_fig_ok.svg").exists()
    assert not (tmp_path / name / f"{name}_fig_ok.svg").exists()
    assert name not in runner.experiments
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(r.exc_info for r in errors)
    assert any("1 of 2 experiments failed" in r.getMessage() for r in errors)


def test_a_runner_can_raise_for_an_experiment_which_fails_to_initialize() -> None:
    """A fit builds its runner with `on_error="raise"` and sees the error."""
    with pytest.raises(KeyError, match="no such figure"):
        ExperimentRunner(
            experiment_classes=[BrokenFigures, GoodExperiment],
            simulator=Simulator(),
            base_path=Path("."),
            data_path=Path("."),
            on_error="raise",
        )


def test_a_failed_initialization_is_reported_and_raised(tmp_path: Path) -> None:
    """The report lists it, `raise_on_failure` raises for it after the run."""
    with pytest.raises(ExperimentRunError, match="no such figure") as info:
        run_experiments(
            [BrokenFigures, GoodExperiment], tmp_path, raise_on_failure=True
        )
    assert [r.failed for r in info.value.results] == [True, False]
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "BrokenFigures" in index
    assert "no such figure" in index
    page = tmp_path / "BrokenFigures" / "BrokenFigures.html"
    assert "The experiment failed" in page.read_text(encoding="utf-8")


def _stale(path: Path) -> Path:
    """A figure file which an earlier run left behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<svg>an earlier run</svg>", encoding="utf-8")
    return path


@pytest.mark.parametrize("broken", [BrokenTask, BadKey])
def test_a_failed_experiment_has_no_figures_of_an_earlier_run(
    broken: type[SimulationExperiment], tmp_path: Path
) -> None:
    """A stale figure is removed and the report counts no figure for it."""
    name = broken.__name__
    stale = _stale(tmp_path / name / f"{name}_fig_ok.svg")
    results = _runner(broken, GoodExperiment).run_experiments(output_path=tmp_path)
    assert results[0].error
    assert results[0].figures == {}
    assert not stale.exists()

    ExperimentReport(results).create_report(output_path=tmp_path)
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    card = index.split(f'href="{name}/{name}.html"')[1].split('href="Good')[0]
    assert "0 figure(s)" in card
    assert "fig_ok" not in card
    page = (tmp_path / name / f"{name}.html").read_text(encoding="utf-8")
    figures = page.split('<section id="figures">')[1].split("</section>")[0]
    assert "Figures (0)" in figures
    assert "fig_ok" not in figures


def test_a_failed_figure_has_no_file_of_an_earlier_run(tmp_path: Path) -> None:
    """The stale file of a figure which fails in this run is removed."""

    class BadCurve(_Base):
        def figures(self) -> dict:
            ok, bad = _figure("fig_ok"), _bad_figure()
            ok.experiment = bad.experiment = self
            return {"fig_ok": ok, "fig_bad": bad}

    stale = _stale(tmp_path / "BadCurve" / "BadCurve_fig_bad.svg")
    (result,) = _runner(BadCurve).run_experiments(output_path=tmp_path)
    assert set(result.failed_figures) == {"fig_bad"}
    assert result.figures == {"fig_ok": ["svg"]}
    assert not stale.exists()


def test_a_figure_failing_in_its_second_format_keeps_no_file(tmp_path: Path) -> None:
    """The formats saved before the error of a figure are removed as well."""
    (result,) = _runner(GoodExperiment).run_experiments(
        output_path=tmp_path, figure_formats=["svg", "nosuchformat"]
    )
    assert "fig_ok" in result.failed_figures
    assert result.figures == {}
    assert not (tmp_path / "GoodExperiment" / "GoodExperiment_fig_ok.svg").exists()


def test_a_figure_failing_only_as_page_keeps_its_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report shows the image of a figure whose interactive page failed."""
    pytest.importorskip("plotly")
    from sbmlsim.plot import serialization_plotly

    def broken(*args: Any, **kwargs: Any) -> dict:
        raise RuntimeError("no page")

    monkeypatch.setattr(serialization_plotly, "figures_to_html", broken)
    out = tmp_path / "GoodExperiment"
    stale = _stale(out / "GoodExperiment_fig_ok.html")
    (result,) = _runner(GoodExperiment).run_experiments(
        output_path=tmp_path, figure_formats=["svg", "html"]
    )
    assert "no page" in result.failed_figures["fig_ok"]
    assert result.figures == {"fig_ok": ["svg"]}
    assert (out / "GoodExperiment_fig_ok.svg").exists()
    assert not stale.exists()

    ExperimentReport([result]).create_report(output_path=tmp_path)
    page = (out / "GoodExperiment.html").read_text(encoding="utf-8")
    figures = page.split('<section id="figures">')[1].split("</section>")[0]
    assert "Figures (1)" in figures
    assert 'src="GoodExperiment_fig_ok.svg"' in figures
    assert "GoodExperiment_fig_ok.html" not in figures


def test_the_error_of_a_run_can_be_pickled(tmp_path: Path) -> None:
    """A copy keeps the message and the failures, not the results."""
    results = _runner(BrokenTask, GoodExperiment).run_experiments(tmp_path)
    error = ExperimentRunError(results)
    copy = pickle.loads(pickle.dumps(error))
    assert isinstance(copy, ExperimentRunError)
    assert str(copy) == str(error)
    assert "experiment broken" in str(copy)
    assert copy.failures == error.failures
    assert copy.results == []


class RaisesAfterRun(_Base):
    """An experiment whose override of `run` post-processes and raises."""

    @override
    def run(self, *args: Any, **kwargs: Any) -> ExperimentResult:
        super().run(*args, **kwargs)
        raise RuntimeError("post-processing broken")


class OldSignature(_Base):
    """An experiment whose override of `run` predates `keep_results`."""

    @override
    def run(
        self,
        simulator: Simulator | None = None,
        output_path: Path | None = None,
        show_figures: bool = False,
        save_results: bool = False,
        figure_formats: list[str] | None = None,
        reduced_selections: bool = True,
    ) -> ExperimentResult:
        return super().run(
            simulator=simulator,
            output_path=output_path,
            show_figures=show_figures,
            save_results=save_results,
            figure_formats=figure_formats,
            reduced_selections=reduced_selections,
        )


class RaisesBeforeRun(_Base):
    """An override of `run` with the old signature which fails itself."""

    @override
    def run(
        self, simulator: Simulator | None = None, output_path: Path | None = None
    ) -> ExperimentResult:
        raise RuntimeError("override broken")


def test_an_override_of_run_which_raises_does_not_stop_the_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The backstop of the runner records the error and goes on (#270)."""
    stale = _stale(tmp_path / "RaisesAfterRun" / "RaisesAfterRun_fig_ok.svg")
    with caplog.at_level(logging.INFO):
        results = _runner(RaisesAfterRun, GoodExperiment).run_experiments(
            output_path=tmp_path
        )
    assert "post-processing broken" in (results[0].error or "")
    assert results[0].failed
    assert results[0].figures == {}
    assert results[1].error is None
    assert (tmp_path / "GoodExperiment" / "GoodExperiment_fig_ok.svg").exists()
    assert not stale.exists()
    # logged once, with the traceback
    logged = [r for r in caplog.records if "post-processing broken" in r.getMessage()]
    assert len([r for r in caplog.records if r.exc_info]) == 1
    assert logged or "post-processing broken" in caplog.text
    ExperimentReport(results).create_report(output_path=tmp_path)
    assert "post-processing broken" in (tmp_path / "index.html").read_text(
        encoding="utf-8"
    )


def test_an_override_of_run_with_the_old_signature_is_run(tmp_path: Path) -> None:
    """The new keyword arguments are passed only to a `run` which takes them."""
    results = _runner(OldSignature, RaisesBeforeRun, GoodExperiment).run_experiments(
        output_path=tmp_path
    )
    assert results[0].error is None
    assert (tmp_path / "OldSignature" / "OldSignature_fig_ok.svg").exists()
    assert "override broken" in (results[1].error or "")
    assert results[2].error is None
    assert (tmp_path / "GoodExperiment" / "GoodExperiment_fig_ok.svg").exists()


def test_a_raising_override_of_run_can_raise_in_a_runner_which_raises(
    tmp_path: Path,
) -> None:
    """The backstop is for the runner which logs, `on_error="raise"` raises."""
    runner = ExperimentRunner(
        experiment_classes=[RaisesAfterRun],
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
        on_error="raise",
    )
    with pytest.raises(RuntimeError, match="post-processing broken"):
        runner.run_experiments(output_path=tmp_path)


class BrokenFiguresMpl(_Base):
    """An experiment whose custom figures raise."""

    def figures_mpl(self) -> dict[str, FigureMPL]:
        raise RuntimeError("custom figures broken")


@pytest.mark.parametrize("broken", [BrokenFigures, BrokenFiguresMpl, BadKey])
def test_the_runner_removes_the_stale_figures_of_unknown_keys(
    broken: type[SimulationExperiment], tmp_path: Path
) -> None:
    """The figure keys are unknown, so the directory of the experiment is cleaned."""
    name = broken.__name__
    stale = _stale(tmp_path / name / f"{name}_never_known.svg")
    page = _stale(tmp_path / name / f"{name}_never_known.html")
    other = _stale(tmp_path / name / "other_never_known.svg")
    (result, _) = _runner(broken, GoodExperiment).run_experiments(
        output_path=tmp_path, figure_formats=["svg", "html"]
    )
    assert result.failed
    assert not stale.exists()
    assert not page.exists()
    # not a file of the experiment
    assert other.exists()
    # a figure which this run wrote stays
    for key, formats in result.figures.items():
        for fig_format in formats:
            assert (tmp_path / name / f"{name}_{key}.{fig_format}").exists()


def test_a_direct_run_in_a_shared_directory_keeps_files_of_unknown_keys(
    tmp_path: Path,
) -> None:
    """Only the directory of an experiment of its own is cleaned by the runner."""
    runner = _runner(BrokenFiguresMpl)
    experiment = runner.experiments["BrokenFiguresMpl"]
    stale = _stale(tmp_path / "BrokenFiguresMpl_never_known.svg")
    result = experiment.run(runner.simulator, output_path=tmp_path, on_error="log")
    assert result.failed_figures
    assert stale.exists()


def test_the_report_lists_no_stale_dataset_of_a_failed_experiment(
    tmp_path: Path,
) -> None:
    """A dataset file of an earlier run is not one of this run."""

    class BrokenWithDataset(_Base):
        def datasets(self) -> dict:
            return {
                "table": DataSet.from_df(pd.DataFrame({"time": [0.0]}), ureg=self.ureg)
            }

        def evaluate_fit_mappings(self) -> None:
            raise RuntimeError("experiment broken")

    stale = tmp_path / "BrokenWithDataset" / "BrokenWithDataset_table.tsv"
    stale.parent.mkdir(parents=True)
    stale.write_text("time\n0\n", encoding="utf-8")
    results = _runner(BrokenWithDataset).run_experiments(output_path=tmp_path)
    ExperimentReport(results).create_report(output_path=tmp_path)
    page = (tmp_path / "BrokenWithDataset" / "BrokenWithDataset.html").read_text(
        encoding="utf-8"
    )
    assert 'href="BrokenWithDataset_table.tsv"' not in page
