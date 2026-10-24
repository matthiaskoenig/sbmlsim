"""Tests of the progress and the timings of a run of experiments."""

import logging
import time
from pathlib import Path
from typing import Any, override

import pytest
from rich.console import Console

from sbmlsim.experiment import ExperimentResult, ExperimentRunner, SimulationExperiment
from sbmlsim.experiment import progress as progress_module
from sbmlsim.experiment.progress import (
    Event,
    Phase,
    PhaseClock,
    RunProgress,
    reporting,
    scan_progress,
    show_progress,
    show_timings,
    timings_table,
)
from sbmlsim.simulator import Simulator
from tests.experiment.test_experiment_run import FitMappingExperiment
from tests.experiment.test_runner_errors import BrokenTask, GoodExperiment
from tests.experiment.test_runner_parallel import DiesInAWorker, First, Second, Third

#: the seconds an experiment sleeps after its run, see `Slow`
SLEEP = 0.2


class Slow(GoodExperiment):
    """An experiment whose `run` does something after the run of its base class."""

    @override
    def run(self, *args: Any, **kwargs: Any) -> ExperimentResult:
        result = super().run(*args, **kwargs)
        time.sleep(SLEEP)
        return result


def _runner(*classes: type[SimulationExperiment]) -> ExperimentRunner:
    return ExperimentRunner(
        list(classes),
        simulator=Simulator(n_workers=1),
        base_path=Path("."),
        data_path=Path("."),
    )


def _record(monkeypatch: pytest.MonkeyPatch) -> list[Event]:
    """Record the events the display of a run receives."""
    events: list[Event] = []
    show = RunProgress.__call__

    def record(self: RunProgress, event: Event) -> None:
        events.append(event)
        show(self, event)

    monkeypatch.setattr(RunProgress, "__call__", record)
    return events


def test_a_run_without_progress_prints_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`progress=False` prints nothing and logs no timings.

    The runner is used as a library, e.g. in a notebook, where a rule
    printed before every experiment showed as a bare line; the timings of
    the steps of a run were log messages.
    """
    runner = ExperimentRunner(FitMappingExperiment, simulator=Simulator())
    with caplog.at_level(logging.INFO, logger="sbmlsim"):
        [result] = runner.run_experiments(
            output_path=tmp_path, save_results=False, progress=False
        )
    assert capsys.readouterr().out == ""
    assert caplog.records == []
    # the timings are measured all the same
    assert result.timings[Phase.SIMULATIONS] > 0


def test_a_run_shows_no_progress_when_the_output_is_redirected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """By default a run shows its progress on a terminal only.

    The output of pytest is no terminal, as a file or a pipe is not, which
    would collect the frames of the bars.
    """
    for variable in ("FORCE_COLOR", "TTY_COMPATIBLE", "TTY_INTERACTIVE"):
        monkeypatch.delenv(variable, raising=False)
    runner = ExperimentRunner(FitMappingExperiment, simulator=Simulator())
    runner.run_experiments(output_path=tmp_path, save_results=False)
    assert capsys.readouterr().out == ""


def test_show_progress() -> None:
    """`None` shows the progress on an interactive terminal and in a notebook."""
    assert show_progress(True, Console(force_terminal=False))
    assert not show_progress(False, Console(force_terminal=True))
    assert show_progress(None, Console(force_terminal=True, force_interactive=True))
    assert not show_progress(None, Console(force_terminal=False))
    assert not show_progress(
        None, Console(force_terminal=True, force_interactive=False)
    )
    assert show_progress(None, Console(force_terminal=False, force_jupyter=True))


def test_a_run_with_progress_ends_with_its_timings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The progress ends in a single line and the table of the timings.

    A console which is no terminal gets the last frame of the progress
    only, no frame of the run.
    """
    runner = _runner(GoodExperiment, BrokenTask)
    runner.run_experiments(output_path=tmp_path, n_workers=1, progress=True)
    out = capsys.readouterr().out
    lines = out.splitlines()
    words = " ".join(out.split())

    assert len([line for line in lines if "experiments" in line]) == 2
    [overall] = [line for line in lines if line.startswith("✓ experiments")]
    assert "2/2" in overall and "1 failed" in overall
    assert "Timings in seconds" in out
    assert "2 experiments, initialized in" in words
    assert "in this process" in words
    for row in ("✓ GoodExperiment", "✗ BrokenTask", "total", "share"):
        assert any(line.startswith(row) for line in lines), row
    header = next(line for line in lines if line.startswith("experiment "))
    for phase in ("initialize", "simulations", "figures", "outputs", "total"):
        assert phase in header


def test_the_timings_of_an_experiment(tmp_path: Path) -> None:
    """The phases are measured where they run and add up to the experiment.

    The models are loaded once per runner, so the second experiment of the
    same model has no time for them; what a `run` of a subclass does around
    the run of its base class is `other`.
    """
    results = _runner(GoodExperiment, Slow).run_experiments(
        output_path=tmp_path, n_workers=1, progress=False
    )
    good, slow = (r.timings for r in results)
    assert set(good) == {phase.value for phase in Phase}
    assert set(slow) == {phase.value for phase in Phase} - {Phase.MODELS}
    assert all(seconds >= 0 for seconds in [*good.values(), *slow.values()])
    assert slow[Phase.OTHER] >= SLEEP > good[Phase.OTHER]
    assert results[0].to_dict()["timings"] == good


def test_the_timings_of_a_failed_definition(tmp_path: Path) -> None:
    """An experiment whose definition failed has the time until it failed."""

    class BrokenDefinition(GoodExperiment):
        @override
        def tasks(self) -> dict:
            raise RuntimeError("definition broken")

    runner = _runner(BrokenDefinition)
    [result] = runner.run_experiments(output_path=tmp_path, progress=False)
    assert result.error is not None
    assert set(result.timings) == {Phase.MODELS, Phase.INITIALIZE}


def test_the_events_of_a_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The display of a run sees every activity of an experiment and its steps."""
    events = _record(monkeypatch)
    _runner(GoodExperiment).run_experiments(
        output_path=tmp_path, n_workers=1, progress=True
    )
    activities = list(dict.fromkeys((e.phase, e.activity) for e in events))
    assert activities == [
        (Phase.SIMULATIONS, "simulations"),
        (Phase.DATA, "fit mappings"),
        (Phase.OUTPUTS, "writing datasets"),
        (Phase.FIGURES, "drawing figures"),
        (Phase.FIGURES, "writing figures"),
        (Phase.OUTPUTS, "writing JSON"),
    ]
    assert {e.sid for e in events} == {"GoodExperiment"}
    simulations = [e for e in events if e.phase == Phase.SIMULATIONS]
    assert [(e.completed, e.total) for e in simulations] == [(0, 1), (1, 1)]


def test_the_workers_report_their_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The workers send their events to the display of the parent.

    Every worker loads the models of its experiments itself, which its
    timings show.
    """
    events = _record(monkeypatch)
    results = _runner(First, Second, Third).run_experiments(
        output_path=tmp_path, n_workers=2, progress=True
    )
    out = capsys.readouterr().out
    words = " ".join(out.split())

    assert {e.sid for e in events} == {"First", "Second", "Third"}
    phases = {e.phase for e in events}
    assert {Phase.INITIALIZE, Phase.SIMULATIONS, Phase.FIGURES} <= phases
    assert all(r.timings[Phase.SIMULATIONS] > 0 for r in results)
    assert any(Phase.MODELS in r.timings for r in results)
    assert "on 2 workers" in words
    [overall] = [line for line in out.splitlines() if "✓ experiments" in line]
    assert "3/3" in overall and "2 workers" in overall


def test_a_run_without_progress_sends_no_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pool of a run without progress has no queue to the parent."""
    queues: list[object] = []
    monkeypatch.setattr(
        "sbmlsim.experiment.runner._set_events", lambda q: queues.append(q)
    )
    results = _runner(First, Second).run_experiments(
        output_path=tmp_path, n_workers=2, progress=False
    )
    assert queues == []
    assert all(r.timings[Phase.SIMULATIONS] > 0 for r in results)


def test_a_phase_clock() -> None:
    """A clock adds up its phases and reports them to the listener of its context."""
    events: list[Event] = []
    with reporting(events.append):
        clock = PhaseClock("e")
    with clock.phase(Phase.FIGURES, "drawing figures", 2):
        progress_module.advance()
        progress_module.advance()
    with clock.phase(Phase.FIGURES, "writing figures", 0):
        pass
    assert [(e.activity, e.completed, e.total) for e in events] == [
        ("drawing figures", 0, 2),
        ("drawing figures", 1, 2),
        ("drawing figures", 2, 2),
        ("writing figures", 0, 1),
    ]
    assert set(clock.timings) == {"figures"}
    # outside of a phase there is nothing to advance
    progress_module.advance()
    assert len(events) == 4


def test_a_failing_listener_does_not_fail_a_phase() -> None:
    """The display of a run never fails the run."""

    def fail(event: Event) -> None:
        raise RuntimeError("display broken")

    with reporting(fail):
        clock = PhaseClock("e")
    with clock.phase(Phase.SIMULATIONS, "simulations"):
        progress_module.advance()
    assert Phase.SIMULATIONS in clock.timings


def test_scan_progress() -> None:
    """A run without progress shows no bars of its scans either."""
    assert scan_progress() is None
    with reporting(None):
        assert scan_progress() is False
    with reporting(lambda event: None):
        assert scan_progress() is None


def _result(sid: str, timings: dict[str, float], error: str | None = None) -> Any:
    experiment = type("E", (), {"sid": sid})()
    return ExperimentResult(
        experiment=experiment,  # ty: ignore[invalid-argument-type]
        output_path=None,
        error=error,
        timings=timings,
    )


def test_the_table_of_the_timings() -> None:
    """A row per experiment, the totals and the shares; phases without time are left out."""
    results = [
        _result("a", {"models": 1.0, "simulations": 2.0, "other": 0.001}),
        _result("b", {"simulations": 1.0}, error="broken"),
    ]
    console = Console(width=100, record=True)
    console.print(timings_table(results))
    text = console.export_text()
    lines = text.splitlines()
    header = lines[0]
    assert header.split() == ["experiment", "models", "simulations", "total"]
    rows = {line.split()[0]: line.split() for line in lines[2:] if line[0] != "─"}
    assert rows["✓"][1:5] == ["a", "1.00", "2.00", "3.00"]
    assert rows["✗"][1:5] == ["b", "-", "1.00", "1.00"]
    assert rows["total"][1:4] == ["1.00", "3.00", "4.00"]
    assert rows["share"][1:4] == ["25%", "75%", "100%"]
    assert rows["✓"][5].startswith("━")


def test_the_bars_give_way_on_a_narrow_console() -> None:
    """The numbers stay readable, the bars are left out on a narrow console."""
    results = [_result("experiment_with_a_long_name", {"simulations": 2.0})]
    narrow = Console(width=40, record=True)
    show_timings(results, initialized=0.5, elapsed=2.0, where="here", console=narrow)
    text = narrow.export_text()
    assert "━" not in text
    assert "2.00" in text
    wide = Console(width=100, record=True)
    show_timings(results, initialized=0.5, elapsed=2.0, where="here", console=wide)
    text = wide.export_text()
    assert "━" in text
    assert "1 experiment, initialized in 0.50 s and run in 2.00 s here" in text


def test_a_second_run_has_no_initialization(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The initialization of the runner belongs to its first run only."""
    runner = _runner(GoodExperiment)
    [first] = runner.run_experiments(output_path=tmp_path, progress=True)
    assert "initialized in" in " ".join(capsys.readouterr().out.split())
    [second] = runner.run_experiments(output_path=tmp_path, progress=True)
    out = " ".join(capsys.readouterr().out.split())
    assert {Phase.MODELS, Phase.INITIALIZE} <= set(first.timings)
    assert not {Phase.MODELS, Phase.INITIALIZE} & set(second.timings)
    assert "1 experiment run in" in out and "initialized" not in out


def test_an_experiment_whose_worker_died_is_counted_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An experiment whose worker died runs again here and is counted then."""
    results = _runner(First, DiesInAWorker, Second).run_experiments(
        output_path=tmp_path, n_workers=2, progress=True
    )
    out = capsys.readouterr().out
    assert [r.error for r in results] == [None, None, None]
    [overall] = [line for line in out.splitlines() if "✓ experiments" in line]
    assert "3/3" in overall
    assert all(r.timings[Phase.SIMULATIONS] > 0 for r in results)
