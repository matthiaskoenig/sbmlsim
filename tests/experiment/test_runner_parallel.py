"""Tests of the experiments of a run in a pool of worker processes."""

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, override

import matplotlib
import pytest

from sbmlsim import parallel
from sbmlsim.experiment import ExperimentResult, ExperimentRunner, SimulationExperiment
from sbmlsim.experiment.runner import EXPERIMENTS_PER_WORKER, _Collect, _run_job
from sbmlsim.simulator import Simulator
from tests.experiment.test_runner_errors import (
    BrokenTask,
    GoodExperiment,
    _bad_figure,
    _Base,
    _figure,
)

#: the file in which an experiment records where it ran
PROCESS_FILE = "process.json"


class _Recorded(_Base):
    """An experiment which records its process and a setting of matplotlib."""

    @override
    def run(self, *args: Any, **kwargs: Any) -> ExperimentResult:
        result = super().run(*args, **kwargs)
        output_path = kwargs["output_path"]
        (output_path / PROCESS_FILE).write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "linewidth": matplotlib.rcParams["lines.linewidth"],
                }
            )
        )
        return result


class First(_Recorded):
    """An experiment which works."""


class Second(_Recorded):
    """An experiment which works."""


class Third(_Recorded):
    """An experiment which works."""


class BadFigure(_Recorded):
    """An experiment with a figure which fails."""

    @override
    def figures(self) -> dict:
        ok, bad = _figure("fig_ok"), _bad_figure()
        ok.experiment = bad.experiment = self
        return {"fig_ok": ok, "fig_bad": bad}


class DiesInAWorker(_Recorded):
    """An experiment whose worker dies, it runs again in the parent."""

    @override
    def run(self, *args: Any, **kwargs: Any) -> ExperimentResult:
        if parallel.in_worker():
            os._exit(1)
        return super().run(*args, **kwargs)


CLASSES: list[type[SimulationExperiment]] = [
    First,
    BrokenTask,
    Second,
    BadFigure,
    GoodExperiment,
    Third,
]


def _runner(
    *classes: type[SimulationExperiment], simulator: Simulator | None = None
) -> ExperimentRunner:
    return ExperimentRunner(
        experiment_classes=list(classes),
        simulator=simulator or Simulator(n_workers=1),
        base_path=Path("."),
        data_path=Path("."),
    )


def _process(path: Path, sid: str) -> dict[str, Any]:
    return json.loads((path / sid / PROCESS_FILE).read_text())


def _outcome(result: ExperimentResult) -> tuple[Any, ...]:
    return (
        result.experiment.sid,
        result.error,
        sorted(result.failed_figures),
        result.figures,
        result.datasets,
    )


def _run(
    path: Path,
    n_workers: int | None,
    classes: list[type[SimulationExperiment]],
    simulator: Simulator | None = None,
) -> list[ExperimentResult]:
    with matplotlib.rc_context({"lines.linewidth": 4.25}):
        return _runner(*classes, simulator=simulator).run_experiments(
            output_path=path, n_workers=n_workers
        )


def test_the_experiments_of_a_pool_give_the_results_of_a_serial_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The experiments run in workers, with the settings of matplotlib of the run."""
    with caplog.at_level(logging.INFO):
        pooled = _run(tmp_path / "pooled", 3, CLASSES)
    serial = _run(tmp_path / "serial", 1, CLASSES)

    assert [_outcome(r) for r in pooled] == [_outcome(r) for r in serial]
    assert [r.experiment.sid for r in pooled] == [c.__name__ for c in CLASSES]
    # the experiments of the run, not copies of a worker, are in the results
    assert all(
        isinstance(r.experiment, c) for r, c in zip(pooled, CLASSES, strict=True)
    )
    for sid in ("First", "Second", "Third", "BadFigure"):
        assert _process(tmp_path / "pooled", sid)["pid"] != os.getpid()
        assert _process(tmp_path / "serial", sid)["pid"] == os.getpid()
        assert _process(tmp_path / "pooled", sid)["linewidth"] == 4.25
        for fig_format in ("svg",):
            assert (tmp_path / "pooled" / sid / f"{sid}_fig_ok.{fig_format}").exists()
    # the messages of the workers reach the logging of the run, in its order
    messages = [r.getMessage() for r in caplog.records]
    running = [m for m in messages if m.startswith("Running SimulationExperiment")]
    assert running == [f"Running SimulationExperiment: '{c.__name__}'" for c in CLASSES]
    assert any(
        r.levelno == logging.ERROR and "fig_bad" in r.getMessage()
        for r in caplog.records
    )
    assert "experiment broken" in caplog.text
    assert any("2 of 6 experiments failed" in m for m in messages)


def test_an_experiment_whose_worker_dies_runs_in_the_parent(tmp_path: Path) -> None:
    results = _run(tmp_path, 2, [First, DiesInAWorker, Second])

    assert [r.error for r in results] == [None, None, None]
    assert _process(tmp_path, "DiesInAWorker")["pid"] == os.getpid()


def test_a_run_of_few_experiments_is_serial_by_default(tmp_path: Path) -> None:
    _run(tmp_path / "few", None, CLASSES[:EXPERIMENTS_PER_WORKER])
    _run(tmp_path / "more", None, CLASSES)

    assert _process(tmp_path / "few", "First")["pid"] == os.getpid()
    assert _process(tmp_path / "more", "First")["pid"] != os.getpid()


def test_experiments_which_cannot_be_pickled_run_serially(tmp_path: Path) -> None:
    """A class of a function cannot reach a worker, the run is serial."""

    class Local(First):
        pass

    results = _run(tmp_path, 2, [Local, Second])

    assert [r.error for r in results] == [None, None]
    assert _process(tmp_path, "Local")["pid"] == os.getpid()
    assert _process(tmp_path, "Second")["pid"] == os.getpid()


def test_a_number_of_workers_below_one_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _run(tmp_path, 0, [First])


def test_a_simulator_with_a_pool_runs_the_experiments_serially_by_default(
    tmp_path: Path,
) -> None:
    """The scans of an experiment run serially in a worker.

    A simulator with a pool of its own runs every scan on all its workers, a
    pool of experiments would run them serially; an explicit number of
    workers runs the experiments in a pool all the same.
    """
    _run(tmp_path / "default", None, CLASSES, Simulator(n_workers=4))
    _run(tmp_path / "explicit", 2, CLASSES, Simulator(n_workers=4))

    assert _process(tmp_path / "default", "First")["pid"] == os.getpid()
    assert _process(tmp_path / "explicit", "First")["pid"] != os.getpid()


def test_an_experiment_of_the_main_script_runs_serially(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A worker imports the script without its guard, a class in it may be missing.

    A class defined in the guard pickles in the script and broke the pool.
    """
    main = type("InTheScript", (First,), {"__module__": "__main__"})
    monkeypatch.setattr(sys.modules["__main__"], "InTheScript", main, raising=False)

    with caplog.at_level(logging.INFO):
        results = _run(tmp_path, 2, [main, Second])

    assert [r.error for r in results] == [None, None]
    assert _process(tmp_path, "InTheScript")["pid"] == os.getpid()
    assert "died" not in caplog.text


def test_the_results_of_a_pooled_experiment_are_released(tmp_path: Path) -> None:
    (result, _) = _run(tmp_path, 2, [First, Second])

    with pytest.raises(RuntimeError, match="released"):
        _ = result.experiment.results


def test_a_worker_logs_its_messages_only_through_the_run(tmp_path: Path) -> None:
    """A handler of the worker, e.g. of a script which configures logging, is
    silent while an experiment runs: the run logs its messages once."""
    seen: list[logging.LogRecord] = []

    class Seen(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record)

    runner = _runner(BadFigure)
    job = runner._jobs(
        tmp_path,
        ["svg"],
        None,
        {
            "show_figures": False,
            "save_results": False,
            "reduced_selections": True,
            "keep_results": False,
            "figure_settings": None,
        },
    )["BadFigure"]
    handler = Seen()
    logging.root.addHandler(handler)
    try:
        outcome = _run_job(job)
    finally:
        logging.root.removeHandler(handler)

    assert seen == []
    assert any("fig_bad" in r.getMessage() for r in outcome.records)


def test_a_message_whose_arguments_cannot_be_pickled_is_sent_as_text() -> None:
    collect = _Collect()
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "%s", (object(),), None)
    record.unpicklable = lambda: None
    collect.emit(record)

    (sent,) = collect.records
    assert pickle.loads(pickle.dumps(sent)).getMessage().startswith("<object object")
