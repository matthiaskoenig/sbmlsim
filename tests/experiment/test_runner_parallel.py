"""Tests of the experiments of a run in a pool of worker processes."""

import json
import logging
import os
from pathlib import Path
from typing import Any, override

import matplotlib
import pytest

from sbmlsim import parallel
from sbmlsim.experiment import ExperimentResult, ExperimentRunner, SimulationExperiment
from sbmlsim.experiment.runner import EXPERIMENTS_PER_WORKER
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


def _runner(*classes: type[SimulationExperiment]) -> ExperimentRunner:
    return ExperimentRunner(
        experiment_classes=list(classes),
        simulator=Simulator(n_workers=4),
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
    path: Path, n_workers: int | None, classes: list[type[SimulationExperiment]]
) -> list[ExperimentResult]:
    with matplotlib.rc_context({"lines.linewidth": 4.25}):
        return _runner(*classes).run_experiments(output_path=path, n_workers=n_workers)


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
