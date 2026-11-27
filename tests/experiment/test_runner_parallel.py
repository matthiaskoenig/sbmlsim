"""Tests of the experiments of a run in a pool of worker processes."""

import dataclasses
import json
import logging
import os
import pickle
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, override

import pytest

from sbmlsim import parallel
from sbmlsim.engine import Engine
from sbmlsim.experiment import ExperimentResult, ExperimentRunner, SimulationExperiment
from sbmlsim.experiment import runner as runner_module
from sbmlsim.experiment.runner import (
    EXPERIMENTS_PER_WORKER,
    _Collect,
    _run_job,
    _start_early_pool,
    _warm,
)
from sbmlsim.model.model_roadrunner import resolve_jit_optimization
from sbmlsim.simulation import Dimension, Scan, Simulation
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
    """An experiment which records its process."""

    @override
    def run(self, *args: Any, **kwargs: Any) -> ExperimentResult:
        result = super().run(*args, **kwargs)
        output_path = kwargs["output_path"]
        (output_path / PROCESS_FILE).write_text(
            json.dumps({"pid": os.getpid()}),
            encoding="utf-8",
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


#: the points of the scan of `Scanned`
SCAN_POINTS = 4


class Scanned(_Recorded):
    """An experiment whose simulation is a scan, which a simulator may pool."""

    @override
    def simulations(self) -> dict:
        return {
            "sim": Scan(
                Simulation(end=20, steps=20),
                [Dimension("k", values={"KM": [40.0 + k for k in range(SCAN_POINTS)]})],
            )
        }

    @override
    def figures(self) -> dict:
        """No figure: a curve of the task would need the scan dimension."""
        return {}


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
    return json.loads((path / sid / PROCESS_FILE).read_text(encoding="utf-8"))


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
    return _runner(*classes, simulator=simulator).run_experiments(
        output_path=path, n_workers=n_workers
    )


def test_the_experiments_of_a_pool_give_the_results_of_a_serial_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The experiments run in workers and give the results of a serial run."""
    with caplog.at_level(logging.DEBUG, logger="sbmlsim.experiment.runner"):
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
        for fig_format in ("svg", "json"):
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


def test_the_experiments_without_a_scan_run_in_a_pool_beside_a_pooled_simulator(
    tmp_path: Path,
) -> None:
    """A simulator with a pool of its own is no reason to keep an experiment here.

    Only a scan runs in the pool of the simulator, so an experiment which has
    none is faster in a worker whatever the simulator does (#554).
    """
    _run(tmp_path / "default", None, CLASSES, Simulator(n_workers=4))
    _run(tmp_path / "explicit", 2, CLASSES, Simulator(n_workers=4))

    assert _process(tmp_path / "default", "First")["pid"] != os.getpid()
    assert _process(tmp_path / "explicit", "First")["pid"] != os.getpid()


def test_an_experiment_whose_scans_need_the_pool_of_the_simulator_runs_here(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A scan of a simulator with a pool runs in this process, not in a worker (#554).

    Roadrunner runs the points of a scan in processes, which a worker cannot
    start; the engine core runs them on threads, which a worker has, see
    `test_an_experiment_of_the_engine_core_runs_its_scans_in_a_worker`.

    An experiment runs its scans serially in a worker, so an experiment of a
    scan which the simulator would run in a pool of its own is faster here:
    it runs in this process and the experiments which have no such scan run
    in the pool beside it, also when the run named its workers itself, which
    skipped this guard and left every scan of such a run on one core.
    """
    with caplog.at_level(logging.DEBUG, logger="sbmlsim.experiment.runner"):
        results = _run(
            tmp_path,
            2,
            [Scanned, First, Second, Third],
            Simulator(4, engine="roadrunner"),
        )

    assert [r.error for r in results] == [None, None, None, None]
    assert _process(tmp_path, "Scanned")["pid"] == os.getpid()
    for sid in ("First", "Second", "Third"):
        assert _process(tmp_path, sid)["pid"] != os.getpid(), sid
    assert (
        "1 experiments run in this process: the simulator runs the scans in a pool of 4 workers"
        in caplog.text
    )


def test_an_experiment_of_the_engine_core_runs_its_scans_in_a_worker(
    tmp_path: Path,
) -> None:
    """The engine core runs a scan on threads, which a worker has (sbmlsim#664).

    A simulator which names its workers no longer sends every experiment of a
    scan into this process: with the draws of an uncertainty run that was
    every experiment, one after the other. A scan below the threshold of a
    pool of every CPU runs in its worker, on the share of the worker of the
    CPUs (`parallel.worker_cores`).
    """
    results = _run(
        tmp_path, 2, [Scanned, First, Second, Third], Simulator(4, engine="core")
    )

    assert [r.error for r in results] == [None, None, None, None]
    for sid in ("Scanned", "First", "Second", "Third"):
        assert _process(tmp_path, sid)["pid"] != os.getpid(), sid


def test_the_threads_of_a_worker_are_its_share_of_the_cpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker of a pool of n takes 1/n of the CPUs, also when a run names more."""
    from sbmlsim.simulator.worker import core_threads

    monkeypatch.setattr(parallel, "in_worker", lambda: True)
    monkeypatch.setattr(parallel.cgroup, "available_cpus", lambda: 20)
    monkeypatch.setattr(parallel, "_POOL_SIZE", 7)
    assert parallel.worker_cores() == 2
    assert core_threads(None, 100) == 2
    assert core_threads(16, 100) == 2
    assert core_threads(1, 100) == 1
    assert core_threads(None, 1) == 1
    monkeypatch.setattr(parallel, "_POOL_SIZE", None)
    assert core_threads(None, 100) == 1


def test_the_share_of_a_worker_follows_the_workers_which_run_a_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CPUs which the workers out of tasks leave go to the ones which run one.

    Without a worker counted busy a worker of a pool of n has 1/n of the CPUs;
    the last worker which runs a task has every CPU (sbmlsim#664).
    """
    import multiprocessing

    from sbmlsim.simulator.worker import core_threads

    monkeypatch.setattr(parallel, "in_worker", lambda: True)
    monkeypatch.setattr(parallel.cgroup, "available_cpus", lambda: 20)
    monkeypatch.setattr(parallel, "_POOL_SIZE", 7)
    busy = multiprocessing.get_context("spawn").Value("i", 0)
    monkeypatch.setattr(parallel, "_BUSY", busy)

    assert parallel.worker_cores() == 2
    with parallel.working():
        assert busy.value == 1
        assert parallel.worker_cores() == 20
        assert core_threads(None, 100) == 20
        assert core_threads(8, 100) == 8
        # two other workers run a task in python, one CPU each
        busy.value += 2
        assert parallel.worker_cores() == 18
        busy.value -= 2
    assert busy.value == 0
    assert parallel.worker_cores() == 2


def test_the_workers_which_run_threads_share_the_cpus_the_others_leave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker in python takes one CPU, the ones which run threads share the rest.

    The long experiment of an uncertainty run got a fifth of the CPUs of a
    pool of five busy workers while the other four ran python, their
    initialization and their figures, and left most of the machine idle.
    """
    import multiprocessing

    from sbmlsim.simulator.worker import core_threads

    context = multiprocessing.get_context("spawn")
    busy, threaded = context.Value("i", 0), context.Value("i", 0)
    monkeypatch.setattr(parallel, "in_worker", lambda: True)
    monkeypatch.setattr(parallel.cgroup, "available_cpus", lambda: 20)
    monkeypatch.setattr(parallel, "_POOL_SIZE", 7)
    monkeypatch.setattr(parallel, "_BUSY", busy)
    monkeypatch.setattr(parallel, "_THREADED", threaded)

    with parallel.working():
        # four other workers in python: this one takes the other 16 CPUs
        busy.value += 4
        assert parallel.worker_cores() == 16
        with parallel.threads():
            assert threaded.value == 1
            assert core_threads(None, 100) == 16
            # a second worker runs threads as well: they share 16 CPUs
            threaded.value += 1
            assert parallel.worker_cores() == 8
            threaded.value -= 1
            with parallel.threads():
                # counted once
                assert threaded.value == 1
        assert threaded.value == 0
        busy.value -= 4
    assert busy.value == 0


def test_a_process_beside_its_pool_shares_the_cpus_with_its_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The experiment which a run keeps in its process counts among the busy workers.

    Its threads are its share of the CPUs, as a worker's, and not every CPU
    beside the threads of the workers; a pool it did not start counts nothing.
    """
    from sbmlsim.simulator.worker import core_threads

    monkeypatch.setattr(parallel.cgroup, "available_cpus", lambda: 20)
    executor = parallel.start_pool(2)
    try:
        _, busy, threaded = parallel._COUNTERS[executor]
        assert not parallel.shares_cpus()
        with parallel.beside(executor):
            assert parallel.shares_cpus()
            assert busy.value == 1
            assert core_threads(None, 100) == 20
            assert core_threads(4, 100) == 4
            # a worker runs threads as well
            threaded.value += 1
            with parallel.threads():
                assert threaded.value == 2
                assert core_threads(None, 100) == 10
            threaded.value -= 1
            with parallel.beside(executor):
                assert busy.value == 1
        assert busy.value == 0
        assert not parallel.shares_cpus()
    finally:
        parallel.stop(executor)
    with parallel.beside(executor):
        assert not parallel.shares_cpus()


class Threads(Scanned):
    """An experiment of a scan which records the threads of its simulator."""

    @override
    def run(self, *args: Any, **kwargs: Any) -> ExperimentResult:
        result = super().run(*args, **kwargs)
        simulator = args[0] if args else kwargs["simulator"]
        busy, threaded = parallel._BUSY, parallel._THREADED
        (kwargs["output_path"] / "threads.json").write_text(
            json.dumps(
                {
                    "threads": simulator._workers(100)[1],
                    "busy": None if busy is None else busy.value,
                    "threaded": None if threaded is None else threaded.value,
                    "cores": parallel.worker_cores(),
                }
            ),
            encoding="utf-8",
        )
        return result


def test_a_worker_runs_the_scans_of_the_engine_core_on_its_share(
    tmp_path: Path,
) -> None:
    """A worker builds a simulator of the workers of the run, held to its share.

    It built a simulator of one worker, which ran the engine core on one thread
    whatever its share of the CPUs was. The worker counts itself busy while it
    runs the experiment, and among the ones which run threads while it scans.
    """
    results = _run(tmp_path, 2, [Threads, First], Simulator(4, engine="core"))

    assert [r.error for r in results] == [None, None]
    assert _process(tmp_path, "Threads")["pid"] != os.getpid()
    record = json.loads(
        (tmp_path / "Threads" / "threads.json").read_text(encoding="utf-8")
    )
    assert record["busy"] >= 1
    assert record["threads"] == min(4, record["cores"])
    cpus = parallel.cgroup.available_cpus()
    # the worker runs threads, the other busy workers without threads one
    # CPU each
    threaded = record["threaded"] + 1
    single = max(0, record["busy"] - threaded)
    assert record["cores"] == max(1, (cpus - single) // threaded)


def test_an_experiment_of_a_scan_which_fills_the_machine_runs_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A simulator without a pool runs a scan of many points in one of every CPU."""
    monkeypatch.setattr(parallel, "POOL_THRESHOLD", SCAN_POINTS)

    _run(tmp_path, 2, [Scanned, First, Second, Third], Simulator())

    assert _process(tmp_path, "Scanned")["pid"] == os.getpid()
    assert _process(tmp_path, "First")["pid"] != os.getpid()


class LargerScan(Scanned):
    """An experiment of a scan of more points than `Scanned`."""

    @override
    def simulations(self) -> dict:
        return {
            "sim": Scan(
                Simulation(end=20, steps=20),
                [
                    Dimension(
                        "k",
                        values={"KM": [40.0 + k for k in range(2 * SCAN_POINTS)]},
                    )
                ],
            )
        }


def test_the_engine_core_runs_one_experiment_of_such_a_scan_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The process runs the most expensive one, the others run in the pool.

    One process runs one experiment after the other: the HCTZ model of
    pkdb_models ran two experiments of such a scan here over the draws of its
    parameters, the second after the first, while the workers of the engine
    core run such a scan on their share of the CPUs as well.
    """
    monkeypatch.setattr(parallel, "POOL_THRESHOLD", SCAN_POINTS)

    results = _run(
        tmp_path, 2, [Scanned, LargerScan, First, Second], Simulator(engine="core")
    )

    assert [r.error for r in results] == [None, None, None, None]
    assert _process(tmp_path, "LargerScan")["pid"] == os.getpid()
    for sid in ("Scanned", "First", "Second"):
        assert _process(tmp_path, sid)["pid"] != os.getpid(), sid


def test_a_run_of_nothing_but_such_experiments_is_serial(tmp_path: Path) -> None:
    """A pool of one experiment is no pool, so the run is serial with its reason."""
    results = _run(
        tmp_path, 2, [Scanned, First], Simulator(n_workers=4, engine="roadrunner")
    )

    assert [r.error for r in results] == [None, None]
    assert _process(tmp_path, "Scanned")["pid"] == os.getpid()
    assert _process(tmp_path, "First")["pid"] == os.getpid()


def test_the_timings_name_the_experiments_which_ran_in_this_process(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The table of the timings says what was overridden and why (#554)."""
    _runner(
        Scanned,
        First,
        Second,
        Third,
        simulator=Simulator(n_workers=4, engine="roadrunner"),
    ).run_experiments(output_path=tmp_path, n_workers=2, progress=True)

    words = " ".join(capsys.readouterr().out.split())
    assert (
        "on 2 workers, 1 in this process (the simulator runs the scans in a pool "
        "of 4 workers)" in words
    )


class _OwnSimulator(Simulator):
    """A simulator of its own, which a worker does not build."""


def test_a_subclass_of_the_simulator_runs_the_experiments_serially(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG, logger="sbmlsim.experiment.runner"):
        results = _run(tmp_path, 2, [First, Second], _OwnSimulator(n_workers=1))

    assert [r.error for r in results] == [None, None]
    assert _process(tmp_path, "First")["pid"] == os.getpid()
    assert _process(tmp_path, "Second")["pid"] == os.getpid()
    assert "the simulator is a '_OwnSimulator', a worker builds a 'Simulator'" in (
        caplog.text
    )


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


class _BrokenSimulations(_Recorded):
    """An experiment whose definition raises."""

    @override
    def simulations(self) -> dict:
        raise ValueError("the simulations are broken")


def _early(
    classes: list[type[SimulationExperiment]],
    n_workers: int | None,
    simulator: Simulator | None = None,
) -> runner_module._EarlyPool | None:
    return _start_early_pool(
        classes,
        simulator or Simulator(),
        {},
        n_workers,
        False,
        Path("."),
        Path("."),
        resolve_jit_optimization(None),
    )


def test_run_experiments_starts_the_pool_before_it_initializes_the_experiments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The workers start while the experiments are initialized, the run takes them.

    The start of the pool, the import of the modules of the experiments in
    the forkserver, waited for the initialization of every experiment, and
    the workers loaded the models after it.
    """
    calls: list[str] = []
    start_pool_soon = parallel.start_pool_soon
    initialize = ExperimentRunner.initialize

    def soon(*args: Any, **kwargs: Any) -> parallel.PendingPool | None:
        calls.append("pool")
        return start_pool_soon(*args, **kwargs)

    def initialized(self: ExperimentRunner, *args: Any, **kwargs: Any) -> None:
        calls.append("initialize")
        initialize(self, *args, **kwargs)

    started: list[int] = []
    start_pool = parallel.start_pool

    def start(n_workers: int, **kwargs: Any) -> Any:
        started.append(n_workers)
        return start_pool(n_workers, **kwargs)

    monkeypatch.setattr(parallel, "start_pool_soon", soon)
    monkeypatch.setattr(parallel, "start_pool", start)
    monkeypatch.setattr(ExperimentRunner, "initialize", initialized)
    classes: list[type[SimulationExperiment]] = [First, Second, Third]
    results = runner_module.run_experiments(
        classes, tmp_path, base_path=Path("."), n_workers=2, progress=False
    )

    assert calls == ["pool", "initialize"]
    # the run takes the pool which started early, it starts no other
    assert started == [2]
    assert [r.error for r in results] == [None, None, None]
    for sid in ("First", "Second", "Third"):
        assert _process(tmp_path, sid)["pid"] != os.getpid()


def test_the_early_pool_has_the_workers_of_the_run() -> None:
    """At most one per experiment, one per four experiments by default."""
    classes = CLASSES * 2
    for n_workers, expected in ((None, 2), (3, 3), (20, len(set(classes)))):
        early = _early(classes, n_workers)
        assert early is not None
        try:
            assert early.pending.n_workers == expected
            assert early.events is None
        finally:
            early.pending.stop()


def test_no_early_pool_where_the_run_starts_none_before_its_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No pool for a serial run, nor before the pool of a simulator of processes."""

    class Local(First):
        pass

    assert _early([First, Second], 1) is None
    assert _early(CLASSES[:EXPERIMENTS_PER_WORKER], None) is None
    assert _early([Local, Second], 2) is None
    assert _early([First, Second], 2, Simulator(engine=Engine.ROADRUNNER)) is None
    # roadrunner which runs every scan serially starts no pool of its own
    early = _early([First, Second], 2, Simulator(n_workers=1, engine="roadrunner"))
    assert early is not None
    early.pending.stop()
    monkeypatch.setattr(parallel, "in_worker", lambda: True)
    assert _early([First, Second], 2) is None


def test_a_pool_of_more_workers_is_taken_by_the_run() -> None:
    early = _early([First, Second, Third], 3)
    assert early is not None
    try:
        assert early.fits(3, listener=False)
        assert early.fits(2, listener=False)
        assert not early.fits(4, listener=False)
        assert not early.fits(1, listener=False)
        # its workers report to no queue
        assert not early.fits(2, listener=True)
    finally:
        early.pending.stop()


def test_a_worker_warms_with_an_experiment_of_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The models of the experiment are in the cache the experiments of the worker take."""
    monkeypatch.setattr(parallel, "_CACHE", OrderedDict())
    runner = _runner(First, simulator=Simulator())
    job = runner._jobs(tmp_path, ["svg"], None, {})["First"]

    _warm(job)

    models = parallel._CACHE[("experiment models", job.engine, job.jit_optimization)]
    assert isinstance(models, dict)
    assert list(models) == list(runner.models)
    # nothing is written, the experiment is not run
    assert not any(tmp_path.iterdir())


def test_the_warm_up_of_a_broken_experiment_is_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The parent reports the experiment, the worker does not report it twice."""
    monkeypatch.setattr(parallel, "_CACHE", OrderedDict())
    runner = _runner(First, simulator=Simulator())
    job = dataclasses.replace(
        runner._jobs(tmp_path, ["svg"], None, {})["First"],
        experiment_class=_BrokenSimulations,
    )
    with caplog.at_level(logging.DEBUG):
        _warm(job)
    assert caplog.records == []
