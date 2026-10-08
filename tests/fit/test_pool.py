"""Tests of the pool of a parallel fit: restarts, timeouts, the preload."""

import multiprocessing
import os
import time
from collections.abc import Iterable
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, override

import numpy as np
import pytest

from sbmlsim import parallel
from sbmlsim.fit import FitSettings, runner
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.runner import run_optimization

_REAL_WORKER_RUN = runner._worker_run
#: the repeats which the workers below break on and hold back
BAD_RUN, HELD_RUN = 1, 0
DEADLINE = 60.0


def _wait_for(marker: Path) -> None:
    """Wait until a marker exists, which a test never does for long."""
    deadline = time.monotonic() + DEADLINE
    while not marker.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"no {marker}")
        time.sleep(0.01)


def die_once_with_company(
    token: str, problem: Any, settings: Any, task: dict[str, Any]
) -> tuple[Any, ...]:
    """Kill a worker once while another repeat runs, the token is a path.

    The held repeat starts and waits for the death, the bad repeat waits for it
    to have started and dies. After the restart both run through.
    """
    base = Path(token)
    died, started = base.with_suffix(".died"), base.with_suffix(".started")
    if task["run"] == HELD_RUN and not died.exists():
        started.touch()
        _wait_for(died)
    if task["run"] == BAD_RUN and not died.exists():
        _wait_for(started)
        died.touch()
        os._exit(1)
    return _REAL_WORKER_RUN(token, problem, settings, task)


def exit_always(
    token: str, problem: Any, settings: Any, task: dict[str, Any]
) -> tuple[Any, ...]:
    """Kill the worker every time the bad repeat runs."""
    if task["run"] == BAD_RUN:
        os._exit(1)
    return _REAL_WORKER_RUN(token, problem, settings, task)


def sleep_always(
    token: str, problem: Any, settings: Any, task: dict[str, Any]
) -> tuple[Any, ...]:
    """Never answer."""
    time.sleep(120)
    raise AssertionError("not reached")


def held_and_bad(token: str, problem: Any, settings: Any, task: dict[str, Any]) -> str:
    """Like `die_once_with_company`, but the bad task dies every time."""
    base = Path(task["dir"]) / "m"
    died, started = base.with_suffix(".died"), base.with_suffix(".started")
    if task["role"] == "held" and not died.exists():
        started.touch()
        _wait_for(died)
    if task["role"] == "bad":
        if not died.exists():
            _wait_for(started)
            died.touch()
        os._exit(1)
    return task["role"]


@pytest.fixture
def fit_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Give the next fit a token which is a path in `tmp_path`."""
    token = str(tmp_path / "fit")
    monkeypatch.setattr(runner, "uuid4", lambda: SimpleNamespace(hex=token))
    return token


def _once(outcomes: Iterable[tuple[int, Any]], tasks: dict[int, Any]) -> dict[int, Any]:
    """Check that every task is handed out exactly once."""
    handed_out = list(outcomes)
    assert sorted(key for key, _ in handed_out) == sorted(tasks)
    return dict(handed_out)


def _fit(op: OptimizationProblem, settings: FitSettings, **kwargs: Any) -> Any:
    arguments: dict[str, Any] = {
        "problem": op,
        "settings": settings,
        "seed": 1234,
        "show_progress": False,
        "max_nfev": 3,
    }
    return run_optimization(**{**arguments, **kwargs})


@pytest.mark.usefixtures("fit_token")
def test_a_worker_which_died_once_loses_no_repeat(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The pool is started again and the repeats which were running run again."""
    monkeypatch.setattr(runner, "_worker_run", die_once_with_company)
    result = _fit(op_hctz_pk, fit_settings, size=3, n_cores=2)
    assert result.size == 3
    assert caplog.text.count("restarting the pool") == 1
    assert not multiprocessing.active_children()


def test_only_the_repeat_which_kills_its_worker_is_given_up(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The others complete, whether they were running at the crash or not."""
    monkeypatch.setattr(runner, "_worker_run", exit_always)
    result = _fit(op_hctz_pk, fit_settings, size=3, n_cores=2)
    assert result.size == 2
    assert f"repeat {BAD_RUN}" in caplog.text
    assert f"repeat {HELD_RUN}" not in caplog.text
    assert not multiprocessing.active_children()


def test_the_tasks_in_flight_at_a_crash_run_alone_again(tmp_path: Path) -> None:
    """The crasher fails when it runs alone, the innocent tasks complete."""
    problem, settings = SimpleNamespace(opid="p", mapping_collections=[]), None
    tasks = {
        0: {"role": "held", "dir": str(tmp_path)},
        1: {"role": "bad", "dir": str(tmp_path)},
        2: {"role": "late", "dir": str(tmp_path)},
    }
    with runner.worker_pool(problem, settings, 2) as pool:  # ty: ignore[invalid-argument-type]
        outcomes = _once(pool.run(held_and_bad, tasks), tasks)
        restarts = pool.restarts
    assert outcomes[0] == "held"
    assert outcomes[2] == "late"
    assert isinstance(outcomes[1], RuntimeError)
    # once for the crash, once more for 1 alone if 2 was still waiting then;
    # which of them was in flight at the crash depends on the timing
    assert restarts in (1, 2)


def test_tasks_which_did_not_run_fail_if_the_pool_cannot_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error of the restart does not lose the results which were handed out."""
    problem, settings = SimpleNamespace(opid="p", mapping_collections=[]), None
    tasks = {
        0: {"role": "held", "dir": str(tmp_path)},
        1: {"role": "bad", "dir": str(tmp_path)},
    }
    real, calls = parallel.start_pool, []

    def start_pool(n_workers: int, preload: Any = ()) -> Any:
        calls.append(n_workers)
        if len(calls) > 1:
            raise RuntimeError("no workers")
        return real(n_workers, preload=preload)

    monkeypatch.setattr(parallel, "start_pool", start_pool)
    with runner.worker_pool(problem, settings, 2) as pool:  # ty: ignore[invalid-argument-type]
        outcomes = _once(pool.run(held_and_bad, tasks), tasks)
    assert str(outcomes[1]) == "no workers"
    # the result of 0 was handed out before the pool broke, or it fails too
    assert str(outcomes[0]) == "no workers" or outcomes[0] == "held"


def test_a_parallel_fit_equals_the_serial_fit(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A seed gives the same repeats, in the same order, for any workers."""
    serial = _fit(op_hctz_pk, fit_settings, size=3, serial=True)
    pooled = _fit(op_hctz_pk, fit_settings, size=3, n_cores=2)
    assert pooled.size == serial.size
    np.testing.assert_allclose(pooled.xopt, serial.xopt, rtol=1e-6)
    for fit_pooled, fit_serial in zip(pooled.fits, serial.fits, strict=True):
        np.testing.assert_allclose(fit_pooled.x, fit_serial.x, rtol=1e-6)


def test_repeats_without_a_result_are_failed_after_the_timeout(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pending repeats are lost after the timeout and the pool is stopped."""
    monkeypatch.setattr(runner, "_worker_run", sleep_always)
    monkeypatch.setattr(runner, "START_GRACE", 0.0)
    with pytest.raises(ValueError, match="every repeat failed") as info:
        _fit(op_hctz_pk, fit_settings, size=2, n_cores=2, timeout=0.0)
    assert "no result within" in str(info.value)
    assert not multiprocessing.active_children()


def test_the_pool_of_a_fit_preloads_the_modules_of_the_problem(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`worker_pool` hands the modules of the problem to `start_pool`."""
    calls: list[tuple[int, Any]] = []
    real = parallel.start_pool

    def start_pool(n_workers: int, preload: Any = ()) -> Any:
        calls.append((n_workers, list(preload)))
        return real(1)

    monkeypatch.setattr(parallel, "start_pool", start_pool)
    with runner.worker_pool(op_hctz_pk, fit_settings, 2):
        pass
    assert calls == [(2, runner._preload(op_hctz_pk))]


class TakenFuture(Future[Any]):
    """A future which knows whether its result was taken."""

    taken = False

    @override
    def result(self, timeout: float | None = None) -> Any:
        self.taken = True
        return super().result(timeout)


class FakeExecutor:
    """An executor without processes, whose tasks behave by their role.

    An `ok` task finishes at once. A `held` task stays pending, unless the
    executor is `forgiving`, then it finishes at once. A `bad` task breaks the
    pool as a dead worker does: every pending future fails with
    `BrokenProcessPool`, the later submits raise it. A `dying` task breaks the
    pool too, but a strict executor notices it only at the next submit, as
    CPython may: that submit fails the pending futures, then raises; a
    forgiving executor notices it at once. `break_at_submit` breaks the pool at
    that submit with nothing running on it.
    """

    _processes: ClassVar[dict[int, Any]] = {}

    def __init__(
        self,
        forgiving: bool = False,
        break_at_submit: int | None = None,
        log: list[int] | None = None,
    ) -> None:
        self.forgiving = forgiving
        self.break_at_submit = break_at_submit
        self.broken = False
        self.dying = False
        self.futures: list[TakenFuture] = []
        #: the tasks in flight after each submit, i.e. the futures of this
        #: executor whose result was not taken yet, the new one included
        self.in_flight: list[int] = log if log is not None else []

    def _break(self) -> None:
        self.broken = True
        for future in self.futures:
            if not future.done():
                future.set_exception(BrokenProcessPool("a worker died"))

    def submit(self, function: Any, *args: Any) -> Future[Any]:
        task = args[-1]
        if self.dying:
            self._break()
        if self.broken or self.break_at_submit == len(self.futures):
            self.broken = True
            raise BrokenProcessPool("broken")
        future = TakenFuture()
        self.futures.append(future)
        self.in_flight.append(sum(not f.taken for f in self.futures))
        role = task["role"]
        if role == "bad" or (role == "dying" and self.forgiving):
            self._break()
        elif role == "dying":
            self.dying = True
        elif role == "ok" or self.forgiving:
            future.set_result(task["key"])
        return future

    def shutdown(self, *args: Any, **kwargs: Any) -> None:
        pass


def _fake_pool(
    monkeypatch: pytest.MonkeyPatch,
    first: FakeExecutor,
    n_cores: int = 2,
    later: Any = None,
) -> tuple[runner.FitPool, list[FakeExecutor]]:
    """Make a `FitPool` on a fake executor, which restarts to forgiving ones."""
    started: list[FakeExecutor] = []

    def start_pool(n_workers: int, preload: Any = ()) -> FakeExecutor:
        executor = later() if later else FakeExecutor(forgiving=True)
        started.append(executor)
        return executor

    monkeypatch.setattr(parallel, "start_pool", start_pool)
    pool = runner.FitPool(
        executor=first,  # ty: ignore[invalid-argument-type]
        token="t",
        problem=SimpleNamespace(opid="p"),  # ty: ignore[invalid-argument-type]
        settings=None,  # ty: ignore[invalid-argument-type]
        n_cores=n_cores,
    )
    return pool, started


def _tasks(*roles: str) -> dict[int, dict[str, Any]]:
    return {k: {"role": role, "key": k} for k, role in enumerate(roles)}


def _run(pool: runner.FitPool, tasks: dict[int, dict[str, Any]]) -> dict[int, Any]:
    return _once(pool.run(lambda *args: None, tasks), tasks)


def test_the_suspects_of_a_crash_run_alone_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The task in flight with the crasher completes, the crasher fails."""
    first, log = FakeExecutor(), []
    pool, started = _fake_pool(
        monkeypatch, first, later=lambda: FakeExecutor(forgiving=True, log=log)
    )
    # 0 is held in the first pool and finishes in the next ones, 1 breaks each
    outcomes = _run(pool, _tasks("held", "bad", "ok"))
    assert outcomes[0] == 0
    assert isinstance(outcomes[1], RuntimeError)
    assert outcomes[2] == 2
    assert pool.restarts == 2
    # 0 and 1 were in flight together in the first pool, then 0 and 1 ran
    # alone in the second, 2 in the third
    assert first.in_flight == [1, 2]
    assert log == [1, 1, 1]
    assert len(started) == 2


def test_the_suspects_fail_without_a_second_run_after_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The running tasks fail, the remaining task still runs in a new pool."""
    pool, started = _fake_pool(monkeypatch, FakeExecutor())
    pool.restarts = runner.MAX_POOL_RESTARTS
    outcomes = _run(pool, _tasks("held", "bad", "ok"))
    assert isinstance(outcomes[0], RuntimeError)
    assert isinstance(outcomes[1], RuntimeError)
    assert "killed its worker" not in str(outcomes[0])
    assert outcomes[2] == 2
    assert pool.restarts == runner.MAX_POOL_RESTARTS + 1
    assert len(started) == 1


def test_a_crash_noticed_at_a_submit_is_found_after_a_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The result handed out before is kept, the crasher runs alone and fails."""
    first = FakeExecutor()
    pool, started = _fake_pool(monkeypatch, first)
    outcomes = _run(pool, _tasks("ok", "dying", "ok"))
    assert outcomes[0] == 0
    assert str(outcomes[1]) == "the task killed its worker"
    assert outcomes[2] == 2
    # 2 was not submitted into the broken pool, it ran in the third one
    assert len(first.futures) == 2
    assert pool.restarts == 2
    assert [len(executor.futures) for executor in started] == [1, 1]


def test_a_submit_into_a_broken_pool_restarts_the_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pool is marked broken between two results: no task is lost."""
    pool, started = _fake_pool(monkeypatch, FakeExecutor(break_at_submit=1), n_cores=1)
    outcomes = _run(pool, _tasks("ok", "ok", "ok"))
    assert outcomes == {0: 0, 1: 1, 2: 2}
    assert pool.restarts == 1
    assert len(started) == 1


def test_a_submit_into_a_broken_pool_keeps_the_results_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task which finished is no suspect of the crash."""
    pool, _started = _fake_pool(monkeypatch, FakeExecutor(break_at_submit=1))
    outcomes = _run(pool, _tasks("ok", "ok", "ok"))
    assert outcomes == {0: 0, 1: 1, 2: 2}
    assert pool.restarts == 1


def test_a_pool_which_is_broken_with_nothing_in_flight_restarts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker which died while idle costs one restart and no task."""
    pool, _started = _fake_pool(monkeypatch, FakeExecutor(break_at_submit=0))
    outcomes = _run(pool, _tasks("ok", "ok"))
    assert outcomes == {0: 0, 1: 1}
    assert pool.restarts == 1


def test_a_pool_which_breaks_after_every_result_runs_every_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only restarts in a row without a result end the run."""

    def breaking() -> FakeExecutor:
        return FakeExecutor(break_at_submit=1)

    pool, _started = _fake_pool(monkeypatch, breaking(), n_cores=1, later=breaking)
    tasks = _tasks(*["ok"] * (runner.MAX_POOL_RESTARTS + 2))
    outcomes = _run(pool, tasks)
    assert outcomes == {k: k for k in tasks}
    assert pool.restarts == runner.MAX_POOL_RESTARTS + 1


def test_a_pool_which_always_breaks_ends_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The restarts are capped, the tasks which did not run fail."""
    pool, _started = _fake_pool(
        monkeypatch,
        FakeExecutor(break_at_submit=0),
        later=lambda: FakeExecutor(break_at_submit=0),
    )
    outcomes = _run(pool, _tasks("ok", "ok"))
    assert all(isinstance(v, RuntimeError) for v in outcomes.values())
    assert pool.restarts == runner.MAX_POOL_RESTARTS
