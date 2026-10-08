"""Tests of the pool of a parallel fit: restarts, timeouts, the preload."""

import multiprocessing
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
        outcomes = dict(pool.run(held_and_bad, tasks))
        restarts = pool.restarts
    assert outcomes[0] == "held"
    assert outcomes[2] == "late"
    assert isinstance(outcomes[1], RuntimeError)
    assert sorted(outcomes) == [0, 1, 2]
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
        outcomes = dict(pool.run(held_and_bad, tasks))
    assert str(outcomes[1]) == "no workers"
    # the result of 0 was handed out before the pool broke, or it fails too
    assert str(outcomes[0]) == "no workers" or outcomes[0] == "held"
    assert sorted(outcomes) == [0, 1]


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
