"""Tests of the pool of a parallel fit: restarts, timeouts, the preload."""

import multiprocessing
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sbmlsim import parallel
from sbmlsim.fit import FitSettings, runner
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.runner import run_optimization

_REAL_WORKER_RUN = runner._worker_run
#: the repeat which the workers below break on
BAD_RUN = 1


def exit_once(
    token: str, problem: Any, settings: Any, task: dict[str, Any]
) -> tuple[Any, ...]:
    """Kill the worker the first time the bad repeat runs, in any worker."""
    marker = Path(tempfile.gettempdir()) / f"sbmlsim-test-exit-{token}"
    if task["run"] == BAD_RUN and not marker.exists():
        marker.touch()
        os._exit(1)
    return _REAL_WORKER_RUN(token, problem, settings, task)


def exit_always(
    token: str, problem: Any, settings: Any, task: dict[str, Any]
) -> tuple[Any, ...]:
    """Kill the worker every time the bad repeat runs."""
    if task["run"] == BAD_RUN:
        # the other repeats finish while this one runs
        time.sleep(2.0)
        os._exit(1)
    return _REAL_WORKER_RUN(token, problem, settings, task)


def sleep_always(
    token: str, problem: Any, settings: Any, task: dict[str, Any]
) -> tuple[Any, ...]:
    """Never answer."""
    time.sleep(120)
    raise AssertionError("not reached")


@pytest.fixture(autouse=True)
def _clean_markers() -> Any:
    """Remove the markers of `exit_once`."""
    yield
    for marker in Path(tempfile.gettempdir()).glob("sbmlsim-test-exit-*"):
        marker.unlink(missing_ok=True)


def _fit(op: OptimizationProblem, settings: FitSettings, **kwargs: Any) -> Any:
    arguments: dict[str, Any] = {
        "problem": op,
        "settings": settings,
        "seed": 1234,
        "show_progress": False,
        "max_nfev": 3,
    }
    return run_optimization(**{**arguments, **kwargs})


def test_a_worker_which_died_once_loses_no_repeat(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pool is started again and the unfinished repeats run again."""
    monkeypatch.setattr(runner, "_worker_run", exit_once)
    result = _fit(op_hctz_pk, fit_settings, size=3, n_cores=2)
    assert result.size == 3
    assert not multiprocessing.active_children()


def test_a_repeat_which_always_kills_its_worker_is_given_up(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fit ends after the restarts with the repeats which finished."""
    monkeypatch.setattr(runner, "_worker_run", exit_always)
    result = _fit(op_hctz_pk, fit_settings, size=3, n_cores=2)
    assert result.size == 2
    assert "restarting the pool" in caplog.text
    assert caplog.text.count("restarting the pool") == runner.MAX_POOL_RESTARTS
    assert f"repeat {BAD_RUN}" in caplog.text
    assert not multiprocessing.active_children()


def test_a_parallel_fit_equals_the_serial_fit(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A seed gives the same repeats, in the same order, for any workers."""
    serial = _fit(op_hctz_pk, fit_settings, size=3, serial=True)
    pooled = _fit(op_hctz_pk, fit_settings, size=3, n_cores=2)
    assert pooled.size == serial.size
    np.testing.assert_allclose(pooled.xopt, serial.xopt, rtol=1e-6)


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
