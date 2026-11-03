"""Tests of differential evolution which evaluates its population on workers.

A fit of differential evolution with fewer runs than workers runs its runs one
after another and evaluates the members of the population of a run on the
workers of the fit, see `runner.split_cores`. The result of a seed does not
depend on the number of workers.
"""

import multiprocessing
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sbmlsim.fit import FitSettings, runner
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ObjectiveType, OptimizationAlgorithmType
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization, split_cores

DE = OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION
LSQ = OptimizationAlgorithmType.LEAST_SQUARE

_REAL_WORKER_COST = runner._worker_cost


def cost_with_pid(token: str, problem: Any, settings: Any, task: dict[str, Any]) -> Any:
    """Evaluate a member and leave the process id behind, the token is a path."""
    Path(f"{token}.{os.getpid()}.pid").touch()
    return _REAL_WORKER_COST(token, problem, settings, task)


def die_once(token: str, problem: Any, settings: Any, task: dict[str, Any]) -> Any:
    """Kill the worker of the first member which is evaluated, once."""
    died = Path(f"{token}.died")
    if not died.exists():
        died.touch()
        os._exit(1)
    return _REAL_WORKER_COST(token, problem, settings, task)


def sleep_long(token: str, problem: Any, settings: Any, task: dict[str, Any]) -> Any:
    """Never answer within a test."""
    time.sleep(120)
    raise AssertionError("not reached")


def fail_always(token: str, problem: Any, settings: Any, task: dict[str, Any]) -> Any:
    """Fail like a worker which could not initialize the problem."""
    raise RuntimeError("the worker could not initialize the problem: broken")


@pytest.fixture
def fit_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Give the next fit a token which is a path in `tmp_path`."""
    token = str(tmp_path / "fit")
    monkeypatch.setattr(runner, "uuid4", lambda: SimpleNamespace(hex=token))
    return token


def _processes(token: str) -> set[str]:
    """Get the processes which evaluated a member, see `cost_with_pid`."""
    return {path.name for path in Path(token).parent.glob("fit.*.pid")}


def _de(
    op: OptimizationProblem, settings: FitSettings, **kwargs: Any
) -> OptimizationResult:
    """Run a short fit of differential evolution with a seed."""
    arguments: dict[str, Any] = {
        "problem": op,
        "settings": settings,
        "algorithm": DE,
        "size": 1,
        "seed": 1234,
        "show_progress": False,
        # the smallest population (2 x 3 parameters) and few generations
        "maxiter": 3,
        "popsize": 2,
    }
    return run_optimization(**{**arguments, **kwargs})


def _assert_same_fits(pooled: OptimizationResult, serial: OptimizationResult) -> None:
    """Check that two fits found the same optimum on the same path."""
    assert pooled.size == serial.size
    for fit_pooled, fit_serial in zip(pooled.fits, serial.fits, strict=True):
        np.testing.assert_array_equal(fit_pooled.x, fit_serial.x)
        assert fit_pooled.cost == fit_serial.cost
        assert fit_pooled.nfev == fit_serial.nfev
    assert pooled.trajectories == serial.trajectories


@pytest.mark.parametrize(
    "algorithm, size, n_cores, kwargs, expected",
    [
        # one worker is a serial fit
        (DE, 1, 1, {}, (1, 1)),
        (LSQ, 1, 1, {}, (1, 1)),
        # least squares runs its runs in parallel, never a population
        (LSQ, 1, 4, {}, (1, 1)),
        (LSQ, 2, 4, {}, (2, 1)),
        (LSQ, 10, 4, {}, (4, 1)),
        # fewer runs than workers: the population of every run on all workers
        (DE, 1, 4, {}, (1, 4)),
        (DE, 3, 4, {}, (1, 4)),
        (DE, 3, 4, {"updating": "deferred"}, (1, 4)),
        # as many runs as workers or more: a run per worker
        (DE, 4, 4, {}, (4, 1)),
        (DE, 10, 4, {}, (4, 1)),
        # a population which is updated after every member is evaluated serially
        (DE, 1, 4, {"updating": "immediate"}, (1, 1)),
        (DE, 4, 4, {"updating": "immediate"}, (4, 1)),
        (DE, 1, 4, {"vectorized": True}, (1, 1)),
    ],
)
def test_split_cores(
    algorithm: OptimizationAlgorithmType,
    size: int,
    n_cores: int,
    kwargs: dict[str, Any],
    expected: tuple[int, int],
) -> None:
    """The workers go to the runs or to the population of differential evolution."""
    assert split_cores(algorithm, size, n_cores, kwargs) == expected


def test_a_population_on_workers_equals_the_serial_population(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    fit_token: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One run on two workers finds the optimum of the serial run of the seed.

    The polishing of the best member by L-BFGS-B evaluates on the workers too.
    """
    serial = _de(op_hctz_pk, fit_settings, n_cores=1)
    monkeypatch.setattr(runner, "_worker_cost", cost_with_pid)
    pooled = _de(op_hctz_pk, fit_settings, n_cores=2)
    _assert_same_fits(pooled, serial)
    assert len(_processes(fit_token)) == 2
    assert not multiprocessing.active_children()


def test_a_likelihood_on_workers_equals_the_serial_likelihood(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The cost of a worker is the negative log-likelihood with its offsets."""
    settings = replace(fit_settings, objective=ObjectiveType.LIKELIHOOD)
    serial = _de(op_hctz_pk, settings, n_cores=1, polish=False)
    pooled = _de(op_hctz_pk, settings, n_cores=2, polish=False)
    _assert_same_fits(pooled, serial)
    assert np.isfinite(pooled.fits[0].cost)


def test_the_trajectory_has_every_evaluation(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The parent records the cost of every member the workers evaluate.

    The trajectory has the evaluations of the optimizer, `nfev`, and the check
    of the optimum, see `OptimizationProblem._validate_optimum`.
    """
    for n_cores in (1, 2):
        result = _de(op_hctz_pk, fit_settings, n_cores=n_cores, polish=False)
        fit = result.fits[0]
        assert len(result.trajectories[0]) == fit.nfev + 1
        assert min(result.trajectories[0]) == pytest.approx(fit.cost)


@pytest.mark.skipif(
    (os.process_cpu_count() or 1) < 3, reason="needs three cores for three workers"
)
def test_runs_one_after_another_with_a_population_on_workers(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    fit_token: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two runs on three workers run one after another, on one pool."""
    serial = _de(op_hctz_pk, fit_settings, size=2, n_cores=1, polish=False)
    calls: list[int] = []
    real = runner.worker_pool

    def worker_pool(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "worker_pool", worker_pool)
    monkeypatch.setattr(runner, "_worker_cost", cost_with_pid)
    pooled = _de(op_hctz_pk, fit_settings, size=2, n_cores=3, polish=False)
    _assert_same_fits(pooled, serial)
    assert len(_processes(fit_token)) > 1
    assert calls == [1]
    assert not multiprocessing.active_children()


def test_runs_on_workers_equal_the_serial_runs(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Runs of differential evolution as tasks of the pool equal the serial runs.

    A run evaluates its population serially in its worker, which gives the
    same result as the population on workers, both update it once per
    generation.
    """
    serial = _de(op_hctz_pk, fit_settings, size=2, n_cores=1, polish=False)
    pooled = _de(op_hctz_pk, fit_settings, size=2, n_cores=2, polish=False)
    _assert_same_fits(pooled, serial)


@pytest.mark.usefixtures("fit_token")
def test_a_worker_which_died_loses_no_member(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The pool is started again and the members which were running run again."""
    serial = _de(op_hctz_pk, fit_settings, n_cores=1, polish=False)
    monkeypatch.setattr(runner, "_worker_cost", die_once)
    pooled = _de(op_hctz_pk, fit_settings, n_cores=2, polish=False)
    _assert_same_fits(pooled, serial)
    assert caplog.text.count("restarting the pool") == 1
    assert not multiprocessing.active_children()


def test_a_population_on_workers_keeps_the_timeout(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run ends at its timeout, the workers which still run are stopped."""
    monkeypatch.setattr(runner, "_worker_cost", sleep_long)
    start = time.monotonic()
    result = _de(op_hctz_pk, fit_settings, n_cores=2, timeout=1.0)
    assert time.monotonic() - start < 60.0
    fit = result.fits[0]
    assert not fit.success
    assert "FitTimeout" in fit.message
    assert not multiprocessing.active_children()


def test_a_population_on_workers_which_fail_fails_the_run(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error of a worker ends the run with its message, as it would serially."""
    monkeypatch.setattr(runner, "_worker_cost", fail_always)
    result = _de(op_hctz_pk, fit_settings, n_cores=2)
    fit = result.fits[0]
    assert not fit.success
    assert "could not initialize the problem: broken" in fit.message
    assert not multiprocessing.active_children()


def test_the_workers_of_the_population_belong_to_the_fit(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """`workers` of scipy is refused, `n_cores` gives the workers."""
    with pytest.raises(ValueError, match="n_cores"):
        _de(op_hctz_pk, fit_settings, n_cores=1, workers=2)


def test_a_member_is_evaluated_without_a_record(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A worker evaluates the cost of a member, the parent records it."""
    op_hctz_pk.initialize(fit_settings)
    xlog = op_hctz_pk.to_scale(
        [0.5 * (p.lower_bound + p.upper_bound) for p in op_hctz_pk.parameters]
    )
    cost = op_hctz_pk.evaluate_cost(xlog)
    assert op_hctz_pk._trajectory == []
    assert op_hctz_pk._best is None
    assert cost == op_hctz_pk.cost_least_square(xlog)
    assert op_hctz_pk._trajectory == [cost]
