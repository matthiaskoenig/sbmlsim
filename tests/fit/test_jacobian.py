"""Tests of a least squares run which evaluates its jacobian on workers.

A fit with fewer runs than the jacobian of a run is wide runs its runs one
after another and evaluates the points of the finite differences of every
jacobian on the workers of the fit, see `runner.split_cores`. The points of a
jacobian are evaluated at the same step either way, so the jacobian of the
workers is the jacobian of a serial run bitwise and the fit of a seed does not
depend on the number of workers.
"""

import multiprocessing
import os
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sbmlsim.fit import FitSettings, runner
from sbmlsim.fit.identifiability import (
    IdentifiabilityResult,
    ProfileSettings,
    profile_likelihood,
)
from sbmlsim.fit.optimization import OptimizationProblem, PoolError
from sbmlsim.fit.options import OptimizationAlgorithmType
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization

DE = OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION
LSQ = OptimizationAlgorithmType.LEAST_SQUARE

_REAL_WORKER_RESIDUALS = runner._worker_residuals
_REAL_WORKER_WARM = runner._worker_warm

#: the last point of the jacobian of the reference problem, which has three
#: parameters and a forward difference
LAST_POINT = 2

#: settings of a fast profile of a single parameter, a few large steps with
#: the re-optimization of the other parameters
PROFILE_SETTINGS = ProfileSettings(
    initial_step=0.5, min_step=0.1, max_step=2.0, max_points=3
)

#: seconds a point sleeps which never answers within a test
NEVER = 600.0


def residuals_with_pid(
    token: str, problem: Any, settings: Any, task: dict[str, Any]
) -> Any:
    """Evaluate a point and leave the process id behind, the token is a path."""
    Path(f"{token}.{os.getpid()}.pid").touch()
    return _REAL_WORKER_RESIDUALS(token, problem, settings, task)


def die_once(token: str, problem: Any, settings: Any, task: dict[str, Any]) -> Any:
    """Kill the worker of the first point which is evaluated, once."""
    died = Path(f"{token}.died")
    if not died.exists():
        died.touch()
        os._exit(1)
    return _REAL_WORKER_RESIDUALS(token, problem, settings, task)


def kill_the_last_point(
    token: str, problem: Any, settings: Any, task: dict[str, Any]
) -> Any:
    """Kill the worker of the last point of every jacobian, also alone."""
    if task["index"] == LAST_POINT:
        os._exit(1)
    return _REAL_WORKER_RESIDUALS(token, problem, settings, task)


def sleep_long(token: str, problem: Any, settings: Any, task: dict[str, Any]) -> Any:
    """Never answer within a test."""
    time.sleep(NEVER)
    raise AssertionError("not reached")


def warm_slowly(token: str, problem: Any, settings: Any, task: dict[str, Any]) -> Any:
    """Initialize the problem in a worker which takes longer than a run may."""
    time.sleep(6.0)
    return _REAL_WORKER_WARM(token, problem, settings, task)


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
    """Get the processes which evaluated a point, see `residuals_with_pid`."""
    return {path.name for path in Path(token).parent.glob("fit.*.pid")}


def _lsq(
    op: OptimizationProblem, settings: FitSettings, **kwargs: Any
) -> OptimizationResult:
    """Run a short least squares fit from a sampled start."""
    arguments: dict[str, Any] = {
        "problem": op,
        "settings": settings,
        "algorithm": LSQ,
        "size": 1,
        "seed": 1234,
        "show_progress": False,
        # a few steps of the optimizer, i.e. a few jacobians
        "max_nfev": 3,
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


def test_a_jacobian_on_workers_equals_the_serial_fit(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    fit_token: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One run on two workers takes the path of the serial run of the start."""
    serial = _lsq(op_hctz_pk, fit_settings, n_cores=1)
    monkeypatch.setattr(runner, "_worker_residuals", residuals_with_pid)
    pooled = _lsq(op_hctz_pk, fit_settings, n_cores=2)
    _assert_same_fits(pooled, serial)
    assert len(_processes(fit_token)) == 2
    assert not multiprocessing.active_children()


def test_the_trajectory_has_every_evaluation(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The parent records the cost of every point the workers evaluate.

    The trajectory of a least squares run has the evaluations of the optimizer
    and the points of its jacobians, serially and on workers alike.
    """
    serial = _lsq(op_hctz_pk, fit_settings, n_cores=1)
    pooled = _lsq(op_hctz_pk, fit_settings, n_cores=2)
    fit = pooled.fits[0]
    # the evaluations, the points of the jacobians and the check of the optimum
    assert len(pooled.trajectories[0]) > fit.nfev + 1
    assert pooled.trajectories[0] == serial.trajectories[0]
    # the cost of the optimum is among them; a point of a jacobian may be
    # below it, since the trajectory has every evaluation and not only the
    # steps the optimizer took (on macOS and windows one is, #505)
    assert any(cost == pytest.approx(fit.cost) for cost in pooled.trajectories[0])
    assert min(pooled.trajectories[0]) <= fit.cost


def test_a_serial_fit_starts_no_workers(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """`serial=True` evaluates the jacobian in the calling process."""
    calls: list[int] = []
    real = runner.worker_pool

    def worker_pool(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "worker_pool", worker_pool)
        _lsq(op_hctz_pk, fit_settings, n_cores=4, serial=True)
    assert calls == []


@pytest.mark.skipif(
    (os.process_cpu_count() or 1) < 3, reason="needs three cores for three workers"
)
def test_runs_one_after_another_with_a_jacobian_on_workers(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    fit_token: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two runs on three workers run one after another, on one pool."""
    serial = _lsq(op_hctz_pk, fit_settings, size=2, n_cores=1)
    calls: list[int] = []
    real = runner.worker_pool

    def worker_pool(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "worker_pool", worker_pool)
    monkeypatch.setattr(runner, "_worker_residuals", residuals_with_pid)
    pooled = _lsq(op_hctz_pk, fit_settings, size=2, n_cores=3)
    _assert_same_fits(pooled, serial)
    assert len(_processes(fit_token)) > 1
    assert calls == [1]
    assert not multiprocessing.active_children()


@pytest.mark.usefixtures("fit_token")
def test_a_worker_which_died_loses_no_point(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The pool is started again and the points which were running run again."""
    serial = _lsq(op_hctz_pk, fit_settings, n_cores=1)
    monkeypatch.setattr(runner, "_worker_residuals", die_once)
    pooled = _lsq(op_hctz_pk, fit_settings, n_cores=2)
    _assert_same_fits(pooled, serial)
    assert caplog.text.count("restarting the pool") == 1
    assert not multiprocessing.active_children()


@pytest.mark.usefixtures("fit_token")
def test_a_point_which_kills_its_worker_alone_ends_the_run(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A column of the jacobian cannot be built without its point.

    The run ends with a `PoolError` and keeps the best parameters it reached,
    unlike a member of a population, whose cost is infinite.
    """
    monkeypatch.setattr(runner, "_worker_residuals", kill_the_last_point)
    result = _lsq(op_hctz_pk, fit_settings, n_cores=2)
    fit = result.fits[0]
    assert not fit.success
    assert "cannot be evaluated without it" in fit.message
    # the evaluation at the start values, which the run keeps
    assert np.isfinite(fit.cost)
    assert not multiprocessing.active_children()


def test_a_jacobian_on_workers_keeps_the_timeout(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run ends at its timeout, the workers which still run are stopped."""
    monkeypatch.setattr(runner, "_worker_residuals", sleep_long)
    start = time.monotonic()
    result = _lsq(op_hctz_pk, fit_settings, n_cores=2, timeout=1.0)
    assert time.monotonic() - start < NEVER / 5
    fit = result.fits[0]
    assert not fit.success
    assert "FitTimeout" in fit.message
    assert not multiprocessing.active_children()


def test_the_start_of_the_workers_is_not_charged_to_the_timeout(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The workers initialize the problem before the budget of a run starts."""
    monkeypatch.setattr(runner, "_worker_warm", warm_slowly)
    result = _lsq(op_hctz_pk, fit_settings, n_cores=2, timeout=5.0)
    fit = result.fits[0]
    assert "FitTimeout" not in fit.message
    assert np.isfinite(fit.cost)


def test_a_jacobian_on_workers_which_fail_fails_the_run(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error of a worker ends the run with its message, as it would serially."""
    monkeypatch.setattr(runner, "_worker_residuals", fail_always)
    result = _lsq(op_hctz_pk, fit_settings, n_cores=2)
    fit = result.fits[0]
    assert not fit.success
    assert "could not initialize the problem: broken" in fit.message
    assert not multiprocessing.active_children()


def test_the_workers_of_the_jacobian_belong_to_the_fit(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """`workers` of scipy is refused, `n_cores` gives the workers.

    `workers=1` and `workers=None`, the default of scipy, are accepted.
    """
    with pytest.raises(ValueError, match="n_cores"):
        _lsq(op_hctz_pk, fit_settings, n_cores=1, workers=2)
    serial = _lsq(op_hctz_pk, fit_settings, n_cores=1, workers=1)
    pooled = _lsq(op_hctz_pk, fit_settings, n_cores=2, workers=None)
    _assert_same_fits(pooled, serial)


def test_a_jacobian_evaluator_is_for_least_squares(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A jacobian evaluator of another algorithm, or of a given jacobian, raises."""
    op_hctz_pk.initialize(fit_settings)
    jacobian = runner.JacobianPool(_fake_pool())
    with pytest.raises(ValueError, match="jacobian evaluator is for least squares"):
        op_hctz_pk._optimize_single(algorithm=DE, jacobian=jacobian)
    with pytest.raises(ValueError, match="finite differences"):
        op_hctz_pk._optimize_single(
            algorithm=LSQ, jacobian=jacobian, jac=lambda x: np.zeros((1, 1))
        )


def test_a_point_is_evaluated_without_a_record(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A worker evaluates the residuals of a point, the parent records the cost."""
    op_hctz_pk.initialize(fit_settings)
    xlog = op_hctz_pk.to_scale(
        [0.5 * (p.lower_bound + p.upper_bound) for p in op_hctz_pk.parameters]
    )
    residuals, cost = op_hctz_pk.evaluate_residuals(xlog)
    assert op_hctz_pk._trajectory == []
    assert op_hctz_pk._best is None
    np.testing.assert_array_equal(residuals, op_hctz_pk.residuals(xlog))
    assert op_hctz_pk._trajectory == [cost]
    assert cost == op_hctz_pk.cost_least_square(xlog)


def _fake_pool(*outcomes: Any) -> Any:
    """A pool whose run hands out the given outcomes of the points."""

    def run(function: Any, tasks: Any, deadline: Any = None) -> Any:
        yield from enumerate(outcomes)

    return SimpleNamespace(
        run=run,
        restarts=0,
        stopped=False,
        problem=SimpleNamespace(opid="p"),
        stop=lambda: None,
        start=lambda: None,
        executor=None,
        n_cores=1,
    )


def test_a_point_which_died_alone_ends_the_jacobian() -> None:
    """A point killed alone ends the run, its column cannot be built."""
    jacobian = runner.JacobianPool(
        _fake_pool(
            (np.zeros(2), 1.0),
            runner.WorkerDied("the task killed its worker", alone=True),
        )
    )
    with pytest.raises(PoolError, match="cannot be evaluated without it"):
        list(jacobian([np.zeros(1)] * 2, None))


def test_points_in_flight_at_the_cap_end_the_run() -> None:
    """Points which failed with others at the cap of the restarts are no result."""
    died = runner.WorkerDied("a worker died while the task was running", alone=False)
    jacobian = runner.JacobianPool(_fake_pool((np.zeros(2), 1.0), died))
    with pytest.raises(PoolError, match="a worker died while the task was running"):
        list(jacobian([np.zeros(1)] * 2, None))


def test_the_jacobian_map_embeds_a_point(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A point of some of the parameters is embedded before it is evaluated.

    The re-optimization of a scan of a profile optimizes the parameters which
    are not fixed, so the points of its jacobian are shorter than the
    parameter vector of the problem, see `profile_likelihood`.
    """
    op_hctz_pk.initialize(fit_settings)
    theta = op_hctz_pk.to_scale(
        [0.5 * (p.lower_bound + p.upper_bound) for p in op_hctz_pk.parameters]
    )
    seen: list[np.ndarray] = []

    class Evaluator:
        """Hands out the index of every point with a residual vector of it."""

        def start(self) -> None:
            """Nothing to start."""

        def __call__(
            self, points: Sequence[np.ndarray], deadline: float | None
        ) -> Iterator[tuple[int, tuple[np.ndarray, float]]]:
            """Give every point its index as its residuals and its cost."""
            seen.extend(points)
            for k in range(len(points)):
                yield k, (np.zeros(2) + k, float(k))

    def embed(free: np.ndarray) -> np.ndarray:
        full = theta.copy()
        full[1:] = free
        return full

    evaluate = op_hctz_pk.jacobian_map(Evaluator(), embed=embed)
    values = evaluate(None, [theta[1:], theta[1:] + 0.5])
    assert [point.tolist() for point in seen] == [
        embed(theta[1:]).tolist(),
        embed(theta[1:] + 0.5).tolist(),
    ]
    # the residuals in the order of the points, the costs recorded in it
    np.testing.assert_array_equal(values[0], np.zeros(2))
    np.testing.assert_array_equal(values[1], np.ones(2))
    assert op_hctz_pk._trajectory == [0.0, 1.0]


# ---------------------------------------------------------------------------
# the jacobian of the re-optimizations of a profile
# ---------------------------------------------------------------------------
def _profile_of(
    problem: OptimizationProblem, settings: FitSettings, **kwargs: Any
) -> IdentifiabilityResult:
    """Profile the first parameter of the problem with re-optimization."""
    problem.initialize(settings)
    return profile_likelihood(
        problem=problem,
        settings=settings,
        parameter_set=problem.parameter_set_model(),
        profile_settings=PROFILE_SETTINGS,
        pids=[problem.pids[0]],
        show_progress=False,
        **kwargs,
    )


def test_a_profile_splits_its_cores_between_scans_and_the_jacobian(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scans of one parameter are two, the jacobian is the free parameters."""
    recorded: dict[str, Any] = {}
    real = runner.split_cores

    def split_cores(*args: Any, **kwargs: Any) -> tuple[int, int]:
        recorded.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "split_cores", split_cores)
    _profile_of(op_hctz_pk, fit_settings, n_cores=1, serial=True)
    assert recorded["size"] == 2
    # one parameter is fixed, the others are re-optimized
    assert recorded["n_parameters"] == len(op_hctz_pk.parameters) - 1


def test_a_profile_with_the_jacobian_on_workers_equals_the_serial_profile(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scans run here and their re-optimizations on the workers.

    The reference problem has three parameters, so the jacobian of a
    re-optimization is two points wide and `split_cores` gives the two scans
    to the workers; the split is forced here to run the scans one after
    another with the jacobian on two workers.
    """
    serial = _profile_of(op_hctz_pk, fit_settings, n_cores=1, serial=True)
    monkeypatch.setattr(runner, "split_cores", lambda *args, **kwargs: (1, 2))
    pooled = _profile_of(op_hctz_pk, fit_settings, n_cores=2)
    pid = op_hctz_pk.pids[0]
    np.testing.assert_array_equal(
        pooled.profiles[pid].values, serial.profiles[pid].values
    )
    np.testing.assert_array_equal(
        pooled.profiles[pid].costs, serial.profiles[pid].costs
    )
    np.testing.assert_array_equal(
        pooled.profiles[pid].paths, serial.profiles[pid].paths
    )
    assert not multiprocessing.active_children()
