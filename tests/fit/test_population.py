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
from sbmlsim.fit.optimization import OptimizationProblem, PoolError
from sbmlsim.fit.options import ObjectiveType, OptimizationAlgorithmType
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization, split_cores

DE = OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION
LSQ = OptimizationAlgorithmType.LEAST_SQUARE

_REAL_WORKER_COST = runner._worker_cost
_REAL_WORKER_WARM = runner._worker_warm
#: the calls on which `kill_on_spread_calls` kills its worker, in different
#: generations, more than `MAX_POOL_RESTARTS` of them
SPREAD_KILLS = {3, 15, 27, 39}
#: the calls on which `kill_on_calls_5_and_7` kills its worker: 7 is the first
#: member which runs alone after the crash of 5
ALONE_KILLS = {5, 7}
#: the last member of a population of 2 x 3 parameters
LAST_MEMBER = 5
#: seconds a member sleeps which never answers within a test; the tests of the
#: timeout pass in a fraction of it, also on a slow runner which starts its
#: workers in seconds
NEVER = 600.0


def _claim(token: str) -> int:
    """Claim the number of the next evaluation of a fit, across its workers."""
    n = 1
    while True:
        try:
            os.close(os.open(f"{token}.call.{n}", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            n += 1
            continue
        return n


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


def kill_on_spread_calls(
    token: str, problem: Any, settings: Any, task: dict[str, Any]
) -> Any:
    """Kill the worker on the calls of `SPREAD_KILLS`, the token is a path."""
    if _claim(token) in SPREAD_KILLS:
        os._exit(1)
    return _REAL_WORKER_COST(token, problem, settings, task)


def kill_on_calls_5_and_7(
    token: str, problem: Any, settings: Any, task: dict[str, Any]
) -> Any:
    """Kill the worker on the calls of `ALONE_KILLS`, the token is a path."""
    if _claim(token) in ALONE_KILLS:
        os._exit(1)
    return _REAL_WORKER_COST(token, problem, settings, task)


def kill_the_last_member(
    token: str, problem: Any, settings: Any, task: dict[str, Any]
) -> Any:
    """Kill the worker of the last member of the first generation, every time.

    The first generation is evaluated in the calls up to 8: its six members
    and the two which run again alone after the crash.
    """
    call = _claim(token)
    if task["index"] == LAST_MEMBER and call <= 8:
        os._exit(1)
    return _REAL_WORKER_COST(token, problem, settings, task)


def warm_slowly(token: str, problem: Any, settings: Any, task: dict[str, Any]) -> Any:
    """Initialize the problem in a worker which takes longer than a run may."""
    time.sleep(6.0)
    return _REAL_WORKER_WARM(token, problem, settings, task)


def sleep_from_member_2(
    token: str, problem: Any, settings: Any, task: dict[str, Any]
) -> Any:
    """Evaluate the members 0 and 1, never answer for the others."""
    if task["index"] >= 2:
        time.sleep(NEVER)
    return _REAL_WORKER_COST(token, problem, settings, task)


def sleep_long(token: str, problem: Any, settings: Any, task: dict[str, Any]) -> Any:
    """Never answer within a test."""
    time.sleep(NEVER)
    raise AssertionError("not reached")


def fail_always(token: str, problem: Any, settings: Any, task: dict[str, Any]) -> Any:
    """Fail like a worker which could not initialize the problem."""
    raise RuntimeError("the worker could not initialize the problem: broken")


def _analytic_jacobian(x: np.ndarray) -> np.ndarray:
    """A jacobian of the caller, which is no jacobian of finite differences."""
    raise AssertionError("not called")


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
    "algorithm, size, n_cores, kwargs, n_parameters, expected",
    [
        # one worker is a serial fit
        (DE, 1, 1, {}, 3, (1, 1)),
        (LSQ, 1, 1, {}, 3, (1, 1)),
        # fewer runs than the jacobian of a run is wide: the jacobian of every
        # run on its points, at most one worker per point
        (LSQ, 1, 4, {}, 3, (1, 3)),
        (LSQ, 2, 4, {}, 3, (1, 3)),
        (LSQ, 1, 2, {}, 3, (1, 2)),
        (LSQ, 1, 4, {"jac": "3-point"}, 2, (1, 4)),
        # as many runs as the jacobian is wide or more: a run per worker
        (LSQ, 3, 4, {}, 3, (3, 1)),
        (LSQ, 10, 4, {}, 3, (4, 1)),
        # a jacobian of a single point, or one of the caller, spreads over nothing
        (LSQ, 1, 4, {}, 1, (1, 1)),
        (LSQ, 1, 4, {"jac": _analytic_jacobian}, 3, (1, 1)),
        # fewer runs than workers: the population of every run on all workers
        (DE, 1, 4, {}, 3, (1, 4)),
        (DE, 3, 4, {}, 3, (1, 4)),
        (DE, 3, 4, {"updating": "deferred"}, 3, (1, 4)),
        # as many runs as workers or more: a run per worker
        (DE, 4, 4, {}, 3, (4, 1)),
        (DE, 10, 4, {}, 3, (4, 1)),
        # a population which is updated after every member is evaluated serially
        (DE, 1, 4, {"updating": "immediate"}, 3, (1, 1)),
        (DE, 4, 4, {"updating": "immediate"}, 3, (4, 1)),
        (DE, 1, 4, {"vectorized": True}, 3, (1, 1)),
    ],
)
def test_split_cores(
    algorithm: OptimizationAlgorithmType,
    size: int,
    n_cores: int,
    kwargs: dict[str, Any],
    n_parameters: int,
    expected: tuple[int, int],
) -> None:
    """The workers go to the runs or to the evaluations of a single run."""
    assert (
        split_cores(algorithm, size, n_cores, kwargs, n_parameters=n_parameters)
        == expected
    )


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
    # the run ends at its timeout and not after the members
    assert time.monotonic() - start < NEVER / 5
    fit = result.fits[0]
    assert not fit.success
    assert "FitTimeout" in fit.message
    assert not multiprocessing.active_children()


def test_the_next_run_after_a_timeout_starts_new_workers(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The workers of a run which timed out are stopped, the next run has new ones.

    Without the stop, the second run would wait for the members in flight of
    the first, which sleep `NEVER` s.
    """
    monkeypatch.setattr(runner, "_worker_cost", sleep_long)
    start = time.monotonic()
    result = _de(op_hctz_pk, fit_settings, size=2, n_cores=3, timeout=1.0)
    assert time.monotonic() - start < NEVER / 5
    assert result.size == 2
    for fit in result.fits:
        assert "FitTimeout" in fit.message
    assert not multiprocessing.active_children()


def test_a_timeout_keeps_the_members_which_were_evaluated(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The members which came back before the deadline are in the trajectory."""
    monkeypatch.setattr(runner, "_worker_cost", sleep_from_member_2)
    result = _de(op_hctz_pk, fit_settings, n_cores=2, timeout=5.0)
    fit = result.fits[0]
    assert "FitTimeout" in fit.message
    trajectory = result.trajectories[0]
    assert len(trajectory) == 2
    assert all(np.isfinite(trajectory))
    # the best member which was evaluated is kept
    assert fit.cost == min(trajectory)


def test_the_start_of_the_workers_is_not_charged_to_the_timeout(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The workers initialize the problem before the budget of a run starts."""
    monkeypatch.setattr(runner, "_worker_warm", warm_slowly)
    result = _de(op_hctz_pk, fit_settings, n_cores=2, timeout=5.0, polish=False)
    fit = result.fits[0]
    assert "FitTimeout" not in fit.message
    assert np.isfinite(fit.cost)
    assert len(result.trajectories[0]) == fit.nfev + 1


def test_crashes_spread_over_generations_lose_no_member(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    fit_token: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The cap of the restarts counts per generation, not over the fit.

    Four crashes in different generations, one more than `MAX_POOL_RESTARTS`:
    the members in flight at every crash run again alone, so the fit equals
    the serial fit.
    """
    assert len(SPREAD_KILLS) > runner.MAX_POOL_RESTARTS
    serial = _de(op_hctz_pk, fit_settings, n_cores=1, polish=False, maxiter=6)
    monkeypatch.setattr(runner, "_worker_cost", kill_on_spread_calls)
    pooled = _de(op_hctz_pk, fit_settings, n_cores=2, polish=False, maxiter=6)
    assert _claim(fit_token) > max(SPREAD_KILLS)
    _assert_same_fits(pooled, serial)
    assert caplog.text.count("restarting the pool") == len(SPREAD_KILLS)
    assert "killed its worker" not in caplog.text
    assert not multiprocessing.active_children()


@pytest.mark.usefixtures("fit_token")
def test_a_member_which_kills_its_worker_alone_has_an_infinite_cost(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The member of call 7 kills its worker alone, the run goes on without it.

    The pool is restarted once for the crash of call 5; whichever member was
    the last to run alone, the next generation counts no crash.
    """
    monkeypatch.setattr(runner, "_worker_cost", kill_on_calls_5_and_7)
    pooled = _de(op_hctz_pk, fit_settings, n_cores=2, polish=False)
    trajectory = pooled.trajectories[0]
    assert sum(1 for cost in trajectory if not np.isfinite(cost)) == 1
    assert np.isfinite(pooled.fits[0].cost)
    assert len(trajectory) == pooled.fits[0].nfev + 1
    assert caplog.text.count("restarting the pool") == 1
    assert caplog.text.count("killed its worker, its cost is infinite") == 1
    assert not multiprocessing.active_children()


@pytest.mark.usefixtures("fit_token")
def test_a_last_member_which_kills_its_worker_leaves_no_broken_pool(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The next generation starts new workers and counts no phantom crash."""
    monkeypatch.setattr(runner, "_worker_cost", kill_the_last_member)
    pooled = _de(op_hctz_pk, fit_settings, n_cores=2, polish=False)
    trajectory = pooled.trajectories[0]
    # the last member of the first generation
    assert not np.isfinite(trajectory[LAST_MEMBER])
    assert sum(1 for cost in trajectory if not np.isfinite(cost)) == 1
    assert caplog.text.count("restarting the pool") == 1
    assert "a worker died; restarting the pool." not in caplog.text
    assert not multiprocessing.active_children()


def _fake_pool(*outcomes: Any) -> Any:
    """A pool whose run hands out the given outcomes of the members."""

    def run(function: Any, tasks: Any, deadline: Any = None) -> Any:
        yield from enumerate(outcomes)

    return SimpleNamespace(
        run=run,
        restarts=0,
        stopped=False,
        problem=SimpleNamespace(opid="p"),
        stop=lambda: None,
    )


def test_only_a_member_which_died_alone_has_an_infinite_cost() -> None:
    """A member killed alone is a failed simulation, its cost is infinite."""
    population = runner.PopulationPool(
        _fake_pool(1.0, runner.WorkerDied("the task killed its worker", alone=True))
    )
    assert list(population([np.zeros(1)] * 2, None)) == [(0, 1.0), (1, np.inf)]


def test_members_in_flight_at_the_cap_end_the_run() -> None:
    """Members which failed with others at the cap of the restarts are no result.

    Which of them killed the worker is not known, so the run ends with a
    `PoolError` instead of an infinite cost, which would depend on the timing.
    """
    died = runner.WorkerDied("a worker died while the task was running", alone=False)
    population = runner.PopulationPool(_fake_pool(1.0, died))
    with pytest.raises(PoolError, match="a worker died while the task was running"):
        list(population([np.zeros(1)] * 2, None))


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
    """`workers` of scipy is refused, `n_cores` gives the workers.

    `workers=1`, the default of scipy, is accepted, with one or more workers.
    """
    with pytest.raises(ValueError, match="n_cores"):
        _de(op_hctz_pk, fit_settings, n_cores=1, workers=2)
    serial = _de(op_hctz_pk, fit_settings, n_cores=1, polish=False, workers=1)
    pooled = _de(op_hctz_pk, fit_settings, n_cores=2, polish=False, workers=1)
    _assert_same_fits(pooled, serial)


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
