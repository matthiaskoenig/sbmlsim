"""Test that a fit survives runs which fail or run out of time."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sbmlsim.fit import FitSettings, runner
from sbmlsim.fit.optimization import (
    FitTimeout,
    OptimizationProblem,
    RuntimeErrorOptimizeResult,
)
from sbmlsim.fit.options import OptimizationAlgorithmType
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization


def test_failed_result_is_an_optimize_result() -> None:
    """A failed run is a scipy result, so it is processed like a good one."""
    result = RuntimeErrorOptimizeResult(
        x=np.array([1.0]), x0=np.array([2.0]), cost=5.0, message="boom"
    )
    # attribute access, as the processing of the fits uses
    assert result.success is False
    assert result.cost == 5.0
    assert result.message == "boom"
    # and dictionary access, as the serialization uses
    assert np.allclose(result.get("x"), [1.0])
    assert np.allclose(result["x0"], [2.0])


def test_a_failing_run_keeps_the_other_runs(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One optimization which raises does not lose the others."""
    op = op_hctz_pk
    op.initialize(fit_settings)
    original = op._optimize_single
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("the first run explodes")
        return original(*args, **kwargs)

    monkeypatch.setattr(op, "_optimize_single", flaky)
    fits, trajectories = op.optimize(size=3, seed=1234)

    assert len(fits) == 3
    assert len(trajectories) == 3
    # the run which raised is reported, the others are fine
    assert [fit.success for fit in fits].count(False) >= 1
    assert any(fit.success for fit in fits)
    failed = next(fit for fit in fits if not fit.success)
    assert "the first run explodes" in failed.message


def test_timeout_keeps_what_a_run_reached(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A run which is out of time contributes the best point it found."""
    opt_result = run_optimization(
        problem=op_hctz_pk,
        settings=fit_settings,
        size=2,
        n_cores=1,
        serial=True,
        seed=1234,
        timeout=1e-9,
        show_progress=False,
    )
    assert opt_result.size == 2
    assert not any(fit.success for fit in opt_result.fits)
    for fit in opt_result.fits:
        assert "FitTimeout" in fit.message
        # the parameters it reached, not nothing
        assert fit.x is not None
        assert np.all(np.isfinite(fit.x))


def test_timeout_does_not_outlive_the_run(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The deadline of a run is over when it is, the report evaluates again."""
    run_optimization(
        problem=op_hctz_pk,
        settings=fit_settings,
        size=1,
        n_cores=1,
        serial=True,
        seed=1234,
        timeout=1e-9,
        show_progress=False,
    )
    # the residuals are evaluated for the report, this must not time out
    residuals = op_hctz_pk.residuals(np.log10(op_hctz_pk.xmodel))
    assert isinstance(residuals, np.ndarray)
    assert np.all(np.isfinite(residuals))


def test_fit_timeout_is_raised_by_the_objective(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The objective is what ends a run which is out of its budget."""
    op = op_hctz_pk
    op.initialize(fit_settings)
    op._deadline = 0.0
    with pytest.raises(FitTimeout):
        op.residuals(np.log10(op.xmodel))
    op._deadline = None


def test_runs_are_stored_while_the_fit_runs(
    tmp_path: Path, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Every run is a file, so an interrupted fit leaves what it finished."""
    runs_dir = tmp_path / "runs"
    opt_result = run_optimization(
        problem=op_hctz_pk,
        settings=fit_settings,
        size=2,
        n_cores=1,
        serial=True,
        seed=1234,
        runs_dir=runs_dir,
        show_progress=False,
    )
    assert len(list(runs_dir.glob("*.json"))) == 2

    # the runs alone recover the result of the fit
    recovered = OptimizationResult.from_directory(runs_dir)
    assert recovered.size == opt_result.size
    assert recovered.settings == fit_settings
    assert recovered.df_fits.cost.iloc[0] == pytest.approx(
        opt_result.df_fits.cost.iloc[0]
    )


def test_from_directory_without_runs(tmp_path: Path) -> None:
    """An empty directory of runs is reported."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No optimization run"):
        OptimizationResult.from_directory(tmp_path / "empty")


def test_run_result(op_hctz_pk: OptimizationProblem, fit_settings: FitSettings) -> None:
    """A single run of a result is a result of its own."""
    opt_result = run_optimization(
        problem=op_hctz_pk,
        settings=fit_settings,
        size=2,
        n_cores=1,
        serial=True,
        seed=1234,
        show_progress=False,
    )
    run = opt_result.run_result(0)
    assert run.size == 1
    assert run.opid == opt_result.opid
    assert run.settings == opt_result.settings
    assert run.sid.endswith("_0")
    # the runs are indexed in the order they ran, not by their cost
    assert np.allclose(run.xopt, opt_result.fits[0].x)
    assert run.trajectories[0] == opt_result.trajectories[0]

    # the parameter sets are ordered by cost instead
    best = opt_result.parameter_set(0)
    assert best.cost == pytest.approx(opt_result.df_fits.cost.iloc[0])


def test_mappings_are_grouped_by_simulation(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The fit mappings which share a simulation are simulated together."""
    op = op_hctz_pk
    op.initialize(fit_settings)

    # every mapping is in exactly one group
    grouped = [k for group in op.mapping_groups for k in group]
    assert sorted(grouped) == list(range(len(op.mapping_keys)))

    # and the mappings of this problem share their simulations
    assert len(op.mapping_groups) < len(op.mapping_keys)
    for group in op.mapping_groups:
        simulations = {id(op.simulations[k]) for k in group}
        assert len(simulations) == 1


def test_differential_evolution_without_convergence(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A global fit which does not converge still reports its cost.

    `differential_evolution` reports `fun`, not `cost`, and a run which only
    reached its iteration limit is not a failure.
    """
    from sbmlsim.fit.options import OptimizationAlgorithmType

    opt_result = run_optimization(
        problem=op_hctz_pk,
        settings=fit_settings,
        algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
        size=1,
        n_cores=1,
        serial=True,
        seed=1234,
        maxiter=1,
        show_progress=False,
    )
    assert opt_result.size == 1
    assert np.isfinite(opt_result.df_fits.cost.iloc[0])


def test_start_values_do_not_depend_on_the_workers(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The runner samples the start points, not the workers."""
    op_hctz_pk.initialize(fit_settings)
    starts = op_hctz_pk.start_values(size=6, seed=1234)
    again = op_hctz_pk.start_values(size=6, seed=1234)

    assert len(starts) == 6
    for x0, x0_again in zip(starts, again, strict=True):
        assert x0 is not None
        assert x0_again is not None
        assert np.allclose(x0, x0_again)
    # and the start points of the runs differ from each other
    assert len({tuple(np.asarray(x0)) for x0 in starts}) == 6


def test_every_global_run_gets_its_own_seed() -> None:
    """One seed for all runs of the global optimizer gives one result."""
    seeds = OptimizationProblem.run_seeds(
        size=4,
        algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
        seed=1234,
    )
    assert len(seeds) == 4
    assert len(set(seeds)) == 4

    # the local optimizer differs in its start values and needs no seed
    assert (
        OptimizationProblem.run_seeds(
            size=4, algorithm=OptimizationAlgorithmType.LEAST_SQUARE, seed=1234
        )
        == [None] * 4
    )


def test_a_run_of_a_worker_never_raises(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A repeat which fails is a result, so the pool keeps the other repeats."""
    op = op_hctz_pk
    op.initialize(fit_settings)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("the run explodes")

    op._optimize_single = boom
    fit, trajectory = op.optimize_run(x0=np.asarray(op.x0, dtype=float), run=2)
    assert fit.success is False
    assert "the run explodes" in fit.message
    assert trajectory == []


def test_a_worker_without_a_problem_reports_it() -> None:
    """A worker which could not initialize reports it for every repeat."""
    runner._WORKER_PROBLEM = None
    runner._WORKER_ERROR = "ValueError: no data"
    run, fit, trajectory = runner._worker_run({"run": 3, "x0": None})

    assert run == 3
    assert fit.success is False
    assert "no data" in fit.message
    assert trajectory == []
