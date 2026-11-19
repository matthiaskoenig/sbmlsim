"""The request of a fit of the engine core, built from an optimization problem (#556).

The engine fits a problem itself: its optimizer, the runs of its starts and
the profiles of its identifiability run in rust on the threads of one batch,
where an optimizer in python evaluates one point at a time and waits for the
engine between the steps. A problem reaches that path when its whole
objective is in a request of the contract `fit/1`, see `sbmlsim.fit.fit_core`.

The request is checked against the problem itself: the residuals the engine
computes from the request are the ones the problem computes for the same
parameters, every point of them, which is what makes the fit of the engine
the fit of the problem. The iv data of the HCTZ example is the problem of
these tests, four training mappings of one simulation, with the weighting of
a real fit (`fit_settings`: normalized residuals, the points of the curves
and the errors of the data).
"""

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sbmlsim.engine import Engine
from sbmlsim.fit import FitSettings, fit_core
from sbmlsim.fit.fit_core import (
    CONVERGED,
    fit_request,
    fits_requests,
    gaps_of,
    gaps_of_the_profiles,
    loaded_model,
    run,
    walk_of,
)
from sbmlsim.fit.identifiability import ProfileSettings, profile_likelihood
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import LossFunctionType, ObjectiveType, ResidualType
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization
from sbmlsim.fit.sampling import SamplingType
from sbmlsim.simulator import Simulator
from tests.engines import require_core_fit


def on_core(problem: OptimizationProblem, settings: FitSettings) -> OptimizationProblem:
    """Initialize a problem on the engine core, the engine which fits itself."""
    require_core_fit()
    problem.initialize(settings)
    problem.set_simulator(Simulator(n_workers=1, engine=Engine.CORE))
    assert problem.engine is Engine.CORE
    return problem


def core_problem(problem: OptimizationProblem) -> Any:
    """Get the objective of the engine for the request of a problem."""
    return loaded_model(problem).fit_problem(fit_request(problem), threads=1)


def test_the_request_of_a_problem_is_its_simulations_and_its_training_data(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A condition is a simulation group of the problem, a curve a training mapping."""
    problem = on_core(op_hctz_iv, fit_settings)

    request = fit_request(problem)

    assert request["contract"] == "fit/1"
    assert len(request["conditions"]) == len(problem.mapping_groups)
    assert [p["id"] for p in request["parameters"]] == [
        parameter.target_id for parameter in problem.parameters
    ]
    training = problem.indices(MappingKind.TRAINING)
    assert len(request["curves"]) == len(training)
    assert [curve["id"] for curve in request["curves"]] == [
        problem.mapping_keys[k] for k in training
    ]
    conditions = {condition["id"] for condition in request["conditions"]}
    for curve in request["curves"]:
        assert curve["condition"] in conditions
        assert len(curve["times"]) == len(curve["values"]) == len(curve["noise"])
        assert all(noise > 0.0 for noise in curve["noise"])


def test_the_residuals_of_the_engine_are_the_ones_of_the_problem(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The engine computes the residuals of the problem, point for point."""
    problem = on_core(op_hctz_iv, fit_settings)
    core = core_problem(problem)

    for factor in (1.0, 1.3, 0.7):
        x = problem.to_scale(problem.xmodel * factor)
        expected = np.asarray(problem.residuals(x), dtype=float)

        residuals = np.asarray(core.residuals(x), dtype=float)

        assert residuals.shape == expected.shape
        np.testing.assert_allclose(residuals, expected, rtol=1e-9, atol=1e-12)
        assert core.cost(x) == pytest.approx(
            0.5 * float(np.sum(np.square(expected))), rel=1e-9
        )


def test_the_start_and_the_bounds_of_the_request_are_the_ones_of_the_parameters(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The search space of the engine is the search space of the problem."""
    problem = on_core(op_hctz_iv, fit_settings)
    core = core_problem(problem)

    np.testing.assert_allclose(core.start, problem.to_scale(problem.xmodel), rtol=1e-12)
    lower, upper = core.bounds
    np.testing.assert_allclose(
        lower,
        problem.to_scale([p.lower_bound for p in problem.parameters]),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        upper,
        problem.to_scale([p.upper_bound for p in problem.parameters]),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"objective": ObjectiveType.LIKELIHOOD}, "objective"),
        ({"loss_function": LossFunctionType.SOFT_L1}, "loss_function"),
        ({"residual": ResidualType.ABSOLUTE_TO_BASELINE}, "residual"),
    ],
)
def test_an_objective_the_engine_has_not_is_a_gap_of_the_problem(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    change: dict[str, object],
    code: str,
) -> None:
    """A problem whose objective the request cannot carry names what keeps it out."""
    problem = on_core(op_hctz_iv, dataclasses.replace(fit_settings, **change))

    gaps = gaps_of(problem)

    assert [gap.code for gap in gaps] == [code]
    assert gaps[0].message


def test_a_problem_of_another_engine_is_no_fit_of_the_engine(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Only a problem which simulates in the engine core fits in it."""
    op_hctz_iv.initialize(fit_settings)
    assert op_hctz_iv.engine is Engine.ROADRUNNER

    gaps = gaps_of(op_hctz_iv)

    assert [gap.code for gap in gaps] == ["engine"]


def test_a_problem_without_gaps_builds_a_request(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The gaps of the problem of a fit of the engine are none."""
    problem = on_core(op_hctz_iv, fit_settings)

    assert gaps_of(problem) == []


def test_the_cost_of_the_optimum_of_the_engine_is_the_cost_of_the_problem(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The engine minimizes the objective of the problem and reports its cost."""
    problem = on_core(op_hctz_iv, fit_settings)
    (start,) = problem.start_values(1, sampling=SamplingType.START)
    assert start is not None
    residuals = np.asarray(problem.residuals(problem.to_scale(start)), dtype=float)
    cost_of_the_start = 0.5 * float(np.sum(np.square(residuals)))

    result = run(
        problem,
        fit_settings,
        size=1,
        sampling=SamplingType.START,
        max_iterations=10,
        threads=2,
    )

    assert len(result.fits) == 1
    fit = result.fits[0]
    assert fit.cost < cost_of_the_start
    assert fit.duration > 0.0
    assert isinstance(fit.message, str) and fit.message
    optimum = np.asarray(problem.residuals(problem.to_scale(fit.x)), dtype=float)
    assert fit.cost == pytest.approx(0.5 * float(np.sum(np.square(optimum))), rel=1e-9)
    np.testing.assert_allclose(fit.x0, np.asarray(start, dtype=float), rtol=1e-12)


def test_a_fit_of_the_engine_of_many_starts_is_a_run_per_start(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Every start of a multi start fit is a run of the result, the best of them first."""
    problem = on_core(op_hctz_iv, fit_settings)

    result = run(problem, fit_settings, size=3, seed=1234, max_iterations=2, threads=2)

    assert len(result.fits) == 3
    starts = problem.start_values(3, seed=1234)
    for fit, start in zip(result.fits, starts, strict=True):
        np.testing.assert_allclose(fit.x0, np.asarray(start, dtype=float), rtol=1e-12)
    costs = list(result.df_fits["cost"])
    assert costs == sorted(costs)
    assert len(result.df_traces) == 0


def test_the_result_of_a_fit_of_the_engine_is_the_result_of_a_fit(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings, tmp_path: Path
) -> None:
    """The result of the engine is an `OptimizationResult` like any other."""
    problem = on_core(op_hctz_iv, fit_settings)

    result = run(problem, fit_settings, size=1, max_iterations=2, threads=2)

    assert result.opid == problem.opid
    assert [p.pid for p in result.parameters] == problem.pids
    assert result.settings == fit_settings
    path = tmp_path / "result.json"
    result.to_json(path)
    read = OptimizationResult.from_json(path)
    np.testing.assert_allclose(read.fits[0].x, result.fits[0].x, rtol=1e-12)
    assert read.fits[0].cost == pytest.approx(result.fits[0].cost)
    assert read.settings == fit_settings


def test_a_fit_of_a_problem_of_the_engine_runs_in_the_engine(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The runner fits a problem without gaps in the engine itself."""
    problem = on_core(op_hctz_iv, fit_settings)

    result = run_optimization(
        problem, fit_settings, size=2, seed=1234, show_progress=False
    )

    assert len(result.fits) == 2
    for fit in result.fits:
        assert fit.optimizer_kwargs["engine"] == "core"
        assert fit.message in CONVERGED or "Max" in fit.message


def test_a_fit_which_the_engine_cannot_run_stays_in_python(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A gap of the problem or of the optimizer keeps the fit in python."""
    problem = on_core(op_hctz_iv, fit_settings)

    # `max_nfev` is a budget of evaluations, which the engine has not
    result = run_optimization(
        problem, fit_settings, size=1, seed=1234, show_progress=False, max_nfev=3
    )

    assert "engine" not in result.fits[0].optimizer_kwargs


def test_a_fit_may_be_held_out_of_the_engine(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """`engine_fit=False` fits with the optimizer in python."""
    problem = on_core(op_hctz_iv, fit_settings)

    result = run_optimization(
        problem,
        fit_settings,
        size=1,
        seed=1234,
        show_progress=False,
        engine_fit=False,
    )

    assert "engine" not in result.fits[0].optimizer_kwargs


def test_a_fit_which_must_run_in_the_engine_names_the_gaps_of_the_problem(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """`engine_fit=True` refuses a fit the engine cannot run, with its gaps."""
    problem = on_core(
        op_hctz_iv, dataclasses.replace(fit_settings, objective="likelihood")
    )

    with pytest.raises(ValueError, match="objective"):
        run_optimization(
            problem,
            problem.settings_initialized,
            size=1,
            show_progress=False,
            engine_fit=True,
        )


def test_bindings_which_cannot_fit_a_request_are_a_gap() -> None:
    """A fit of the engine needs `LoadedModel.fit`, which 0.12.0 is the first to have."""
    assert not fits_requests(None)
    assert not fits_requests(SimpleNamespace(simulate_many=lambda requests: []))
    assert fits_requests(SimpleNamespace(fit=lambda request, threads=None: {}))


def test_the_walk_of_the_engine_is_the_settings_of_the_analysis() -> None:
    """The fields of the walk of the engine are the ones of `ProfileSettings`."""
    settings = ProfileSettings(
        alpha=0.9,
        degrees_of_freedom=2,
        initial_step=0.2,
        min_step=0.05,
        max_step=0.5,
        step_factor=3.0,
        max_cost_fraction=0.1,
        max_points=11,
    )

    walk = walk_of(settings)

    assert walk == {
        "points": 11,
        "step": 0.2,
        "min_step": 0.05,
        "max_step": 0.5,
        "factor": 3.0,
        "rise": 0.1,
        # the delta of the engine is the rise of the cost which bounds a parameter
        "delta": settings.delta / 2.0,
    }
    assert settings.threshold(1.0) == pytest.approx(1.0 + walk["delta"])


def test_a_scan_which_holds_the_other_parameters_is_a_gap_of_the_profiles(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The engine fits the other parameters at every point of a profile."""
    problem = on_core(op_hctz_iv, fit_settings)

    assert gaps_of_the_profiles(problem, ProfileSettings()) == []
    gaps = gaps_of_the_profiles(problem, ProfileSettings(reoptimize=False))

    assert [gap.code for gap in gaps] == ["reoptimize"]


def test_the_profiles_of_the_engine_are_the_profiles_of_python(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The walk of the engine is the walk of `profile_likelihood`, point for point."""
    problem = on_core(op_hctz_iv, fit_settings)
    settings = ProfileSettings(max_points=8)
    parameter_set = problem.parameter_set_model()
    walked = {
        where: profile_likelihood(
            problem,
            fit_settings,
            parameter_set,
            settings,
            show_progress=False,
            engine_profiles=where == "engine",
        )
        for where in ("engine", "python")
    }

    assert walked["engine"].cost == pytest.approx(walked["python"].cost, rel=1e-9)
    for pid, profile in walked["engine"].profiles.items():
        expected = walked["python"].profiles[pid]
        assert len(profile) == len(expected), pid
        np.testing.assert_allclose(profile.values, expected.values, rtol=1e-4)
        np.testing.assert_allclose(profile.costs, expected.costs, rtol=1e-4)
        assert profile.index_optimum == expected.index_optimum
        assert profile.converged.all() == expected.converged.all()
        assert profile.identifiability == expected.identifiability, pid
        for bound, other in (
            (profile.ci_lower, expected.ci_lower),
            (profile.ci_upper, expected.ci_upper),
        ):
            assert (bound is None) == (other is None), pid
            if bound is not None and other is not None:
                assert bound == pytest.approx(other, rel=1e-3)


def test_the_profiles_may_be_held_out_of_the_engine(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`engine_profiles=False` scans in python, which does not reach for the engine."""
    problem = on_core(op_hctz_iv, fit_settings)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the analysis walked in the engine")

    monkeypatch.setattr(fit_core, "profile_points", refuse)

    result = profile_likelihood(
        problem,
        fit_settings,
        problem.parameter_set_model(),
        ProfileSettings(max_points=2),
        show_progress=False,
        engine_profiles=False,
    )

    assert len(result.profiles) == len(problem.pids)


def test_profiles_which_must_be_walked_in_the_engine_name_their_gaps(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """`engine_profiles=True` refuses an analysis the engine cannot walk."""
    problem = on_core(op_hctz_iv, fit_settings)

    with pytest.raises(ValueError, match="reoptimize"):
        profile_likelihood(
            problem,
            fit_settings,
            problem.parameter_set_model(),
            ProfileSettings(reoptimize=False),
            show_progress=False,
            engine_profiles=True,
        )


class Spy:
    """A loaded model which keeps the requests of its fits and may be interrupted.

    The requests say how the runs of a fit were chunked, see `fit_core.run`, and
    `interrupt_at` is the request which raises a `KeyboardInterrupt` instead of
    answering, as a fit which is interrupted on the terminal does.
    """

    def __init__(self, loaded: Any, interrupt_at: int | None = None) -> None:
        self.loaded = loaded
        self.requests: list[Any] = []
        self.interrupt_at = interrupt_at

    def __getattr__(self, name: str) -> Any:
        """Everything but the fit is the loaded model itself."""
        return getattr(self.loaded, name)

    def fit(self, request: Any, **kwargs: Any) -> Any:
        """Answer the fit of a request, or be interrupted at `interrupt_at`."""
        self.requests.append(request)
        if self.interrupt_at is not None and len(self.requests) == self.interrupt_at:
            raise KeyboardInterrupt
        return self.loaded.fit(request, **kwargs)


def spy_on(
    problem: OptimizationProblem,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_at: int | None = None,
) -> Spy:
    """Let the fits of a problem run through a `Spy` on its loaded model."""
    spy = Spy(loaded_model(problem), interrupt_at)
    monkeypatch.setattr(fit_core, "loaded_model", lambda _: spy)
    return spy


def test_a_fit_of_more_starts_than_threads_is_a_request_per_chunk(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The starts of a fit are chunked by the threads, which fills the batch."""
    problem = on_core(op_hctz_iv, fit_settings)
    spy = spy_on(problem, monkeypatch)

    result = run(problem, fit_settings, size=5, seed=1234, max_iterations=2, threads=2)

    assert [len(request["starts"]) for request in spy.requests] == [2, 2, 1]
    assert len(result.fits) == 5
    assert not result.interrupted
    starts = problem.start_values(5, seed=1234)
    for fit, start in zip(result.fits, starts, strict=True):
        np.testing.assert_allclose(fit.x0, np.asarray(start, dtype=float), rtol=1e-12)


def test_a_fit_of_the_engine_reports_a_run_as_soon_as_its_chunk_answered(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """`on_run` gets the index of every run of the fit and its result."""
    problem = on_core(op_hctz_iv, fit_settings)
    reported: list[tuple[int, float]] = []

    result = run(
        problem,
        fit_settings,
        size=3,
        seed=1234,
        max_iterations=2,
        threads=1,
        indices=[2, 0],
        on_run=lambda k, fit: reported.append((k, fit.cost)),
    )

    assert [k for k, _ in reported] == [2, 0]
    assert [cost for _, cost in reported] == [fit.cost for fit in result.fits]


def test_a_fit_of_the_engine_which_is_interrupted_keeps_the_chunks_which_finished(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted fit is the runs of its chunks which answered, as in python."""
    problem = on_core(op_hctz_iv, fit_settings)
    spy = spy_on(problem, monkeypatch, interrupt_at=2)
    reported: list[int] = []

    result = run(
        problem,
        fit_settings,
        size=5,
        seed=1234,
        max_iterations=2,
        threads=2,
        on_run=lambda k, _: reported.append(k),
    )

    assert len(spy.requests) == 2
    assert result.interrupted
    assert len(result.fits) == 2
    assert reported == [0, 1]


def test_a_fit_of_the_engine_which_is_interrupted_in_its_first_chunk_is_no_fit(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fit which has no run to keep is interrupted, it is not an empty result."""
    problem = on_core(op_hctz_iv, fit_settings)
    spy_on(problem, monkeypatch, interrupt_at=1)

    with pytest.raises(KeyboardInterrupt):
        run(problem, fit_settings, size=2, max_iterations=2, threads=2)


@pytest.mark.parametrize(
    ("starts", "threads", "expected"),
    [(8, 3, (3, 3)), (3, 8, (8, 1)), (4, 4, (4, 1)), (1, None, (1, 1)), (0, 2, (2, 1))],
)
def test_the_chunks_of_a_fit_fill_the_threads_of_the_engine(
    starts: int, threads: int | None, expected: tuple[int, int]
) -> None:
    """A chunk is as wide as the threads and a fit is at least one request."""
    assert fit_core.requests_of(starts, threads) == expected
