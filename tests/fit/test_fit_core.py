"""The request of a fit of sbmlsim-core, built from an optimization problem (#556).

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
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, override

import numpy as np
import pandas as pd
import pytest

from sbmlsim import Q
from sbmlsim.data import DataSet
from sbmlsim.engine import Engine
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import (
    FitData,
    FitMapping,
    FitMappingCollection,
    FitParameter,
    FitSettings,
    fit_core,
)
from sbmlsim.fit.fit_core import (
    CONVERGED,
    engine_problem,
    fit_request,
    fits_requests,
    gaps_of,
    gaps_of_the_profiles,
    loaded_model,
    loaded_models,
    model_ids,
    run,
    walk_of,
)
from sbmlsim.fit.identifiability import ProfileSettings, profile_likelihood
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    RUN_IN_PYTHON,
    RUN_IN_THE_ENGINE,
    LossFunctionType,
    ObjectiveType,
    OptimizationAlgorithmType,
    ParameterScaleType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization
from sbmlsim.fit.sampling import SamplingType
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Change, Formula, Observable, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from tests.engines import require_core_fit
from tests.fit.test_data_at_change import (
    MODEL_FILE,
    TIMES,
    TRUE_KE,
    TRUTH,
    UrineStudy,
)
from tests.simulator.models import sbml_pk


def on_core(problem: OptimizationProblem, settings: FitSettings) -> OptimizationProblem:
    """Initialize a problem on sbmlsim-core, the engine which fits itself."""
    require_core_fit()
    problem.initialize(settings)
    problem.set_simulator(Simulator(n_workers=1, engine=Engine.CORE))
    assert problem.engine is Engine.CORE
    return problem


def core_problem(problem: OptimizationProblem) -> Any:
    """Get the objective of the engine for the request of a problem."""
    return engine_problem(problem, fit_request(problem), threads=1)


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


def test_a_posterior_of_parameters_without_priors_is_no_gap(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A posterior of parameters without a prior is the likelihood of its data.

    The parameters of the HCTZ fit carry no prior, whose uniform prior over
    the bounds is a constant of the cost and no residual on either side, so
    the request of such a posterior carries no prior at all.
    """
    problem = on_core(
        op_hctz_iv,
        dataclasses.replace(fit_settings, objective=ObjectiveType.POSTERIOR),
    )

    assert gaps_of(problem) == []
    assert fit_request(problem)["priors"] == []


def test_a_likelihood_of_a_noise_the_fit_does_not_move_is_no_gap(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A likelihood whose noise is fixed is the weighted residuals of a request (#613).

    The data of the HCTZ fit carries its errors, so the scale of the noise of
    every mapping is a number and the residual of a measurement is the one of
    a curve of `noise` that scale.
    """
    problem = on_core(
        op_hctz_iv,
        dataclasses.replace(fit_settings, objective=ObjectiveType.LIKELIHOOD),
    )

    assert gaps_of(problem) == []
    # the loss function is no gap of a likelihood, which does not apply it
    # (the problem itself refuses a residual of a baseline with a likelihood)
    problem = on_core(
        op_hctz_iv,
        dataclasses.replace(
            fit_settings,
            objective=ObjectiveType.LIKELIHOOD,
            loss_function=LossFunctionType.SOFT_L1,
        ),
    )
    assert gaps_of(problem) == []


def test_a_problem_of_another_engine_is_no_fit_of_the_engine(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Only a problem which simulates in sbmlsim-core fits in it."""
    # a problem of another engine than the core, so there is a gap to name
    op_hctz_iv.engine = Engine.ROADRUNNER
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
    assert result.optimizer_of == RUN_IN_THE_ENGINE
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
    assert result.optimizer_of == RUN_IN_PYTHON


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
    assert result.optimizer_of == RUN_IN_PYTHON


def test_a_fit_which_must_run_in_the_engine_names_the_gaps_of_the_problem(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """`engine_fit=True` refuses a fit the engine cannot run, with its gaps."""
    problem = on_core(
        op_hctz_iv,
        dataclasses.replace(fit_settings, loss_function=LossFunctionType.SOFT_L1),
    )

    with pytest.raises(ValueError, match="loss_function"):
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

    assert walked["engine"].profiles_of == RUN_IN_THE_ENGINE
    assert walked["python"].profiles_of == RUN_IN_PYTHON
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
    """A loaded model which keeps the requests of its fits.

    The requests say what a fit asked the engine for, see `fit_core.run`.
    """

    def __init__(self, loaded: Any) -> None:
        self.loaded = loaded
        self.requests: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        """Everything but the fit is the loaded model itself."""
        return getattr(self.loaded, name)

    def fit(self, request: Any, **kwargs: Any) -> Any:
        """Answer the fit of a request."""
        self.requests.append(request)
        return self.loaded.fit(request, **kwargs)


def spy_on(problem: OptimizationProblem, monkeypatch: pytest.MonkeyPatch) -> Spy:
    """Let the fits of a problem run through a `Spy` on its loaded model."""
    spy = Spy(loaded_model(problem))
    monkeypatch.setattr(fit_core, "loaded_models", lambda _: {"model": spy})
    return spy


def interrupt_after(
    problem: OptimizationProblem, monkeypatch: pytest.MonkeyPatch, runs: int
) -> None:
    """Let the engine report `runs` runs of a fit and be interrupted then.

    The fit of the engine reports a run of it as soon as that run finished and
    raises the `KeyboardInterrupt` of a Ctrl-C which reaches it, which this is
    for a fit of the test: the runs are the ones the engine ran, in the order it
    reported them, and the fit is stopped after them.
    """
    loaded = loaded_model(problem)
    fit = loaded.fit

    def interrupted(request: Any, *, on_run: Any = None, **kwargs: Any) -> Any:
        """Report the first runs of the fit of a request, then interrupt it."""
        answer = fit(request, **kwargs)
        for index, finished in enumerate(answer["runs"][:runs]):
            if on_run is not None:
                on_run(index, finished)
        raise KeyboardInterrupt

    monkeypatch.setattr(
        fit_core,
        "loaded_models",
        lambda _: {"model": SimpleNamespace(fit=interrupted)},
    )


def test_the_whole_fit_of_the_engine_is_one_request(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every start of a fit is in one request, which fills the pool of the engine."""
    problem = on_core(op_hctz_iv, fit_settings)
    spy = spy_on(problem, monkeypatch)

    result = run(problem, fit_settings, size=5, seed=1234, max_iterations=2, threads=2)

    assert [len(request["starts"]) for request in spy.requests] == [5]
    assert len(result.fits) == 5
    assert not result.interrupted
    starts = problem.start_values(5, seed=1234)
    for fit, start in zip(result.fits, starts, strict=True):
        np.testing.assert_allclose(fit.x0, np.asarray(start, dtype=float), rtol=1e-12)


def test_a_fit_of_the_engine_reports_a_run_as_soon_as_it_finished(
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


def test_a_fit_of_the_engine_which_is_interrupted_keeps_the_runs_it_reported(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted fit is the runs the engine reported, as in python.

    The engine of the test reports two of the five runs of the fit and is then
    interrupted, as Ctrl-C of a terminal interrupts a fit which runs.
    """
    problem = on_core(op_hctz_iv, fit_settings)
    interrupt_after(problem, monkeypatch, runs=2)
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

    assert result.interrupted
    assert reported == [0, 1]
    starts = problem.start_values(5, seed=1234)[:2]
    assert len(result.fits) == 2
    for fit, start in zip(result.fits, starts, strict=True):
        np.testing.assert_allclose(fit.x0, np.asarray(start, dtype=float), rtol=1e-12)


def test_a_fit_of_the_engine_which_is_interrupted_before_a_run_is_no_fit(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fit which has no run to keep is interrupted, it is not an empty result."""
    problem = on_core(op_hctz_iv, fit_settings)
    interrupt_after(problem, monkeypatch, runs=0)

    with pytest.raises(KeyboardInterrupt):
        run(problem, fit_settings, size=2, max_iterations=2, threads=2)


def test_a_fit_of_bindings_which_report_no_run_reports_them_when_it_answered(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runs of a fit are reported, whether the bindings report them or not.

    `on_run` of `LoadedModel.fit` is new in sbmlsim-core 0.12.0, where the fit
    itself is as well, so bindings of a release between the two report the runs
    of a fit when it answered, see `fit_core.reports_runs`.
    """
    problem = on_core(op_hctz_iv, fit_settings)
    monkeypatch.setattr(fit_core, "reports_runs", lambda _: False)
    reported: list[int] = []

    result = run(
        problem,
        fit_settings,
        size=3,
        seed=1234,
        max_iterations=2,
        threads=2,
        on_run=lambda k, _: reported.append(k),
    )

    assert reported == [0, 1, 2]
    assert len(result.fits) == 3
    assert not result.interrupted


def test_a_differential_evolution_of_the_engine_is_a_run_of_its_population(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A differential evolution draws its population in the engine, without a start.

    The starts of the fit are no starts of such a request: the engine draws the
    members from the seed of the evolution and answers with the best of them.
    """
    problem = on_core(op_hctz_iv, fit_settings)

    result = run(
        problem,
        fit_settings,
        size=1,
        algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
        max_iterations=3,
        threads=2,
        evolution={"population": 8, "seed": 1234, "polish": False},
    )

    assert len(result.fits) == 1
    fit = result.fits[0]
    assert fit.algorithm == "DIFFERENTIAL_EVOLUTION"
    assert np.isfinite(fit.cost)
    lower = np.asarray([p.lower_bound for p in problem.parameters], dtype=float)
    upper = np.asarray([p.upper_bound for p in problem.parameters], dtype=float)
    assert np.all(fit.x >= lower) and np.all(fit.x <= upper)


def test_a_differential_evolution_of_many_runs_of_the_engine_is_no_fit(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Every run of such a fit would draw the same population, which is one run."""
    problem = on_core(op_hctz_iv, fit_settings)

    with pytest.raises(ValueError, match="draws its population from one seed"):
        run(
            problem,
            fit_settings,
            size=2,
            algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
            max_iterations=1,
        )


# ----------------------------------------------------------------------------
# the jacobian of the residuals, the Fisher information of the optimum
# ----------------------------------------------------------------------------
def test_the_jacobian_of_the_engine_is_the_derivative_of_the_residuals(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The forward sensitivities are the derivative the differences converge to.

    At a tolerance of `1e-10` the central differences of python reach the
    sensitivities point for point over a step of `1e-3`. The jacobian of the
    data of an iv fit says nothing about the absorption of an oral dose, so
    the columns of its two absorption parameters are zero.
    """
    from sbmlsim.fit.fisher import jacobian

    tight = dataclasses.replace(
        fit_settings, relative_tolerance=1e-10, absolute_tolerance=1e-12
    )
    problem = on_core(op_hctz_iv, tight)
    x = problem.to_scale(problem.xmodel)

    sensitivities = fit_core.residual_jacobian(problem, x, threads=2)

    differences = jacobian(problem, x, step=1e-3)[: sensitivities.shape[0]]
    assert sensitivities.shape == (len(problem.residuals(x)), len(problem.parameters))
    np.testing.assert_allclose(sensitivities, differences, rtol=1e-4, atol=1e-6)
    assert np.all(sensitivities[:, :2] == 0.0)
    assert np.linalg.norm(sensitivities[:, 2]) > 0.0


def test_the_sensitivities_hold_at_the_tolerance_of_a_fit(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Which is what the Fisher information takes them for.

    The sensitivities are integrated next to the states and not differenced,
    so the information they give at the default tolerance of a fit (`1e-6`) is
    the one of a tolerance of `1e-10`; the central differences of that
    tolerance are off it by percents, whatever their step, because they divide
    the noise of the integrator by it.
    """
    from sbmlsim.fit.fisher import jacobian

    tight = dataclasses.replace(
        fit_settings, relative_tolerance=1e-10, absolute_tolerance=1e-12
    )
    problem = on_core(op_hctz_iv, tight)
    x = problem.to_scale(problem.xmodel)
    exact = fit_core.residual_jacobian(problem, x, threads=2)
    information = exact.T @ exact

    problem.initialize(fit_settings)
    sensitivities = fit_core.residual_jacobian(problem, x, threads=2)

    np.testing.assert_allclose(
        sensitivities.T @ sensitivities, information, rtol=1e-3, atol=1e-12
    )
    for step in (1e-2, 1e-3):
        differences = jacobian(problem, x, step=step)[: exact.shape[0]]
        off = abs(float((differences.T @ differences)[2, 2] / information[2, 2] - 1.0))
        assert off > 5e-3, f"the differences of the step {step} are off by {off:g}"


def test_the_jacobian_of_the_engine_is_in_the_space_of_the_problem(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Its point is the one of the problem, carried into the units of the model.

    The engine searches the entity of the model, which the factor of its
    target carries a parameter of its own unit into; the engine itself says
    which values a point of its search space stands for.
    """
    problem = on_core(op_hctz_pk, fit_settings)
    core = core_problem(problem)
    factors = np.asarray(fit_core._factors_of(problem), dtype=float)

    for change in (1.0, 1.3):
        x = problem.to_scale(problem.xmodel * change)

        point = fit_core._point_of(problem, x)

        np.testing.assert_allclose(
            core.values(point),
            np.asarray(problem.from_scale(x), dtype=float) * factors,
            rtol=1e-12,
        )
    np.testing.assert_allclose(
        fit_core._column_factors(problem),
        [
            1.0 if scale.is_log else factor
            for scale, factor in zip(problem.scales_initialized, factors, strict=True)
        ],
        rtol=1e-12,
    )


def test_a_problem_of_another_engine_has_no_jacobian_of_the_engine(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A problem of roadrunner is a gap of the jacobian, as it is one of the fit."""
    problem = op_hctz_iv
    # a problem of another engine than the core, so there is a gap to name
    problem.engine = Engine.ROADRUNNER
    problem.initialize(fit_settings)

    gaps = fit_core.gaps_of_a_jacobian(problem)

    assert [gap.code for gap in gaps] == ["engine"]
    assert "roadrunner" in gaps[0].message


def test_the_information_of_the_engine_is_the_one_of_the_problem(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The Fisher information takes the sensitivities where the engine has them."""
    from sbmlsim.fit.fisher import (
        CENTRAL_DIFFERENCES,
        SENSITIVITIES,
        FisherInformation,
        fisher_information,
    )

    problem = on_core(op_hctz_pk, fit_settings)
    parameter_set = problem.parameter_set_model()

    information = fisher_information(problem, fit_settings, parameter_set, n_cores=2)

    assert information.jacobian_of == SENSITIVITIES
    assert information.step == 0.0
    assert information.rank == information.k
    # the record says which jacobian the matrix came from
    again = FisherInformation.from_dict(information.to_dict())
    assert again.jacobian_of == SENSITIVITIES

    # the differences of python are the same information, to their accuracy
    differences = fisher_information(
        problem, fit_settings, parameter_set, n_cores=2, engine_jacobian=False
    )
    assert differences.jacobian_of == CENTRAL_DIFFERENCES
    np.testing.assert_allclose(
        information.matrix, differences.matrix, rtol=2e-2, atol=1e-6
    )
    np.testing.assert_allclose(
        information.summary_df.se.to_numpy(),
        differences.summary_df.se.to_numpy(),
        rtol=2e-2,
    )


def test_a_step_of_the_information_asks_for_the_differences(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A caller which gives a step wants the differences of that step."""
    from sbmlsim.fit.fisher import CENTRAL_DIFFERENCES, fisher_information

    problem = on_core(op_hctz_pk, fit_settings)

    information = fisher_information(
        problem, fit_settings, problem.parameter_set_model(), step=1e-3, n_cores=2
    )

    assert information.jacobian_of == CENTRAL_DIFFERENCES
    assert information.step == 1e-3


def shifted_problem(
    tmp_path: Path,
    mapping: str = "fm_shifted",
    experiment: type[SimulationExperiment] = UrineStudy,
) -> OptimizationProblem:
    """Get the problem of one urine mapping of `UrineStudy`.

    `fm_shifted` reads the simulation whose protocol runs from the start `-24`
    with the collection reset at `0`, shifted by `24` so that the times of its
    data are the ones of the collection (`tests.fit.test_data_at_change`).
    """
    (tmp_path / MODEL_FILE).write_text(sbml_pk(urine=True), encoding="utf-8")
    return OptimizationProblem(
        "shifted",
        [FitMappingCollection(experiment=experiment, mappings=[mapping])],
        [
            FitParameter(
                pid="ke",
                lower_bound=0.01,
                upper_bound=1.0,
                start_value=0.5,
                unit="1/hr",
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
        engine=Engine.CORE,
    )


def test_the_residuals_of_a_shifted_simulation_are_the_ones_of_the_problem(
    tmp_path: Path,
) -> None:
    """A curve measures in the time of its condition, not in the time of the data.

    The output times of a condition are the ones of its plan, which `time_shift`
    is applied to only on the way back, so a curve of a simulation whose time
    is shifted measures at the times the plan outputs (#613).
    """
    problem = on_core(
        shifted_problem(tmp_path),
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
    )
    assert gaps_of(problem) == []
    core = core_problem(problem)

    x = problem.to_scale(np.array([TRUE_KE]))
    residuals = np.asarray(core.residuals(x), dtype=float)

    expected = np.asarray(problem.residuals(x), dtype=float)
    assert residuals.shape == expected.shape
    np.testing.assert_allclose(residuals, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("mapping", ["fm_shifted", "fm_noisy"])
def test_a_curve_measures_at_the_output_times_of_its_condition(
    tmp_path: Path, mapping: str
) -> None:
    """The time of a point is the output time of the row it reads, exactly.

    The engine resolves the time of a measurement to the row of its own time
    within the resolution of that time, so a curve whose times are the ones
    of the data instead reads the right row only while the difference stays
    inside that tolerance. The problem reads its rows at the output times of
    its plan (`OptimizationProblem._x_plan`), and so does its curve: `fm_shifted`
    simulates in a shifted time, and the data of `fm_noisy` has a time within
    the resolution of the time of a change, whose output time is the time of
    the change (#613).
    """
    problem = on_core(
        shifted_problem(tmp_path, mapping),
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
    )
    assert gaps_of(problem) == []

    request = fit_request(problem)

    output = {
        condition["id"]: condition["request"]["output"]["times"]
        for condition in request["conditions"]
    }
    assert request["curves"]
    for curve in request["curves"]:
        times = output[curve["condition"]]
        assert times
        for time in curve["times"]:
            assert time in times, (
                f"the curve '{curve['id']}' measures at t = {time}, which is no "
                f"output time of its condition"
            )


class FoldedChangeStudy(UrineStudy):
    """`UrineStudy` whose collection is halved at 24 h instead of emptied.

    The executor folds a formula at a change by simulating the segment before
    it, so the number it folds `0.5 * Aurine` into is the one of the point the
    request was built at, see `_gaps_of_the_values`.
    """

    def simulations(self) -> dict[str, Simulation]:
        return {
            **super().simulations(),
            "folded": Simulation(
                end=48,
                changes=[
                    Change(0, {"PODOSE": Q(100, "mg")}),
                    Change(24, {"Aurine": "0.5 * Aurine"}),
                ],
            ),
        }

    def tasks(self) -> dict[str, Task]:
        return {
            **super().tasks(),
            "task_folded": Task(model="m", simulation="folded"),
        }

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm_folded": FitMapping(
                self,
                reference=FitData(self, dataset="urine", xid="time", yid="Aurine"),
                observable=FitData(self, task="task_folded", xid="time", yid="Aurine"),
            )
        }


def test_a_change_of_a_linear_formula_is_no_gap() -> None:
    """A formula the condition carries as a form keeps the fit in the engine."""
    problem = on_core(
        shifted_problem(Path(tempfile.mkdtemp()), "fm_folded", FoldedChangeStudy),
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
    )

    assert gaps_of(problem) == []


def test_the_residuals_of_a_formed_change_follow_the_parameters() -> None:
    """The criterion: the residuals are the problem's away from the build point too.

    `0.5 * Aurine` is a form the engine evaluates from its own state, so the
    residuals follow the parameter where a folded number would hold the change
    at the point the request was built at (#617).
    """
    problem = on_core(
        shifted_problem(Path(tempfile.mkdtemp()), "fm_folded", FoldedChangeStudy),
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
    )
    core = core_problem(problem)
    start = np.asarray(problem.xmodel, dtype=float)
    seen = []
    for factor in (1.0, 1.2, 0.7):
        x = problem.to_scale(start * factor)
        seen.append(np.asarray(problem.residuals(x), dtype=float))
        assert np.asarray(core.residuals(x), dtype=float) == pytest.approx(
            seen[-1], rel=1e-9, abs=1e-12
        )
    # the parameter moves the residuals, so the agreement is not trivial
    assert not np.allclose(seen[0], seen[1])
    assert not np.allclose(seen[0], seen[2])


class NoFormFoldStudy(FoldedChangeStudy):
    """`FoldedChangeStudy` whose change is no linear form, `Aurine * Aurine`."""

    def simulations(self) -> dict[str, Simulation]:
        return {
            **super().simulations(),
            "folded": Simulation(
                end=48,
                changes=[
                    Change(0, {"PODOSE": Q(100, "mg")}),
                    Change(24, {"Aurine": "Aurine * Aurine"}),
                ],
            ),
        }


def test_a_change_of_a_formula_which_is_no_form_keeps_the_gap() -> None:
    """A formula which is no linear form is still fitted in python."""
    problem = on_core(
        shifted_problem(Path(tempfile.mkdtemp()), "fm_folded", NoFormFoldStudy),
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
    )

    gaps = gaps_of(problem)

    assert [gap.code for gap in gaps] == ["change_value"]
    assert "Aurine * Aurine" in gaps[0].message


class ConstantProductStudy(FoldedChangeStudy):
    """`FoldedChangeStudy` whose change is no form but reads a constant, `ka * ka`."""

    def simulations(self) -> dict[str, Simulation]:
        return {
            **super().simulations(),
            "folded": Simulation(
                end=48,
                changes=[
                    Change(0, {"PODOSE": Q(100, "mg")}),
                    Change(24, {"Aurine": "ka * ka"}),
                ],
            ),
        }


def test_a_change_of_constants_is_no_gap_though_it_is_no_form() -> None:
    """A formula over constants folds into one number everywhere (the first gate).

    `ka * ka` is no linear form, so only `_constant_at_the_changes` keeps it
    out of the gaps; without that gate the fit would lose the engine.
    """
    from sbmlsim.simulator.executor_core import form_of

    problem = on_core(
        shifted_problem(Path(tempfile.mkdtemp()), "fm_folded", ConstantProductStudy),
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
    )

    plan = problem.evaluated_plan(0, np.asarray(problem.xmodel, dtype=float), [])
    assignment = next(a for e in plan.events for a in e.assignments if a.formula)
    assert form_of(assignment, fit_core.fit_model(problem)) is None
    assert gaps_of(problem) == []


class MixedFoldStudy(FoldedChangeStudy):
    """`FoldedChangeStudy` with a form after a change of constants which is no form."""

    def simulations(self) -> dict[str, Simulation]:
        return {
            **super().simulations(),
            "folded": Simulation(
                end=48,
                changes=[
                    Change(0, {"PODOSE": Q(100, "mg")}),
                    Change(12, {"Aurine": "ka * ka"}),
                    Change(24, {"Aurine": "0.5 * Aurine"}),
                ],
            ),
        }


def test_a_form_is_a_gap_in_a_plan_which_folds_another_formula() -> None:
    """The request writes the forms all or nothing, so a plan of both folds both."""
    problem = on_core(
        shifted_problem(Path(tempfile.mkdtemp()), "fm_folded", MixedFoldStudy),
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
    )

    gaps = gaps_of(problem)

    assert [gap.code for gap in gaps] == ["change_value"]
    assert "0.5 * Aurine" in gaps[0].message


class PreinitFoldStudy(NoFormFoldStudy):
    """`FoldedChangeStudy` whose pre-initialization sets the state the change reads.

    The pre-initialization sets `Aurine` to a number, which the executor
    resolves a formula *before the initialization* with, but the simulation
    moves it before the change at 24 h, so the number of the request is still
    the one of the point it was built at.
    """

    def simulations(self) -> dict[str, Simulation]:
        return {
            **super().simulations(),
            "folded": Simulation(
                end=48,
                preinit_changes={"Aurine": Q(0, "mg")},
                changes=[
                    Change(0, {"PODOSE": Q(100, "mg")}),
                    Change(24, {"Aurine": "Aurine * Aurine"}),
                ],
            ),
        }


def test_a_change_which_reads_a_state_the_pre_initialization_sets_is_a_gap() -> None:
    """The pre-initialization number of a state is not its value at a change."""
    problem = on_core(
        shifted_problem(Path(tempfile.mkdtemp()), "fm_folded", PreinitFoldStudy),
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
    )

    gaps = gaps_of(problem)

    assert [gap.code for gap in gaps] == ["change_value"]
    assert "'Aurine'" in gaps[0].message


class DerivedFoldStudy(FoldedChangeStudy):
    """`FoldedChangeStudy` whose pre-initialization sets `ka` by a formula over `ke`.

    A derived value of the condition moves `ka` with the search, so the number
    the executor folds the change `ka * ka` into is the one of the point the
    request was built at.
    """

    def simulations(self) -> dict[str, Simulation]:
        return {
            **super().simulations(),
            "folded": Simulation(
                end=48,
                preinit_changes={"ka": "2 * ke"},
                changes=[
                    Change(0, {"PODOSE": Q(100, "mg")}),
                    Change(24, {"Aurine": "ka * ka"}),
                ],
            ),
        }


def test_a_change_which_reads_a_parameter_a_formula_sets_is_a_gap() -> None:
    """A parameter the pre-initialization assigns by a formula is no constant."""
    problem = on_core(
        shifted_problem(Path(tempfile.mkdtemp()), "fm_folded", DerivedFoldStudy),
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
    )

    gaps = gaps_of(problem)

    assert "change_value" in [gap.code for gap in gaps]
    assert "'ka'" in next(g for g in gaps if g.code == "change_value").message


class ConstantFoldStudy(FoldedChangeStudy):
    """`FoldedChangeStudy` whose change reads a constant parameter of the model.

    `ka` is the variable of no rule and of no initial assignment and no change
    of the plan sets it, so the number the executor folds `0.1 * ka` into is
    the same at every point of the search.
    """

    def simulations(self) -> dict[str, Simulation]:
        return {
            **super().simulations(),
            "folded": Simulation(
                end=48,
                changes=[
                    Change(0, {"PODOSE": Q(100, "mg")}),
                    Change(24, {"Aurine": "0.1 * ka"}),
                ],
            ),
        }


def test_a_change_which_reads_a_constant_of_the_model_is_no_gap() -> None:
    """A formula the executor folds into the same number everywhere is no gap."""
    problem = on_core(
        shifted_problem(Path(tempfile.mkdtemp()), "fm_folded", ConstantFoldStudy),
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
    )

    assert gaps_of(problem) == []


class RepeatedFoldStudy(NoFormFoldStudy):
    """`NoFormFoldStudy` which squares the collection at 12 h and at 24 h."""

    def simulations(self) -> dict[str, Simulation]:
        return {
            **super().simulations(),
            "folded": Simulation(
                end=48,
                changes=[
                    Change(0, {"PODOSE": Q(100, "mg")}),
                    Change(12, {"Aurine": "Aurine * Aurine"}),
                    Change(24, {"Aurine": "Aurine * Aurine"}),
                ],
            ),
        }


def test_one_formula_the_executor_cannot_fold_is_one_gap() -> None:
    """A formula at several changes is reported once, with the number of them.

    The report of the gaps of a problem is one message
    (`sbmlsim.fit.runner.run_optimization`), so a formula of a protocol of
    many doses or collections names its changes instead of being one gap per
    change.
    """
    problem = on_core(
        shifted_problem(Path(tempfile.mkdtemp()), "fm_folded", RepeatedFoldStudy),
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
    )

    gaps = gaps_of(problem)

    assert [gap.code for gap in gaps] == ["change_value"]
    assert "2 changes" in gaps[0].message


def dose_problem(
    tmp_path: Path, experiment: type[SimulationExperiment] = UrineStudy
) -> OptimizationProblem:
    """Get the problem whose parameter is the dose a change of its plan sets.

    The simulation `shifted` of `UrineStudy` doses by the plain numeric change
    `Change(-24, {"PODOSE": ...})`, so `Plan.with_values` writes the fitted
    dose into that change (`tests.fit.test_data_at_change`).
    """
    (tmp_path / MODEL_FILE).write_text(sbml_pk(urine=True), encoding="utf-8")
    return OptimizationProblem(
        "dose",
        [FitMappingCollection(experiment=experiment, mappings=["fm_shifted"])],
        [
            FitParameter(
                pid="PODOSE",
                lower_bound=10.0,
                upper_bound=1000.0,
                start_value=100.0,
                unit="mg",
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
        engine=Engine.CORE,
    )


def test_a_parameter_whose_target_a_change_sets_is_a_gap(tmp_path: Path) -> None:
    """A parameter of the fit written at a change is a number of the request.

    `Plan.with_values` replaces the assignment of a target wherever the plan
    has one, so the fitted value lands in the change itself. The parameter of
    the condition overrides the pre-initialization of the plan but not a
    change applied during the simulation, which the request carries as the
    number of the point it was built at.
    """
    problem = on_core(
        dose_problem(tmp_path),
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
    )

    gaps = gaps_of(problem)

    assert [gap.code for gap in gaps] == ["parameter_value"]
    assert "'PODOSE'" in gaps[0].message


def test_the_engine_would_hold_a_parameter_of_a_change_where_it_was_built(
    tmp_path: Path,
) -> None:
    """Why that gap exists: the request of such a problem freezes the parameter.

    The residuals of the problem follow the dose, the ones of the request do
    not: its change carries the number of the point the request was built at.
    Should the engine ever follow the parameter at a change, this test fails
    and the gap can go.
    """
    problem = on_core(
        dose_problem(tmp_path),
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
    )
    core = core_problem(problem)

    costs = []
    for dose in (100.0, 150.0):
        x = problem.to_scale(np.array([dose]))
        python = np.asarray(problem.residuals(x), dtype=float)
        engine = np.asarray(core.residuals(x), dtype=float)
        costs.append((float(np.sum(python**2)), float(np.sum(engine**2))))
    # the dose moves the residuals of the problem
    assert costs[0][0] != pytest.approx(costs[1][0])
    # and leaves the ones of the engine where the request was built
    assert costs[0][1] == pytest.approx(costs[1][1])


#: the standard deviation of the measurements of `ZeroBaselineStudy`, in mg
ZERO_SD = 0.1

#: the units of the urine data with its errors
ZERO_UNITS = {"time": "hr", "Aurine": "mg", "Aurine_sd": "mg"}


class ZeroBaselineStudy(UrineStudy):
    """The urine of `UrineStudy` with the errors of a study on its data.

    `fm_zero` measures nothing in the urine at the first time, a zero
    baseline: `error_weighting` gives that point the weight `|y / sd| = 0.0`,
    which `_noise_of` turns into a positive infinity, the standard deviation
    of a measurement which carries no information. An error of exactly zero
    is no second source of it, since the problem reads such an error as a
    missing one and replaces it with the largest error of the mapping, so the
    other end, an infinite weight whose noise is `0.0`, is the weight of a
    mapping (`test_an_infinite_weight_is_still_a_gap`).
    """

    @override
    def datasets(self) -> dict[str, DataSet]:
        datasets = super().datasets()
        amounts = list(TRUTH["urine"])
        datasets["zero"] = DataSet.from_df(
            pd.DataFrame(
                {
                    "time": TIMES,
                    "Aurine": [0.0, *amounts[1:]],
                    "Aurine_sd": [ZERO_SD] * len(TIMES),
                }
            ),
            udict=ZERO_UNITS,
            ureg=self.ureg,
        )
        return datasets

    @override
    def fit_mappings(self) -> dict[str, FitMapping]:
        mappings = super().fit_mappings()
        mappings["fm_zero"] = FitMapping(
            self,
            reference=FitData(
                self,
                dataset="zero",
                xid="time",
                yid="Aurine",
                yid_sd="Aurine_sd",
            ),
            observable=FitData(self, task="task_collected", xid="time", yid="Aurine"),
        )
        return mappings


class TwoModelStudy(UrineStudy):
    """The study of the urine with a second model of its own changes.

    The mappings of the study read both models, which no request of `fit/1`
    carries: a request has one model and a condition per simulation.
    """

    @override
    def models(self) -> dict[str, AbstractModel | Path]:
        source = Path(self.base_path or ".") / MODEL_FILE
        return {
            "m": AbstractModel(source=source),
            # another model of the same file, loaded with a change of its own
            "m2": AbstractModel(source=source, changes={"ke": 0.4}),
        }

    @override
    def tasks(self) -> dict[str, Task]:
        tasks = super().tasks()
        tasks["task_second"] = Task(model="m2", simulation="collected")
        return tasks

    @override
    def fit_mappings(self) -> dict[str, FitMapping]:
        mappings = super().fit_mappings()
        mappings["fm_second"] = FitMapping(
            self,
            reference=FitData(self, dataset="urine", xid="time", yid="Aurine"),
            observable=FitData(self, task="task_second", xid="time", yid="Aurine"),
        )
        return mappings


def zero_problem(
    tmp_path: Path, mapping: str, weights: float | None = None
) -> OptimizationProblem:
    """Get the problem of one mapping of `ZeroBaselineStudy`."""
    (tmp_path / MODEL_FILE).write_text(sbml_pk(urine=True), encoding="utf-8")
    collection = FitMappingCollection(
        experiment=ZeroBaselineStudy, mappings=[mapping], weights=weights
    )
    return OptimizationProblem(
        "zero",
        [collection],
        [
            FitParameter(
                pid="ke",
                lower_bound=0.01,
                upper_bound=1.0,
                start_value=0.5,
                unit="1/hr",
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
        engine=Engine.CORE,
    )


def error_weighting() -> FitSettings:
    """Get the settings which weight a point by its coefficient of variation."""
    return FitSettings(
        parameter_scale=ParameterScaleType.LINEAR,
        weighting_points=WeightingPointsType.ERROR_WEIGHTING,
    )


def test_a_point_of_a_zero_measurement_is_no_gap_and_carries_an_infinite_noise(
    tmp_path: Path,
) -> None:
    """A measurement which carries no information is a point of the request.

    The weight of a measurement of zero under `error_weighting` is zero, its
    standard deviation an infinity, and the point stays in the curve, so the
    residual vector of the request keeps the length and the order of the one
    of the problem (sbmlsim-core `INFINITE_NOISE_VERSION`).
    """
    problem = on_core(zero_problem(tmp_path, "fm_zero"), error_weighting())

    assert gaps_of(problem) == []

    curve = fit_request(problem)["curves"][0]
    assert len(curve["noise"]) == len(curve["times"]) == len(TIMES)
    assert curve["noise"][0] == float("inf")
    assert all(noise > 0.0 for noise in curve["noise"][1:])
    assert all(np.isfinite(noise) for noise in curve["noise"][1:])


def test_the_residuals_of_a_zero_weight_point_are_the_ones_of_python(
    tmp_path: Path,
) -> None:
    """The engine computes the residuals of the problem, the zero included.

    Python computes `residual * sqrt(0.0)` and the engine `(sim - value) /
    inf`; both are `+-0.0` with the sign of the residual, which comparing the
    whole vector pins.
    """
    problem = on_core(zero_problem(tmp_path, "fm_zero"), error_weighting())
    core = core_problem(problem)

    for factor in (1.0, 1.3, 0.7):
        x = problem.to_scale(problem.xmodel * factor)
        expected = np.asarray(problem.residuals(x), dtype=float)

        residuals = np.asarray(core.residuals(x), dtype=float)

        assert residuals.shape == expected.shape
        np.testing.assert_allclose(residuals, expected, rtol=1e-9, atol=1e-12)
        assert np.signbit(residuals[0]) == np.signbit(expected[0])


def mapping_weighting() -> FitSettings:
    """Get the settings which weight a curve by the weight of its mapping."""
    return FitSettings(
        parameter_scale=ParameterScaleType.LINEAR,
        weighting_curves=(WeightingCurvesType.MAPPING,),
    )


def test_a_user_weight_of_zero_is_no_gap(tmp_path: Path) -> None:
    """A weight of a mapping of zero zeroes every point of that mapping.

    `weight = weight_curve * weight_points`, so the weight of the curve is
    the second source of a zero weight, and every point of the curve is then
    a measurement which carries no information.
    """
    problem = on_core(
        zero_problem(tmp_path, "fm_urine", weights=0.0), mapping_weighting()
    )

    assert gaps_of(problem) == []

    curve = fit_request(problem)["curves"][0]
    assert all(noise == float("inf") for noise in curve["noise"])
    core = core_problem(problem)
    x = problem.to_scale(problem.xmodel)
    np.testing.assert_allclose(
        np.asarray(core.residuals(x), dtype=float),
        np.asarray(problem.residuals(x), dtype=float),
        rtol=1e-9,
        atol=1e-12,
    )
    assert core.cost(x) == pytest.approx(0.0)


def two_model_problem(tmp_path: Path) -> OptimizationProblem:
    """Get the initialized problem of the two models of `TwoModelStudy`."""
    (tmp_path / MODEL_FILE).write_text(sbml_pk(urine=True), encoding="utf-8")
    problem = OptimizationProblem(
        "two",
        [
            FitMappingCollection(
                experiment=TwoModelStudy, mappings=["fm_urine", "fm_second"]
            )
        ],
        [
            FitParameter(
                pid="ke",
                lower_bound=0.01,
                upper_bound=1.0,
                start_value=0.5,
                unit="1/hr",
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
        engine=Engine.CORE,
    )
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    return problem


def test_a_problem_of_two_models_is_a_fit_of_the_engine(tmp_path: Path) -> None:
    """A problem whose mappings read two models is a fit of the engine.

    Every condition of the request names the model of its simulation, which
    the engine fits from `MULTI_MODEL_VERSION` on, so the two models of a
    study no longer keep the optimizer in python.
    """
    problem = two_model_problem(tmp_path)

    assert gaps_of(problem) == []


def test_the_condition_of_a_fit_of_two_models_names_its_model(tmp_path: Path) -> None:
    """Every condition of the request names the model of its simulation."""
    problem = two_model_problem(tmp_path)

    request = fit_request(problem)

    models = {condition["model"] for condition in request["conditions"]}
    assert models == set(model_ids(problem))
    assert len(models) == 2
    assert sorted(loaded_models(problem)) == sorted(models)


def test_the_models_of_a_fit_of_one_model_are_the_one_model(tmp_path: Path) -> None:
    """A condition of a fit of one model names none, as every release reads it."""
    problem = on_core(zero_problem(tmp_path, "fm_zero"), error_weighting())

    request = fit_request(problem)

    assert len(loaded_models(problem)) == 1
    assert all("model" not in condition for condition in request["conditions"])


def test_the_residuals_of_a_fit_of_two_models_are_the_ones_of_the_problem(
    tmp_path: Path,
) -> None:
    """The engine computes the residuals of a problem of two models.

    The models differ in the elimination the second one fixes, so the
    residuals of the two conditions are different: a request which ran both
    conditions on one model would answer the same values twice.
    """
    problem = two_model_problem(tmp_path)

    core = core_problem(problem)
    x = problem.to_scale(problem.xmodel)

    residuals = np.asarray(core.residuals(x), dtype=float)

    expected = np.asarray(problem.residuals(x), dtype=float)
    assert residuals.shape == expected.shape
    np.testing.assert_allclose(residuals, expected, rtol=1e-9, atol=1e-12)


def test_an_infinite_weight_is_still_a_gap(tmp_path: Path) -> None:
    """A weight which is an infinity is no fit of the engine.

    Its standard deviation is `0.0`, which no standard deviation of a curve
    can be, so such a point stays a gap while a weight of zero is none.
    """
    problem = zero_problem(tmp_path, "fm_urine", weights=float("inf"))
    problem.initialize(mapping_weighting())

    gaps = gaps_of(problem)

    assert [gap.code for gap in gaps] == ["weights"]
    assert "inf" in gaps[0].message


class PointwiseStudy(UrineStudy):
    """`UrineStudy` with observables which read each other and a declared unit.

    `urine_four` reads the observable `urine_twice` before it, which the
    model of the fit must carry as the parameter of that observable and not
    as a symbol it has not, and `urine_grams` declares a unit the derived one
    converts into, whose factor the column of the fit carries.
    """

    @override
    def observables(self) -> dict[str, Observable]:
        observables = super().observables()
        observables["urine_four"] = Formula("urine_four", "2 * urine_twice")
        observables["urine_grams"] = Formula("urine_grams", "2 * Aurine", unit="g")
        return observables

    @override
    def datasets(self) -> dict[str, DataSet]:
        datasets = super().datasets()
        datasets["four"] = DataSet.from_df(
            pd.DataFrame({"time": TIMES, "Aurine": 4.0 * np.asarray(TRUTH["urine"])}),
            udict={"time": "hr", "Aurine": "mg"},
            ureg=self.ureg,
        )
        datasets["grams"] = DataSet.from_df(
            pd.DataFrame({"time": TIMES, "Aurine": 2e-3 * np.asarray(TRUTH["urine"])}),
            udict={"time": "hr", "Aurine": "g"},
            ureg=self.ureg,
        )
        return datasets

    @override
    def fit_mappings(self) -> dict[str, FitMapping]:
        mappings = super().fit_mappings()
        for key, dataset, observable in (
            ("fm_four", "four", "urine_four"),
            ("fm_grams", "grams", "urine_grams"),
        ):
            mappings[key] = FitMapping(
                self,
                reference=FitData(self, dataset=dataset, xid="time", yid="Aurine"),
                observable=FitData(
                    self, task="task_collected", xid="time", yid=observable
                ),
            )
        return mappings


class ThriceStudy(UrineStudy):
    """`UrineStudy` whose observable `urine_twice` is three times the urine.

    Two experiments of one fit may define an observable of the same id with
    different formulas, which the parameters of the model of the fit must
    keep apart.
    """

    @override
    def observables(self) -> dict[str, Observable]:
        observables = super().observables()
        observables["urine_twice"] = Formula("urine_twice", "3 * Aurine")
        return observables

    @override
    def datasets(self) -> dict[str, DataSet]:
        datasets = super().datasets()
        datasets["twice"] = DataSet.from_df(
            pd.DataFrame({"time": TIMES, "Aurine": 3.0 * np.asarray(TRUTH["urine"])}),
            udict={"time": "hr", "Aurine": "mg"},
            ureg=self.ureg,
        )
        return datasets


def observable_problem(
    tmp_path: Path,
    mappings: Sequence[str],
    experiment: type[SimulationExperiment] = UrineStudy,
    second: type[SimulationExperiment] | None = None,
) -> OptimizationProblem:
    """Get the problem of the mappings of one or of two urine experiments."""
    (tmp_path / MODEL_FILE).write_text(sbml_pk(urine=True), encoding="utf-8")
    collections = [FitMappingCollection(experiment=experiment, mappings=list(mappings))]
    if second is not None:
        collections.append(
            FitMappingCollection(experiment=second, mappings=list(mappings))
        )
    return OptimizationProblem(
        "observables",
        collections,
        [
            FitParameter(
                pid="ke",
                lower_bound=0.01,
                upper_bound=1.0,
                start_value=0.5,
                unit="1/hr",
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
        engine=Engine.CORE,
    )


def linear_settings() -> FitSettings:
    """Get the settings of a fit on the linear scale of its parameters."""
    return FitSettings(parameter_scale=ParameterScaleType.LINEAR)


def test_a_pointwise_formula_observable_is_a_column_and_no_gap(
    tmp_path: Path,
) -> None:
    """An observable of the experiment is a parameter of the model of the fit.

    `urine_twice` is `2 * Aurine`, a timecourse without a reduction, so its
    value at a time of the data does not depend on the other output times
    and the model of the fit carries it as a column.
    """
    problem = on_core(observable_problem(tmp_path, ["fm_twice"]), linear_settings())

    assert gaps_of(problem) == []

    curve = fit_request(problem)["curves"][0]
    assert curve["column"] == fit_core._group_observable(0, "urine_twice")
    assert fit_core.fit_model(problem) is not problem.models[0]


def test_the_residuals_of_a_formula_observable_are_the_ones_of_python(
    tmp_path: Path,
) -> None:
    """The column of the model is the value python computes for the observable."""
    problem = on_core(observable_problem(tmp_path, ["fm_twice"]), linear_settings())
    core = core_problem(problem)

    for factor in (1.0, 1.3, 0.7):
        x = problem.to_scale(problem.xmodel * factor)
        expected = np.asarray(problem.residuals(x), dtype=float)

        residuals = np.asarray(core.residuals(x), dtype=float)

        assert residuals.shape == expected.shape
        np.testing.assert_allclose(residuals, expected, rtol=1e-6, atol=1e-9)


def test_an_observable_which_reads_another_observable_reads_its_parameter(
    tmp_path: Path,
) -> None:
    """The nodes of the graph are ordered, so a formula reads the ones before it.

    `urine_four` is `2 * urine_twice`, which the model of the fit must read
    as the parameter of `urine_twice` and not as a symbol it has not.
    """
    problem = on_core(
        observable_problem(tmp_path, ["fm_four"], experiment=PointwiseStudy),
        linear_settings(),
    )

    assert gaps_of(problem) == []

    curve = fit_request(problem)["curves"][0]
    assert curve["column"] == fit_core._group_observable(0, "urine_four")
    core = core_problem(problem)
    x = problem.to_scale(problem.xmodel)
    np.testing.assert_allclose(
        np.asarray(core.residuals(x), dtype=float),
        np.asarray(problem.residuals(x), dtype=float),
        rtol=1e-6,
        atol=1e-9,
    )


def test_two_groups_whose_observables_share_an_id_do_not_collide(
    tmp_path: Path,
) -> None:
    """Each group reads the formula of its own experiment.

    `ThriceStudy` defines `urine_twice` as `3 * Aurine`, so a column named
    after the observable alone would take one formula for the other.
    """
    problem = on_core(
        observable_problem(
            tmp_path, ["fm_twice"], experiment=UrineStudy, second=ThriceStudy
        ),
        linear_settings(),
    )

    assert gaps_of(problem) == []

    columns = [curve["column"] for curve in fit_request(problem)["curves"]]
    assert len(set(columns)) == len(columns)
    core = core_problem(problem)
    x = problem.to_scale(problem.xmodel)
    np.testing.assert_allclose(
        np.asarray(core.residuals(x), dtype=float),
        np.asarray(problem.residuals(x), dtype=float),
        rtol=1e-6,
        atol=1e-9,
    )


def test_the_factor_of_a_declared_unit_is_in_the_column(tmp_path: Path) -> None:
    """A declared unit is a factor of the formula, which the column carries.

    `urine_grams` is `2 * Aurine` of the model in `mg` declared in `g`, so
    python multiplies its values by `1e-3` and the parameter of the model
    carries the same multiplication.
    """
    problem = on_core(
        observable_problem(tmp_path, ["fm_grams"], experiment=PointwiseStudy),
        linear_settings(),
    )

    assert gaps_of(problem) == []

    core = core_problem(problem)
    for factor in (1.0, 1.3):
        x = problem.to_scale(problem.xmodel * factor)
        np.testing.assert_allclose(
            np.asarray(core.residuals(x), dtype=float),
            np.asarray(problem.residuals(x), dtype=float),
            rtol=1e-6,
            atol=1e-9,
        )


@pytest.mark.parametrize("mapping", ["fm_share", "fm_urine_max"])
def test_a_reduction_and_a_value_per_simulation_are_still_gaps(
    tmp_path: Path, mapping: str
) -> None:
    """A group which keeps the output of its simulation is a gap.

    `urine_share` reads `max(Aurine)` and `fm_urine_max` is a value per
    simulation: both need the output of the simulation and are no column of
    it (`_group_graphs`).
    """
    problem = observable_problem(tmp_path, [mapping])
    problem.initialize(linear_settings())

    gaps = gaps_of(problem)

    assert "observable" in [gap.code for gap in gaps]


def test_the_row_map_of_a_likelihood_is_the_data_rows_of_every_mapping(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The rows of the request sit in the data half of the residuals of each mapping."""
    problem = on_core(
        op_hctz_iv,
        dataclasses.replace(fit_settings, objective=ObjectiveType.LIKELIHOOD),
    )
    sizes = [
        len(problem.y_references[k]) for k in problem.indices(MappingKind.TRAINING)
    ]
    expected = np.concatenate(
        [np.arange(size) + 2 * sum(sizes[:i]) for i, size in enumerate(sizes)]
    )

    np.testing.assert_array_equal(fit_core._residual_rows(problem), expected)
    assert fit_core._data_rows(problem) == 2 * sum(sizes)
    residuals = np.asarray(
        problem.residuals(np.asarray(problem.to_scale(problem.xmodel), dtype=float))
    )
    assert fit_core._data_rows(problem) == residuals.size


def test_the_jacobian_of_a_likelihood_has_zero_rows_for_the_scales(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The scale of a noise the fit does not move is a constant of the data."""
    problem = on_core(
        op_hctz_iv,
        dataclasses.replace(fit_settings, objective=ObjectiveType.LIKELIHOOD),
    )
    x = np.asarray(problem.to_scale(problem.xmodel), dtype=float)

    jac = fit_core.residual_jacobian(problem, x)

    data_rows = fit_core._residual_rows(problem)
    scale_rows = np.setdiff1d(np.arange(jac.shape[0]), data_rows)
    assert jac.shape[0] == fit_core._data_rows(problem)
    assert scale_rows.size == data_rows.size
    np.testing.assert_array_equal(jac[scale_rows], 0.0)
    assert np.any(jac[data_rows] != 0.0)
