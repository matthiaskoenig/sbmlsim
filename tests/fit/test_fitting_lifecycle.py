"""Regression tests for fitting artifacts and independent evaluation sessions."""

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from sbmlsim.fit import FitMappingCollection, FitParameter, FitSettings, ParameterSet
from sbmlsim.fit.fisher import fisher_information
from sbmlsim.fit.identifiability import ProfileSettings, profile_likelihood
from sbmlsim.fit.metrics import FitMetrics
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    OptimizationAlgorithmType,
    ParameterScaleType,
    ResidualType,
)
from sbmlsim.fit.petab_v2.likelihood import gradient, log_likelihood, log_prior
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
from tests.fit.scalar_experiment import (
    TRUE_V,
    BlowupStudy,
    LocalScanStudy,
    ScalarStudy,
)


@pytest.fixture
def scalar_problem(tmp_path: Path) -> OptimizationProblem:
    problem = OptimizationProblem(
        "lifecycle",
        [FitMappingCollection(experiment=ScalarStudy, mappings=["fm_cmax"])],
        [FitParameter("V", 5.0, 1.0, 100.0, unit="l")],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    return problem


def test_units_bind_consistently_across_evaluation_apis(
    scalar_problem: OptimizationProblem,
) -> None:
    problem = scalar_problem
    settings = problem.settings_initialized
    liters = ParameterSet("liters", {"V": 10.0}, {"V": "l"})
    milliliters = ParameterSet("milliliters", {"V": 10000.0}, {"V": "ml"})
    assert FitMetrics(problem, liters).cost() == FitMetrics(problem, milliliters).cost()
    first = fisher_information(problem, settings, liters)
    second = fisher_information(problem, settings, milliliters)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.matrix, second.matrix)
    assert log_likelihood(problem, liters) == log_likelihood(problem, milliliters)
    assert log_prior(problem, liters) == log_prior(problem, milliliters)
    np.testing.assert_array_equal(
        gradient(problem, liters), gradient(problem, milliliters)
    )
    profile = profile_likelihood(
        problem,
        settings,
        milliliters,
        profile_settings=ProfileSettings(reoptimize=False, max_points=4),
        serial=True,
        show_progress=False,
    )
    assert profile.parameter_set.values == {"V": 10.0}
    assert profile.parameter_set.units == {"V": "l"}


def test_model_start_values_use_parameter_units(
    scalar_problem: OptimizationProblem, tmp_path: Path
) -> None:
    problem = scalar_problem
    parameter = problem.parameters[0]
    parameter.unit = "ml"
    parameter.start_value = 10000.0
    parameter.lower_bound = 1000.0
    parameter.upper_bound = 100000.0
    problem.initialize(problem.settings_initialized, force=True)
    model_set = problem.parameter_set_model()
    assert model_set.values == {"V": 10000.0}
    assert model_set.units == {"V": "ml"}
    assert FitMetrics(problem, model_set).cost() == pytest.approx(0.0, abs=1e-12)

    # a target without a unit in the model takes the value as it is, in the
    # model and in the fit
    unitless = OptimizationProblem(
        "unitless",
        [FitMappingCollection(experiment=BlowupStudy, mappings=["fm_cmax"])],
        [
            FitParameter("kb", 0.0, 0.0, 2.0, unit="1/min"),
            FitParameter("B", 1.0, 0.5, 2.0, unit="mg"),
        ],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    unitless.initialize(problem.settings_initialized)
    model_set = unitless.parameter_set_model()
    assert model_set.values == {"kb": 0.0, "B": 1.0}
    assert model_set.units == {"kb": "1/min", "B": "mg"}
    assert FitMetrics(unitless, model_set).cost() == pytest.approx(0.0, abs=1e-9)
    assert [factor for _, _, factor in unitless._group_targets[0]] == [1.0, 1.0]


def test_report_snapshots_inputs_and_reuses_defensive_evaluation_artifact(
    scalar_problem: OptimizationProblem,
) -> None:
    problem = scalar_problem
    original = ParameterSet("fit", {"V": 10000.0}, {"V": "ml"})
    report = FitReport(problem, problem.settings_initialized, original)
    assert report.reference_set.values == {"V": 10.0}
    with patch.object(
        report.problem, "_simulate_groups", wraps=report.problem._simulate_groups
    ) as simulations:
        report.metrics_df()
        report.metrics_mappings_df()
        report.datapoints_df()
        # the given set is the set of the report in other units
        points = report.points(original)
        prediction = float(points.IPRED.iloc[0])
        assert report.metrics(original).cost() == 0.0
        assert simulations.call_count == 1
        points.loc[0, "IPRED"] = -100.0
        data = report.residual_data(original)
        data["y_obsip"][0][0] = -200.0
        assert report.points(original).IPRED.iloc[0] == prediction
        assert report.residual_data(original)["y_obsip"][0][0] == prediction
        assert simulations.call_count == 1
        # an edit of the given set does not reach the report, the edited set
        # is evaluated as it is
        original.values["V"] = 20000.0
        assert report.reference_set.values == {"V": 10.0}
        assert report.points(original).IPRED.iloc[0] == pytest.approx(prediction / 2)
        assert report.points(report.reference_set).IPRED.iloc[0] == prediction
        # an edit of a set of the report replaces its evaluation
        report.reference_set.values["V"] = 20.0
        calls = simulations.call_count
        assert report.points(report.reference_set).IPRED.iloc[0] == pytest.approx(
            prediction / 2
        )
        assert simulations.call_count == calls + 1
        assert len(report._res_data) == len(report._points) == 1
        report.reference_set.values["V"] = 30.0
        report.points(report.reference_set)
        assert len(report._res_data) == len(report._points) == 1


def test_report_evaluates_another_set_of_the_same_id(
    scalar_problem: OptimizationProblem,
) -> None:
    """A set which shares the id of a set of the report, e.g. `fit`, is no set of it."""
    problem = scalar_problem
    report = FitReport(
        problem,
        problem.settings_initialized,
        ParameterSet("fit", {"V": 10.0}, {"V": "l"}),
    )
    prediction = float(report.points(report.reference_set).IPRED.iloc[0])
    other = ParameterSet("fit", {"V": 20000.0}, {"V": "ml"})
    np.testing.assert_array_equal(report.x(other), [20.0])
    assert report.points(other).IPRED.iloc[0] == pytest.approx(prediction / 2)
    assert report.residual_data(other)["y_obsip"][0][0] == pytest.approx(prediction / 2)
    assert report.metrics(other).cost() > report.metrics(report.reference_set).cost()
    assert report.metrics(other).parameter_set.values == {"V": 20000.0}
    # only the sets of the report are kept
    assert len(report._res_data) == len(report._points) == 1
    assert report.points(report.reference_set).IPRED.iloc[0] == prediction


def test_reports_keep_settings_when_other_sessions_reinitialize(
    scalar_problem: OptimizationProblem,
) -> None:
    problem = scalar_problem
    settings = problem.settings_initialized
    pset = ParameterSet("fit", {"V": 5.0}, {"V": "l"})
    first = FitReport(problem, settings, pset)
    absolute_cost = first.metrics(pset).cost()
    normalized = replace(settings, residual=ResidualType.NORMALIZED)
    second = FitReport(problem, normalized, pset)
    assert second.metrics(pset).cost() == pytest.approx(0.5)
    assert problem.settings_initialized == settings
    problem.initialize(normalized)
    assert first.problem.settings_initialized == settings
    assert first.metrics(pset).cost() == absolute_cost
    assert first.problem.runner is not second.problem.runner


def test_failed_initialization_preserves_previous_runtime_and_retry_validation(
    scalar_problem: OptimizationProblem,
) -> None:
    problem = scalar_problem
    settings = problem.settings_initialized
    old_runner = problem.runner
    old_values = problem.xmodel.copy()
    cost = problem.cost_least_square(problem.to_scale([5.0]))
    with (
        patch.object(
            OptimizationProblem, "_compile_plans", side_effect=RuntimeError("late")
        ),
        pytest.raises(RuntimeError, match="late"),
    ):
        problem.initialize(replace(settings, relative_tolerance=1e-5))
    assert problem.settings_initialized == settings
    assert problem.runner is old_runner
    np.testing.assert_array_equal(problem.xmodel, old_values)
    assert problem.cost_least_square(problem.to_scale([5.0])) == cost
    problem.parameters[0].lower_bound = -1.0
    invalid = replace(settings, parameter_scale=ParameterScaleType.LOG10)
    for _ in range(2):
        with pytest.raises(ValueError, match="positive 'lower_bound'"):
            problem.initialize(invalid)
        assert problem.settings_initialized == settings
        assert problem.runner is old_runner


def test_start_updates_and_optimizer_options_survive_result_round_trip(
    scalar_problem: OptimizationProblem,
) -> None:
    problem = scalar_problem
    problem.parameters[0].start_value = 20.0
    assert problem.x0 == [20.0]
    fit, trace = problem.optimize_run(max_nfev=1, diff_step=0.012)
    np.testing.assert_array_equal(fit.x0, [20.0])
    problem.x0 = [15.0]
    assert problem.parameters[0].start_value == 15.0
    result = OptimizationResult(
        problem.parameters, [fit], [trace], settings=problem.settings_initialized
    )
    problem.parameters[0].start_value = 30.0
    fit.x[0] = 30.0
    stored = OptimizationResult.from_json(result.to_json())
    assert stored.parameters[0].start_value == 15.0
    assert stored.fits[0].x[0] == 20.0
    assert stored.fits[0].algorithm == "LEAST_SQUARE"
    assert stored.fits[0].optimizer_kwargs["diff_step"] == 0.012
    assert stored.fits[0].optimizer_kwargs["max_nfev"] == 1
    assert stored.fits[0].optimizer_kwargs["loss"] == "linear"


@pytest.mark.parametrize("options", [{"loss": "soft_l1"}, {"f_scale": 2.0}])
def test_objective_options_fail_before_optimization(
    scalar_problem: OptimizationProblem, options: dict[str, Any]
) -> None:
    problem = scalar_problem
    with pytest.raises(ValueError, match=r"FitSettings\.loss_function"):
        problem.check_optimizer_arguments(
            OptimizationAlgorithmType.LEAST_SQUARE, options
        )
    fit, _ = problem.optimize_run(**options)
    assert not fit.success
    assert "FitSettings.loss_function" in fit.message


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"loss": "soft_l1"}, ValueError, r"FitSettings\.loss_function"),
        ({"f_scale": 2.0}, ValueError, r"FitSettings\.loss_function"),
        ({"maxiter": 3}, TypeError, r"\['maxiter'\]"),
    ],
)
def test_profile_options_fail_before_the_scans(
    scalar_problem: OptimizationProblem,
    options: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    """The reoptimization of a profile takes the arguments a fit takes."""
    problem = scalar_problem
    with pytest.raises(error, match=match):
        profile_likelihood(
            problem,
            problem.settings_initialized,
            ParameterSet("fit", {"V": 10.0}, {"V": "l"}),
            profile_settings=ProfileSettings(optimizer_kwargs=options, max_points=2),
            serial=True,
            show_progress=False,
        )


@pytest.mark.parametrize("algorithm", ["unknown", 123])
def test_unsupported_algorithm_remains_a_failed_run(
    scalar_problem: OptimizationProblem, algorithm: Any
) -> None:
    fit, trace = scalar_problem.optimize_run(algorithm=algorithm)
    assert not fit.success and trace == []
    assert "optimizer is not supported" in fit.message
    assert fit.algorithm == str(algorithm)
    assert fit.optimizer_kwargs == {}


def test_the_start_values_are_the_values_of_the_model_as_loaded(
    tmp_path: Path,
) -> None:
    """A design of an experiment does not set the start values of a fit (#315).

    The local design of `LocalScanStudy` reads its references after the
    pre-initialization of a simulation of twice the volume, on the model
    which the problem reads `xmodel` from.
    """
    problem = OptimizationProblem(
        "xmodel",
        [FitMappingCollection(experiment=LocalScanStudy, mappings=["fm_cmax"])],
        [FitParameter("V", 5.0, 1.0, 100.0, unit="l")],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    np.testing.assert_array_equal(problem.xmodel, [TRUE_V])


def test_the_start_values_are_read_from_the_model_as_loaded(
    scalar_problem: OptimizationProblem,
) -> None:
    """A simulation which left the model pre-initialized does not change `xmodel` (#315)."""
    problem = scalar_problem
    model = problem.models[0]
    Simulator(n_workers=1).run(
        model, Simulation(end=1, steps=2, preinit_changes={"V": 3 * TRUE_V})
    )
    assert model.r_loaded["V"] == 3 * TRUE_V
    problem._store_model_parameters()
    np.testing.assert_array_equal(problem.xmodel, [TRUE_V])
