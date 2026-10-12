"""Tests of the evaluation of an optimization problem."""

import functools
import os
import pickle
import time
import warnings
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scipy
from pint.errors import UndefinedUnitError

from sbmlsim.data import DataSet
from sbmlsim.fit import FitMappingCollection, FitParameter, FitSettings, MappingKind
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.optimization import OptimizationProblem, apply_loss_function
from sbmlsim.fit.options import (
    LossFunctionType,
    OptimizationAlgorithmType,
    ParameterScaleType,
    ResidualType,
)
from sbmlsim.model import AbstractModel
from sbmlsim.simulator.worker import SUNDIALS_LOGS
from tests.fit.scalar_experiment import CMAX_UNIT, TRUE_V, ScalarStudy
from tests.simulator.models import sbml_pk


def test_soft_l1_does_not_report_a_false_perfect_fit() -> None:
    """Small nonzero residuals remain visible to the optimizer."""
    from scipy.optimize import least_squares

    fit = least_squares(
        lambda x: apply_loss_function(x, LossFunctionType.SOFT_L1),
        np.array([1e-8]),
        gtol=1e-14,
        ftol=1e-14,
        xtol=1e-14,
    )
    assert fit.success
    assert fit.x[0] == pytest.approx(0.0, abs=1e-16)
    values = np.array([-1e-300, 1e-8, 1e300])
    transformed = apply_loss_function(values, LossFunctionType.SOFT_L1)
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed[:2], values[:2], rtol=1e-12, atol=0.0)
    assert transformed[2] == pytest.approx(np.sqrt(2.0) * 1e150)


def test_soft_l1_keeps_an_infinite_residual_infinite() -> None:
    """The loss of a large residual is `sign(r) * sqrt(2 |r|)`, infinite at infinity."""
    values = np.array([-np.inf, np.inf, np.nan, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformed = apply_loss_function(values, LossFunctionType.SOFT_L1)
    assert transformed[0] == -np.inf
    assert transformed[1] == np.inf
    assert np.isnan(transformed[2])
    assert transformed[3] == pytest.approx(np.sqrt(2.0 * (np.sqrt(5.0) - 1.0)))


@pytest.mark.parametrize("timeout", [None, 1e-9])
def test_differential_evolution_needs_bounds_without_start_values(
    tmp_path: Path, timeout: float | None
) -> None:
    """The global optimizer draws its population from finite bounds."""
    problem = OptimizationProblem(
        "global",
        [FitMappingCollection(experiment=ScalarStudy, mappings=["fm_cmax"])],
        [FitParameter("V", None, 1.0, 100.0, unit="l")],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    fits, _ = problem.optimize(
        size=1,
        algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
        seed=1,
        maxiter=2,
        popsize=5,
        polish=False,
        timeout=timeout,
    )
    fit = fits[0]
    assert "start_value" not in fit.message
    assert fit.x0 is None
    if timeout is None:
        assert np.isfinite(fit.cost)
        assert fit.x is not None and 1.0 <= fit.x[0] <= 100.0
    else:
        assert not fit.success and "FitTimeout" in fit.message


@pytest.mark.parametrize(
    ("algorithm", "optimizer", "options"),
    [
        (OptimizationAlgorithmType.LEAST_SQUARE, "least_squares", {}),
        (
            OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
            "differential_evolution",
            {"seed": 1, "maxiter": 2, "popsize": 5, "polish": False},
        ),
    ],
)
def test_an_optimum_found_within_the_timeout_is_kept(
    tmp_path: Path,
    algorithm: OptimizationAlgorithmType,
    optimizer: str,
    options: dict[str, Any],
) -> None:
    """The check of the optimum is no part of the budget of the run."""
    problem = OptimizationProblem(
        "deadline",
        [FitMappingCollection(experiment=ScalarStudy, mappings=["fm_cmax"])],
        [FitParameter("V", 5.0, 1.0, 100.0, unit="l")],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    optimize = getattr(scipy.optimize, optimizer)

    @functools.wraps(optimize)
    def converged_at_the_deadline(*args: Any, **kwargs: Any) -> Any:
        """Run the optimizer, which then finishes as the timeout passes."""
        result = optimize(*args, **kwargs)
        problem._deadline = time.monotonic() - 1.0
        return result

    with patch.object(scipy.optimize, optimizer, converged_at_the_deadline):
        fit, _ = problem.optimize_run(algorithm=algorithm, timeout=60.0, **options)
    assert "FitTimeout" not in fit.message
    assert np.isfinite(fit.cost)


class ZeroBlowupStudy(ScalarStudy):
    """Zero measurements on a model whose integration fails for kb=1."""

    def models(self) -> dict[str, AbstractModel | Path]:
        return {"m": AbstractModel(source=sbml_pk(blowup=True))}

    def datasets(self) -> dict[str, DataSet]:
        return {
            "tab": DataSet.from_df(
                pd.DataFrame({"cmax": [0.0], "cmax_sd": [0.1]}),
                udict={"cmax": CMAX_UNIT},
                ureg=self.ureg,
            )
        }


def test_failed_simulations_with_zero_data_cannot_win_a_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real integration failures never return a perfect zero-data fit."""
    for name in SUNDIALS_LOGS:
        monkeypatch.setenv(name, os.devnull)
    problem = OptimizationProblem(
        "zero",
        [FitMappingCollection(experiment=ZeroBlowupStudy, mappings=["fm_cmax"])],
        [FitParameter("kb", 1.0, 0.0, 2.0, unit="dimensionless")],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    working = problem.cost_least_square(np.array([0.0]))
    failed = problem.residuals(np.array([1.0]))
    assert isinstance(failed, np.ndarray)
    assert np.isfinite(failed).all() and np.all(failed > 0.0)
    assert problem.cost_least_square(np.array([1.0])) == np.inf
    assert problem._best is not None and problem._best[1] == working
    fit, _ = problem.optimize_run(x0=np.array([1.0]), max_nfev=3)
    assert not fit.success
    assert fit.cost == np.inf


def test_zero_mean_normalized_data_is_rejected(tmp_path: Path) -> None:
    """Normalized residuals need a nonzero reference mean."""
    problem = OptimizationProblem(
        "zero",
        [FitMappingCollection(experiment=ZeroBlowupStudy, mappings=["fm_cmax"])],
        [FitParameter("kb", 0.0, 0.0, 2.0, unit="dimensionless")],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    with pytest.raises(ValueError, match="nonzero mean"):
        problem.initialize(
            FitSettings(
                parameter_scale=ParameterScaleType.LINEAR,
                residual=ResidualType.NORMALIZED,
            )
        )


def test_zero_mean_data_which_is_not_fitted_is_accepted(tmp_path: Path) -> None:
    """Only the training data enters the cost, validation data may have a zero mean."""
    problem = OptimizationProblem(
        "zero_validation",
        [
            FitMappingCollection(experiment=ScalarStudy, mappings=["fm_cmax"]),
            FitMappingCollection(
                experiment=ZeroBlowupStudy,
                mappings=["fm_cmax"],
                kind=MappingKind.VALIDATION,
            ),
        ],
        [FitParameter("V", 5.0, 1.0, 100.0, unit="l")],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(
        FitSettings(
            parameter_scale=ParameterScaleType.LINEAR,
            residual=ResidualType.NORMALIZED,
        )
    )
    assert problem.mapping_kinds == [MappingKind.TRAINING, MappingKind.VALIDATION]
    assert problem.cost_least_square(np.array([TRUE_V])) == pytest.approx(
        0.0, abs=1e-12
    )
    data = problem.residuals(np.array([TRUE_V]), complete_data=True)
    assert isinstance(data, dict)
    assert np.isnan(data["residuals"][1]).all()


@pytest.mark.parametrize("variable_step_size", [True, False])
def test_the_residuals_do_not_depend_on_earlier_evaluations(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    variable_step_size: bool,
) -> None:
    """The first evaluation of a problem equals every later one, bit for bit.

    The absolute tolerance of the simulator is scaled by the volumes of the
    compartments; the volumes of the state after a simulation must not change
    the tolerance of the next one.
    """
    problem = op_hctz_pk
    problem.initialize(replace(fit_settings, variable_step_size=variable_step_size))
    x0 = problem.to_scale(problem.x0)
    residuals = [np.asarray(problem.residuals(x0), dtype=float) for _ in range(3)]
    np.testing.assert_array_equal(residuals[0], residuals[1])
    np.testing.assert_array_equal(residuals[1], residuals[2])


def test_the_initial_time_step_reaches_the_models(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The initial time step of the settings is the one of every model of a fit."""
    problem = op_hctz_pk
    problem.initialize(replace(fit_settings, initial_time_step=1e-9))
    for model in problem.models:
        integrator = model.r_loaded.getIntegrator()
        assert integrator.getValue("initial_time_step") == pytest.approx(1e-9)


def test_a_parameter_whose_target_is_not_an_entity(
    definition_hctz_pk: FitDefinition, fit_settings: FitSettings
) -> None:
    """A misspelled target is named, not a symbol roadrunner does not know."""
    typo = FitParameter(
        pid="Ka_dis",
        start_value=1.0,
        lower_bound=1e-4,
        upper_bound=10,
        unit="1/hr",
        target="Ka_dis_hctzz",
    )
    definition = replace(
        definition_hctz_pk, parameters=[*definition_hctz_pk.parameters[1:], typo]
    )
    problem = definition.problem(opid="hctz_typo")
    with pytest.raises(
        ValueError,
        match=r"'hctz_typo': FitParameter 'Ka_dis' writes 'Ka_dis_hctzz', which "
        r"is not an entity of the model .*'sciml:<id>'",
    ):
        problem.initialize(fit_settings)


def test_the_unit_of_a_parameter_is_a_unit_id_of_its_model(
    definition_hctz_pk: FitDefinition, fit_settings: FitSettings
) -> None:
    """A unit id of the model converts as its expression (#268).

    `per_min` of the HCTZ model is `1/min`; the model has `Ka_dis_hctz` in
    `1/hr`, so the value of the parameter is converted by the factor 60.
    """
    costs: dict[str, float] = {}
    for k, unit in enumerate(["per_min", "1/min", "1/hr"]):
        ka = FitParameter(
            pid="Ka_dis_hctz",
            start_value=1.0,
            lower_bound=1e-4,
            upper_bound=100,
            unit=unit,
        )
        definition = replace(
            definition_hctz_pk, parameters=[ka, *definition_hctz_pk.parameters[1:]]
        )
        problem = definition.problem(opid=f"hctz_unit_{k}")
        problem.initialize(fit_settings)
        costs[unit] = problem.cost_least_square(problem.to_scale(problem.x0))

    assert costs["per_min"] == pytest.approx(costs["1/min"], rel=1e-12)
    assert costs["per_min"] != pytest.approx(costs["1/hr"])


def test_a_unit_of_a_parameter_which_is_no_unit_names_the_parameter(
    definition_hctz_pk: FitDefinition, fit_settings: FitSettings
) -> None:
    """A unit which is neither of pint nor of the model is an error of pint with a note."""
    ka = FitParameter(
        pid="Ka_dis_hctz",
        start_value=1.0,
        lower_bound=1e-4,
        upper_bound=10,
        unit="per_mn",
    )
    definition = replace(
        definition_hctz_pk, parameters=[ka, *definition_hctz_pk.parameters[1:]]
    )
    problem = definition.problem(opid="hctz_no_unit")
    with pytest.raises(UndefinedUnitError) as info:
        problem.initialize(fit_settings)
    notes = "\n".join(info.value.__notes__)
    assert "'Ka_dis_hctz' in 'per_mn' writes 'Ka_dis_hctz'" in notes


def test_the_start_values_of_a_problem_are_its_own(
    definition_hctz_pk: FitDefinition,
) -> None:
    """Setting `x0` of a problem, e.g. to restart from an optimum, changes no other.

    The problems of a definition and the definition itself keep their start
    values; a pickled problem keeps the start values it was given.
    """
    definition = replace(
        definition_hctz_pk, parameters=deepcopy(definition_hctz_pk.parameters)
    )
    first = definition.problem(opid="first")
    second = definition.problem(opid="second")
    start = list(second.x0)
    doubled = [2.0 * value for value in start if value is not None]
    assert len(doubled) == len(start)
    first.x0 = doubled
    assert first.x0 == doubled
    assert second.x0 == start
    assert [p.start_value for p in definition.parameters] == start
    assert pickle.loads(pickle.dumps(first)).x0 == first.x0
