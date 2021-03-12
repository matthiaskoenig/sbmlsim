from typing import Dict, Any

import pytest

from sbmlsim.examples.experiments.midazolam.fitting_problems import (
    op_mid1oh_iv,
)
from sbmlsim.fit.analysis import OptimizationAnalysis
from sbmlsim.fit.options import (
    FittingStrategyType, ResidualType, WeightingPointsType, OptimizationAlgorithmType
)
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization

fit_kwargs_testdata = [
    # fitting_strategy
    {
        "fitting_strategy": FittingStrategyType.ABSOLUTE_VALUES,
        "residual_type": ResidualType.ABSOLUTE_NORMED_RESIDUALS,
        "weighting_points": WeightingPointsType.ABSOLUTE_ONE_OVER_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    },
    {
        "fitting_strategy": FittingStrategyType.ABSOLUTE_CHANGES_BASELINE,
        "residual_type": ResidualType.ABSOLUTE_NORMED_RESIDUALS,
        "weighting_points": WeightingPointsType.ABSOLUTE_ONE_OVER_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    },
    {
        "fitting_strategy": FittingStrategyType.RELATIVE_CHANGES_BASELINE,
        "residual_type": ResidualType.ABSOLUTE_NORMED_RESIDUALS,
        "weighting_points": WeightingPointsType.ABSOLUTE_ONE_OVER_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    },
    # residual_type
    {
        "fitting_strategy": FittingStrategyType.ABSOLUTE_VALUES,
        "residual_type": ResidualType.ABSOLUTE_RESIDUALS,
        "weighting_points": WeightingPointsType.ABSOLUTE_ONE_OVER_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    },
    {
        "fitting_strategy": FittingStrategyType.ABSOLUTE_VALUES,
        "residual_type": ResidualType.RELATIVE_RESIDUALS,
        "weighting_points": WeightingPointsType.ABSOLUTE_ONE_OVER_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    },
    # weighing local FIXME: rename to weighting_points
    #     NO_WEIGHTING
    #     ABSOLUTE_ONE_OVER_WEIGHTING
    #     RELATIVE_ONE_OVER_WEIGHTING
    {
        "fitting_strategy": FittingStrategyType.ABSOLUTE_VALUES,
        "residual_type": ResidualType.ABSOLUTE_NORMED_RESIDUALS,
        "weighting_points": WeightingPointsType.NO_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    },
    {
        "fitting_strategy": FittingStrategyType.ABSOLUTE_VALUES,
        "residual_type": ResidualType.ABSOLUTE_NORMED_RESIDUALS,
        "weighting_points": WeightingPointsType.RELATIVE_ONE_OVER_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    },
]


@pytest.mark.parametrize("fit_kwargs", fit_kwargs_testdata)
def test_fit_settings(fit_kwargs: Dict[str, Any]) -> None:
    """Test various arguments to optimization problem."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_mid1oh_iv(),
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        **fit_kwargs
    )
    assert opt_result is not None


fit_kwargs_default = {
    "fitting_strategy": FittingStrategyType.ABSOLUTE_VALUES,
    "residual_type": ResidualType.ABSOLUTE_NORMED_RESIDUALS,
    "weighting_points": WeightingPointsType.ABSOLUTE_ONE_OVER_WEIGHTING,
    "absolute_tolerance": 1e-6,
    "relative_tolerance": 1e-6,
}


def test_optimization_analysis(tmp_path):
    op = op_mid1oh_iv()
    opt_result: OptimizationResult = run_optimization(
        problem=op,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        **fit_kwargs_default
    )
    op_analysis = OptimizationAnalysis(
        opt_result=opt_result,
        output_path=tmp_path,
        op=op,
        show_plots=False,
        **fit_kwargs_default
    )
    op_analysis.analyse()


def test_fit_lsq_serial() -> None:
    """Test serial least square fit."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_mid1oh_iv(),
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=True,
        **fit_kwargs_default
    )
    assert opt_result is not None


def test_fit_de_serial() -> None:
    """Test serial differential evolution fit."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_mid1oh_iv(),
        algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
        size=1,
        n_cores=1,
        serial=True,
        **fit_kwargs_default
    )
    assert opt_result is not None


def test_fit_lsq_parallel() -> None:
    """Test parallel least square fit."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_mid1oh_iv(),
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=False,
        **fit_kwargs_default
    )
    assert opt_result is not None


def test_fit_de_parallel():
    """Test parallel differential evolution fit."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_mid1oh_iv(),
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=False,
        **fit_kwargs_default
    )
    assert opt_result is not None
