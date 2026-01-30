"""Test fit."""

from pathlib import Path
from typing import Any

import pytest

from sbmlsim.examples.experiments.midazolam.fitting_problems import op_mid1oh_iv
from sbmlsim.fit.analysis import OptimizationAnalysis
from sbmlsim.fit.options import (
    LossFunctionType,
    OptimizationAlgorithmType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization


fit_kwargs_testdata = []
for residual_type in [
    ResidualType.ABSOLUTE,
    ResidualType.NORMALIZED,
    # ResidualType.ABSOLUTE_TO_BASELINE,
    # ResidualType.NORMALIZED_TO_BASELINE,
]:
    for weighting_curves in [
        [],
        [WeightingCurvesType.POINTS],
        [WeightingCurvesType.MAPPING],
        [WeightingCurvesType.POINTS, WeightingCurvesType.MAPPING],
    ]:
        for weighting_points in [
            WeightingPointsType.NO_WEIGHTING,
            WeightingPointsType.ERROR_WEIGHTING,
        ]:
            fit_kwargs_testdata.append(
                {
                    "residual": residual_type,
                    "weighting_curves": weighting_curves,
                    "weighting_points": weighting_points,
                    "absolute_tolerance": 1e-6,
                    "relative_tolerance": 1e-6,
                }
            )


@pytest.mark.skip(reason="no fit support")
@pytest.mark.parametrize("fit_kwargs", fit_kwargs_testdata)
def test_fit_settings(fit_kwargs: dict[str, Any]) -> None:
    """Test various arguments to optimization problem."""
    op = op_mid1oh_iv()
    opt_result: OptimizationResult = run_optimization(
        problem=op,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=True,
        **fit_kwargs,
    )

    assert opt_result is not None
    assert op.residual == fit_kwargs["residual"]
    assert op.weighting_curves == fit_kwargs["weighting_curves"]
    assert op.weighting_points == fit_kwargs["weighting_points"]


fit_kwargs_default = {
    "residual": ResidualType.NORMALIZED,
    "weighting_curves": [WeightingCurvesType.POINTS],
    "weighting_points": WeightingPointsType.ERROR_WEIGHTING,
    "absolute_tolerance": 1e-6,
    "relative_tolerance": 1e-6,
}


@pytest.mark.skip(reason="no fit support")
def test_optimization_analysis(tmp_path: Path) -> None:
    """Test optimization analysis."""
    op = op_mid1oh_iv()
    opt_result: OptimizationResult = run_optimization(
        problem=op,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        **fit_kwargs_default,
    )
    op_analysis = OptimizationAnalysis(
        opt_result=opt_result,
        output_dir=tmp_path,
        output_name="tests",
        op=op,
        show_plots=False,
        **fit_kwargs_default,
    )
    op_analysis.run()


@pytest.mark.skip(reason="no fit support")
@pytest.mark.parametrize(
    "loss_function",
    [
        LossFunctionType.LINEAR,
        LossFunctionType.SOFT_L1,
        LossFunctionType.CAUCHY,
        LossFunctionType.ARCTAN,
    ],
)
def test_loss_function(loss_function: LossFunctionType) -> None:
    """Test the various loss functions."""
    op = op_mid1oh_iv()
    opt_result: OptimizationResult = run_optimization(
        problem=op,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        loss_function=loss_function,
        size=1,
        n_cores=1,
        serial=True,
        **fit_kwargs_default,
    )
    assert opt_result
    assert op.loss_function == loss_function


@pytest.mark.skip(reason="no fit support")
def test_fit_lsq_serial() -> None:
    """Test serial least square fit."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_mid1oh_iv(),
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=True,
        **fit_kwargs_default,
    )
    assert opt_result is not None


@pytest.mark.skip(reason="no fit support")
def test_fit_de_serial() -> None:
    """Test serial differential evolution fit."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_mid1oh_iv(),
        algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
        size=1,
        n_cores=1,
        serial=True,
        **fit_kwargs_default,
    )
    assert opt_result is not None


@pytest.mark.skip(reason="no fit support")
def test_fit_lsq_parallel() -> None:
    """Test parallel least square fit."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_mid1oh_iv(),
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=False,
        **fit_kwargs_default,
    )
    assert opt_result is not None


@pytest.mark.skip(reason="no fit support")
def test_fit_de_parallel():
    """Test parallel differential evolution fit."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_mid1oh_iv(),
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=False,
        **fit_kwargs_default,
    )
    assert opt_result is not None
