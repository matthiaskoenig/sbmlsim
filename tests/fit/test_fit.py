"""Test fit."""

from pathlib import Path
from typing import Any

import pytest

from sbmlsim.fit.analysis import OptimizationAnalysis
from sbmlsim.fit.optimization import OptimizationProblem
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


@pytest.mark.parametrize("fit_kwargs", fit_kwargs_testdata)
def test_fit_settings(
    fit_kwargs: dict[str, Any], op_hctz_pkiv: OptimizationProblem
) -> None:
    """Test various arguments to optimization problem."""
    op = op_hctz_pkiv
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


def test_optimization_analysis(
    tmp_path: Path,
    op_hctz_pkiv: OptimizationProblem,
    fit_kwargs_default: dict[str, Any],
) -> None:
    """Test optimization analysis."""
    op = op_hctz_pkiv
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

    results_dir = tmp_path / opt_result.sid / "tests"
    assert (results_dir / "index.html").exists()
    assert (results_dir / "report.txt").exists()
    assert (results_dir / "optimization_result.json").exists()
    assert (results_dir / "optimization_result.tsv").exists()
    assert list((results_dir / "plots").glob("*.svg"))


def test_optimization_analysis_rerun(
    tmp_path: Path,
    op_hctz_pkiv: OptimizationProblem,
    fit_kwargs_default: dict[str, Any],
) -> None:
    """A problem which was already initialized keeps its mappings."""
    op = op_hctz_pkiv
    opt_result: OptimizationResult = run_optimization(
        problem=op,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=True,
        **fit_kwargs_default,
    )
    n_mappings = len(op.mapping_keys)
    assert n_mappings > 0

    OptimizationAnalysis(
        opt_result=opt_result,
        output_dir=tmp_path,
        output_name="tests",
        op=op,
        show_plots=False,
        **fit_kwargs_default,
    )
    assert len(op.mapping_keys) == n_mappings


@pytest.mark.parametrize(
    "loss_function",
    [
        LossFunctionType.LINEAR,
        LossFunctionType.SOFT_L1,
        LossFunctionType.CAUCHY,
        LossFunctionType.ARCTAN,
    ],
)
def test_loss_function(
    loss_function: LossFunctionType,
    op_hctz_pkiv: OptimizationProblem,
    fit_kwargs_default: dict[str, Any],
) -> None:
    """Test the various loss functions."""
    op = op_hctz_pkiv
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
    # the loss functions are finite on signed residuals
    assert all(fit.cost >= 0.0 for fit in opt_result.fits)
    assert opt_result.df_fits.cost.notna().all()


def test_fit_lsq_serial(
    op_hctz_pkiv: OptimizationProblem, fit_kwargs_default: dict[str, Any]
) -> None:
    """Test serial least square fit."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_hctz_pkiv,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=True,
        **fit_kwargs_default,
    )
    assert opt_result is not None


def test_fit_de_serial(
    op_hctz_pkiv: OptimizationProblem, fit_kwargs_default: dict[str, Any]
) -> None:
    """Test serial differential evolution fit."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_hctz_pkiv,
        algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
        size=1,
        n_cores=1,
        serial=True,
        maxiter=2,
        **fit_kwargs_default,
    )
    assert opt_result is not None


def test_fit_lsq_parallel(
    op_hctz_pkiv: OptimizationProblem, fit_kwargs_default: dict[str, Any]
) -> None:
    """Test parallel least square fit."""
    opt_result: OptimizationResult = run_optimization(
        problem=op_hctz_pkiv,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=False,
        **fit_kwargs_default,
    )
    assert opt_result is not None
