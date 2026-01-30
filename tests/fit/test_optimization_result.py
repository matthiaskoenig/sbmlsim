"""Test optimization results."""
from pathlib import Path

import pytest

from sbmlsim.examples.experiments.midazolam.fitting_problems import op_mid1oh_iv
from sbmlsim.fit.analysis import OptimizationResult
from sbmlsim.fit.options import (
    OptimizationAlgorithmType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)
from sbmlsim.fit.runner import run_optimization


fit_kwargs_default = {
    "residual": ResidualType.ABSOLUTE,
    "weighting_curves": [WeightingCurvesType.POINTS],
    "weighting_points": WeightingPointsType.ERROR_WEIGHTING,
    "absolute_tolerance": 1e-6,
    "relative_tolerance": 1e-6,
}


@pytest.mark.skip(reason="no fit support")
def test_serialization(tmp_path: Path) -> None:
    """Test serialization of optimization result."""
    opt_res: OptimizationResult = run_optimization(
        problem=op_mid1oh_iv(),
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=True,
        **fit_kwargs_default
    )

    opt_res_path = tmp_path / "opt_res.json"
    opt_res.to_json(path=opt_res_path)
    opt_res2 = OptimizationResult.from_json(json_info=opt_res_path)

    assert opt_res.sid == opt_res2.sid
    assert [p.pid for p in opt_res.parameters] == [p.pid for p in opt_res2.parameters]


@pytest.mark.skip(reason="no fit support")
def test_combine(tmp_path: Path) -> None:
    """Test combination of optimization result."""
    opt_results = []
    for seed in [1234, 4567]:
        opt_res: OptimizationResult = run_optimization(
            problem=op_mid1oh_iv(),
            algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
            size=1,
            n_cores=1,
            serial=True,
            seed=seed,
            **fit_kwargs_default
        )
        opt_results.append(opt_res)

    opt_result = OptimizationResult.combine(opt_results)
    assert len(opt_result.fits) == len(opt_results[0].fits) + len(opt_results[1].fits)
