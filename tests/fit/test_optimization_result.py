"""Test optimization results."""

from pathlib import Path
from typing import Any

import numpy as np

from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import OptimizationAlgorithmType
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization


def test_serialization(
    tmp_path: Path,
    op_hctz_pkiv: OptimizationProblem,
    fit_kwargs_default: dict[str, Any],
) -> None:
    """Test serialization of optimization result."""
    opt_res: OptimizationResult = run_optimization(
        problem=op_hctz_pkiv,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=True,
        **fit_kwargs_default,
    )

    opt_res_path = tmp_path / "opt_res.json"
    opt_res.to_json(path=opt_res_path)
    opt_res2 = OptimizationResult.from_json(json_info=opt_res_path)

    assert opt_res.sid == opt_res2.sid
    assert [p.pid for p in opt_res.parameters] == [p.pid for p in opt_res2.parameters]

    # the parameter vectors survive the round trip as arrays
    assert isinstance(opt_res2.xopt, np.ndarray)
    assert np.allclose(opt_res.xopt, opt_res2.xopt)
    for fit in opt_res2.fits:
        assert isinstance(fit.x, np.ndarray)
        assert isinstance(fit.x0, np.ndarray)


def test_combine(
    op_hctz_pkiv: OptimizationProblem, fit_kwargs_default: dict[str, Any]
) -> None:
    """Test combination of optimization result."""
    opt_results = []
    for seed in [1234, 4567]:
        opt_res: OptimizationResult = run_optimization(
            problem=op_hctz_pkiv,
            algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
            size=1,
            n_cores=1,
            serial=True,
            seed=seed,
            **fit_kwargs_default,
        )
        opt_results.append(opt_res)

    opt_result = OptimizationResult.combine(opt_results)
    assert len(opt_result.fits) == len(opt_results[0].fits) + len(opt_results[1].fits)
