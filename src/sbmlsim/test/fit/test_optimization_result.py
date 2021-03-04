from pathlib import Path

from sbmlsim.examples.experiments.midazolam import fitting_parallel
from sbmlsim.examples.experiments.midazolam.fitting_problems import op_mid1oh_iv
from sbmlsim.fit.analysis import OptimizationResult
from sbmlsim.fit.optimization import FittingType, ResidualType, WeightingLocalType

fit_kwargs_default = {
    "fitting_type": FittingType.ABSOLUTE_VALUES,
    "residual_type": ResidualType.ABSOLUTE_NORMED_RESIDUALS,
    "weighting_local": WeightingLocalType.ABSOLUTE_ONE_OVER_WEIGHTING,
    "absolute_tolerance": 1e-6,
    "relative_tolerance": 1e-6,
}


def test_serialization(tmp_path: Path) -> None:
    """Test serialization of optimization result."""
    opt_res: OptimizationResult
    opt_res, _ = fitting_parallel.fit_lsq(
        problem_factory=op_mid1oh_iv, **fit_kwargs_default
    )
    assert opt_res

    opt_res_path = tmp_path / "opt_res.json"
    opt_res.to_json(path=opt_res_path)
    opt_res2 = OptimizationResult.from_json(json_info=opt_res_path)

    assert opt_res.sid == opt_res2.sid
    assert [p.pid for p in opt_res.parameters] == [p.pid for p in opt_res2.parameters]


def test_combine(tmp_path: Path) -> None:
    """Test combination of optimization result."""
    opt_res1, _ = fitting_parallel.fit_lsq(
        problem_factory=op_mid1oh_iv, seed=1234, **fit_kwargs_default
    )
    opt_res2, _ = fitting_parallel.fit_lsq(
        problem_factory=op_mid1oh_iv, seed=5678, **fit_kwargs_default
    )
    opt_res = OptimizationResult.combine([opt_res1, opt_res2])
    assert len(opt_res.fits) == len(opt_res1.fits) + len(opt_res2.fits)

