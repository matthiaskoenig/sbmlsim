"""Test optimization results."""

from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeResult

from sbmlsim.fit import FitParameter, FitSettings
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import OptimizationAlgorithmType, ParameterScaleType
from sbmlsim.fit.result import OptimizationResult, bound_warnings
from sbmlsim.fit.runner import run_optimization


def test_serialization(
    tmp_path: Path,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
) -> None:
    """Test serialization of optimization result."""
    opt_res: OptimizationResult = run_optimization(
        problem=op_hctz_pk,
        settings=fit_settings,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=1,
        n_cores=1,
        serial=True,
    )

    opt_res_path = tmp_path / "opt_res.json"
    opt_res.to_json(path=opt_res_path)
    opt_res2 = OptimizationResult.from_json(json_info=opt_res_path)

    assert opt_res.sid == opt_res2.sid
    assert [p.pid for p in opt_res.parameters] == [p.pid for p in opt_res2.parameters]

    # the settings of the fit survive the round trip
    assert opt_res2.settings == fit_settings
    assert opt_res2.opid == op_hctz_pk.opid

    # the parameter vectors survive the round trip as arrays
    assert isinstance(opt_res2.xopt, np.ndarray)
    assert np.allclose(opt_res.xopt, opt_res2.xopt)
    for fit in opt_res2.fits:
        assert isinstance(fit.x, np.ndarray)
        assert isinstance(fit.x0, np.ndarray)


def test_parameter_sets(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The result provides the fitted parameters as parameter sets."""
    opt_res: OptimizationResult = run_optimization(
        problem=op_hctz_pk,
        settings=fit_settings,
        size=2,
        n_cores=1,
        serial=True,
        seed=1234,
    )

    pset = opt_res.parameter_set()
    assert set(pset.values) == {p.pid for p in opt_res.parameters}
    assert pset.cost == opt_res.df_fits.cost.iloc[0]
    assert np.allclose(pset.x(op_hctz_pk.pids), opt_res.xopt)

    psets = opt_res.parameter_sets(size=2)
    assert len(psets) == 2
    # the sets are ordered by increasing cost
    costs = [pset.cost for pset in psets]
    assert all(cost is not None for cost in costs)
    assert costs == sorted(costs)  # ty: ignore[invalid-argument-type]


def test_combine(op_hctz_pk: OptimizationProblem, fit_settings: FitSettings) -> None:
    """Test combination of optimization result."""
    opt_results = []
    for seed in [1234, 4567]:
        opt_res: OptimizationResult = run_optimization(
            problem=op_hctz_pk,
            settings=fit_settings,
            algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
            size=1,
            n_cores=1,
            serial=True,
            seed=seed,
        )
        opt_results.append(opt_res)

    opt_result = OptimizationResult.combine(opt_results)
    assert len(opt_result.fits) == len(opt_results[0].fits) + len(opt_results[1].fits)
    # the combined result keeps the settings of the fits
    assert opt_result.settings == fit_settings


def test_the_fit_parameters_of_the_optimum_keep_their_scale_and_selector() -> None:
    """A fit parameter of the optimum is the parameter with another start value."""
    parameters = [
        FitParameter(
            "w", 0.0, -5.0, 5.0, unit="dimensionless", scale=ParameterScaleType.LINEAR
        ),
        FitParameter("k", 1.0, 0.1, 10.0, unit="1/min", mappings=_is_first),
    ]
    result = OptimizationResult(
        parameters=parameters,
        fits=[
            OptimizeResult(
                x=np.array([-4.9, 2.0]),
                cost=1.0,
                status=1,
                success=True,
                duration=0.1,
                x0=None,
            )
        ],
        trajectories=[[1.0]],
    )
    w, k = result.xopt_fit_parameters
    assert (w.start_value, w.scale, w.mappings) == (
        -4.9,
        ParameterScaleType.LINEAR,
        None,
    )
    assert (k.start_value, k.scale, k.mappings) == (2.0, None, _is_first)


def _is_first(key: str, mapping: object) -> bool:
    return key == "first"


def test_the_distance_to_a_bound_is_measured_in_the_scale_of_the_parameter() -> None:
    """A positive parameter on the linear scale is near its bound linearly."""
    parameters = [FitParameter("p", 1.0, 1.0, 1000.0, unit="mM")]
    # 1.2 is 0.03 % of the linear interval, 2.6 % of the logarithmic one
    x = np.array([1.2])
    assert bound_warnings(parameters, x, scales=[ParameterScaleType.LOG10]) == [
        "!Optimal parameter 'p' within 5% of lower bound!"
    ]
    assert bound_warnings(
        parameters, np.array([30.0]), scales=[ParameterScaleType.LINEAR]
    ) == ["!Optimal parameter 'p' within 5% of lower bound!"]
    # 30 is 49 % of the logarithmic interval
    assert (
        bound_warnings(parameters, np.array([30.0]), scales=[ParameterScaleType.LOG10])
        == []
    )

    result = OptimizationResult(
        parameters=[
            FitParameter(
                "p", 1.0, 1.0, 1000.0, unit="mM", scale=ParameterScaleType.LINEAR
            )
        ],
        fits=[
            OptimizeResult(
                x=np.array([30.0]),
                cost=1.0,
                status=1,
                success=True,
                duration=0.1,
                x0=None,
            )
        ],
        trajectories=[[1.0]],
        settings=FitSettings(),
    )
    assert result.scales == [ParameterScaleType.LINEAR]
    assert "within 5% of lower bound" in result.report(print_output=False)
