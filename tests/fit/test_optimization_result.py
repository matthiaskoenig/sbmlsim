"""Test optimization results."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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


def test_a_result_without_provenance_reads_as_none(
    tmp_path: Path,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
) -> None:
    """A fit which records no provenance round trips with `None`."""
    opt_res = run_optimization(
        problem=op_hctz_pk,
        settings=fit_settings,
        size=1,
        n_cores=1,
        serial=True,
    )
    assert opt_res.provenance is None
    assert opt_res.to_dict()["provenance"] is None
    path = tmp_path / "opt_res.json"
    opt_res.to_json(path=path)
    assert OptimizationResult.from_json(path).provenance is None


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


def _single_result(
    parameters: list[FitParameter], x: list[float]
) -> OptimizationResult:
    return OptimizationResult(
        parameters=parameters,
        fits=[OptimizeResult(x=x, x0=x, cost=1.0, success=True, duration=0.1)],
        trajectories=[[1.0]],
    )


def test_combining_reordered_parameters_preserves_values(tmp_path: Path) -> None:
    """Recovered runs preserve parameter values even if their definitions reorder."""
    a = FitParameter("a", 1.0, 0.1, 100.0, unit="dimensionless")
    b = FitParameter("b", 1.0, 0.1, 100.0, unit="dimensionless")
    first = _single_result([a, b], [1.0, 2.0])
    second = _single_result([b, a], [20.0, 10.0])
    second.fits[0].cost = 0.5
    first.to_json(tmp_path / "first.json")
    second.to_json(tmp_path / "second.json")

    combined = OptimizationResult.from_directory(tmp_path)
    assert combined.parameter_set().values == {"a": 10.0, "b": 20.0}
    np.testing.assert_array_equal(combined.fits[1].x0, [10.0, 20.0])
    np.testing.assert_array_equal(second.fits[0].x, [20.0, 10.0])


@pytest.mark.parametrize(
    "change", ["id", "target", "bounds", "scale", "units", "settings"]
)
def test_combining_incompatible_results_raises(change: str) -> None:
    """A shared parameter label does not make incompatible fits comparable."""
    first = _single_result(
        [FitParameter("a", 1.0, 0.1, 100.0, unit="dimensionless")], [1.0]
    )
    second = _single_result(
        [FitParameter("a", 1.0, 0.1, 100.0, unit="dimensionless")], [2.0]
    )
    if change == "id":
        second.parameters[0].pid = "b"
    elif change == "target":
        second.parameters[0].target = "b"
    elif change == "bounds":
        second.parameters[0].upper_bound = 10.0
    elif change == "scale":
        second.parameters[0].scale = ParameterScaleType.LINEAR
    elif change == "units":
        second.units["a"] = "mM"
    else:
        second.settings = FitSettings()
    with pytest.raises(ValueError, match="incompatible"):
        OptimizationResult.combine([first, second])


def test_combining_different_labels_and_starts_preserves_results() -> None:
    """The run label and start value do not change the fitted objective."""
    first = _single_result(
        [FitParameter("a", 1.0, 0.1, 100.0, unit="dimensionless")], [1.0]
    )
    second = _single_result(
        [FitParameter("a", 2.0, 0.1, 100.0, unit="dimensionless")], [2.0]
    )
    first.opid, second.opid = "first", "second"
    assert OptimizationResult.combine([first, second]).size == 2


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


def test_the_tsv_of_a_result_has_one_line_per_run(tmp_path: Path) -> None:
    """Every run is one line of the TSV, the start values are columns too."""
    n = 12
    parameters = [
        FitParameter(f"p{k}", 1.0, 0.1, 10.0, unit="dimensionless") for k in range(n)
    ]
    fits = [
        OptimizeResult(
            x=np.linspace(1.0, 2.0, n) + run,
            x0=np.linspace(0.5, 1.5, n) + run,
            cost=1.0 + run,
            status=1,
            success=True,
            duration=0.1,
            message="`ftol` termination condition\nis satisfied.",
        )
        for run in range(2)
    ]
    fits.append(
        OptimizeResult(
            x=None, x0=None, cost=np.inf, success=False, duration=0.0, message="error"
        )
    )
    result = OptimizationResult(
        parameters=parameters, fits=fits, trajectories=[[1.0], [2.0], []]
    )

    path = tmp_path / "optimization_result.tsv"
    result.to_tsv(path)

    assert len(path.read_text().splitlines()) == 1 + 3
    df = pd.read_csv(path, sep="\t")
    assert len(df) == 3
    assert "x" not in df.columns and "x0" not in df.columns
    pids = [p.pid for p in parameters]
    assert np.allclose(df.loc[0, pids].to_numpy(dtype=float), fits[0].x)
    assert np.allclose(
        df.loc[1, [f"x0.{pid}" for pid in pids]].to_numpy(dtype=float), fits[1].x0
    )
    assert df.loc[2, [f"x0.{pid}" for pid in pids]].isna().all()
    assert df.loc[0, "message"] == "`ftol` termination condition is satisfied."
