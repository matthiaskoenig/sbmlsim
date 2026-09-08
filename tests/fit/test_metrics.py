"""Test the metrics of a fit."""

import numpy as np
import pytest

from sbmlsim.fit import FitMetrics, FitSettings, ParameterSet
from sbmlsim.fit.metrics import (
    aic,
    aic_from_mse,
    mse,
    r_squared,
    rmse,
    rmse_from_mse,
    sse,
)
from sbmlsim.fit.optimization import OptimizationProblem


def test_sse_mse_rmse() -> None:
    """The errors of the residuals."""
    residuals = [1.0, -2.0, 3.0]
    assert sse(residuals) == pytest.approx(14.0)
    assert mse(residuals) == pytest.approx(14.0 / 3.0)
    assert rmse(residuals) == pytest.approx(np.sqrt(14.0 / 3.0))
    assert rmse_from_mse(mse(residuals)) == pytest.approx(rmse(residuals))


def test_mse_requires_residuals() -> None:
    """The MSE of no residuals is an error."""
    with pytest.raises(ValueError, match="at least one residual"):
        mse([])


def test_rmse_negative_mse() -> None:
    """A negative mean squared error is an error."""
    with pytest.raises(ValueError, match="non-negative"):
        rmse_from_mse(-1.0)


def test_aic() -> None:
    """The AIC of the residuals and of the mean squared error agree."""
    residuals = [1.0, -2.0, 3.0]
    assert aic(residuals, k=2) == pytest.approx(
        aic_from_mse(mse=mse(residuals), n=3, k=2)
    )
    # more parameters are penalized
    assert aic(residuals, k=3) > aic(residuals, k=2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mse": 0.0, "n": 3, "k": 1}, "positive MSE"),
        ({"mse": 1.0, "n": 0, "k": 1}, "positive number of points"),
    ],
)
def test_aic_invalid(kwargs: dict, match: str) -> None:
    """The AIC needs a positive MSE and data points."""
    with pytest.raises(ValueError, match=match):
        aic_from_mse(**kwargs)


def test_r_squared() -> None:
    """The coefficient of determination of a prediction."""
    assert r_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    # the mean of the data explains nothing
    assert r_squared([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == pytest.approx(0.0)
    # a prediction worse than the mean is negative
    assert r_squared([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) < 0.0


def test_r_squared_without_variance() -> None:
    """R² is undefined for data without variance."""
    assert np.isnan(r_squared([2.0, 2.0], [1.0, 3.0]))


def test_r_squared_length_mismatch() -> None:
    """As many predictions as data points are required."""
    with pytest.raises(ValueError, match="as many predictions"):
        r_squared([1.0, 2.0], [1.0])


def test_metrics_require_initialized_problem(
    op_hctz_pkiv: OptimizationProblem,
) -> None:
    """The metrics need the resolved data of an initialized problem."""
    pset = ParameterSet(sid="s", values=dict.fromkeys(op_hctz_pkiv.pids, 1.0))
    with pytest.raises(ValueError, match="not initialized"):
        FitMetrics(problem=op_hctz_pkiv, parameter_set=pset)


def test_metrics_datapoints(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The data points carry the data and the predictions."""
    op_hctz_pkiv.initialize(fit_settings)
    metrics = FitMetrics(
        problem=op_hctz_pkiv, parameter_set=op_hctz_pkiv.parameter_set_model()
    )
    dp = metrics.datapoints_df()

    assert list(dp.columns) == [
        "experiment",
        "mapping",
        "x",
        "DV",
        "PRED",
        "IPRED",
        "RES",
        "IRES",
        "IWRES",
    ]
    assert len(dp) == sum(len(y) for y in op_hctz_pkiv.y_references)

    # without individual parameters PRED is IPRED
    assert np.allclose(dp.PRED, dp.IPRED)
    assert np.allclose(dp.RES, dp.IRES)
    # the residuals follow the convention data - prediction
    assert np.allclose(dp.IRES, dp.DV - dp.IPRED)


def test_metrics_population_and_individual(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """PRED and IPRED differ when a population parameter set is given."""
    op_hctz_pkiv.initialize(fit_settings)
    model_set = op_hctz_pkiv.parameter_set_model()
    other_set = ParameterSet(
        sid="individual",
        values={pid: value * 1.5 for pid, value in model_set.values.items()},
    )

    metrics = FitMetrics(
        problem=op_hctz_pkiv,
        parameter_set=other_set,
        population_parameter_set=model_set,
    )
    dp = metrics.datapoints_df()
    assert not np.allclose(dp.PRED, dp.IPRED)
    assert np.allclose(dp.RES, dp.DV - dp.PRED)
    assert np.allclose(dp.IRES, dp.DV - dp.IPRED)


def test_metrics_mappings_and_summary(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The metrics are reported per mapping and over all data points."""
    op_hctz_pkiv.initialize(fit_settings)
    metrics = FitMetrics(
        problem=op_hctz_pkiv, parameter_set=op_hctz_pkiv.parameter_set_model()
    )

    mappings = metrics.mappings_df()
    assert list(mappings.columns) == [
        "experiment",
        "mapping",
        "n",
        "MSE",
        "RMSE",
        "R2",
    ]
    assert len(mappings) == len(op_hctz_pkiv.mapping_keys)
    assert (mappings.n > 0).all()
    assert np.allclose(mappings.RMSE, np.sqrt(mappings.MSE))

    summary = metrics.summary()
    assert set(summary) == {
        "parameter_set",
        "n",
        "k",
        "cost",
        "MSE",
        "RMSE",
        "RMSE_w",
        "R2",
        "AIC",
    }
    assert summary["n"] == int(mappings.n.sum())
    assert summary["k"] == len(op_hctz_pkiv.parameters)
    assert summary["RMSE"] == pytest.approx(np.sqrt(summary["MSE"]))
    assert summary["AIC"] == pytest.approx(
        aic_from_mse(mse=summary["MSE"], n=summary["n"], k=summary["k"])
    )

    # the cost is the objective of the optimization
    assert summary["cost"] == pytest.approx(
        op_hctz_pkiv.cost_least_square(np.log10(op_hctz_pkiv.xmodel))
    )
    assert metrics.report()
