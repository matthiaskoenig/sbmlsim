"""Test the metrics of a fit."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from sbmlsim.fit import FitMetrics, FitSettings, MappingKind, ParameterSet
from sbmlsim.fit.metrics import (
    aic,
    aic_from_mse,
    mse,
    r_squared,
    rmse,
    rmse_from_mse,
    sse,
    within_fold,
)
from sbmlsim.fit.optimization import OptimizationProblem


def test_metrics_report_simulates_once_and_tracks_changed_parameters(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Report tables share predictions without a stale persistent cache."""
    op_hctz_pk.initialize(fit_settings)
    metrics = FitMetrics(op_hctz_pk, op_hctz_pk.parameter_set_model())
    with patch.object(
        op_hctz_pk, "_simulate_groups", wraps=op_hctz_pk._simulate_groups
    ) as simulations:
        original = metrics.report()
        assert simulations.call_count == 1
        metrics.parameter_set.values[op_hctz_pk.pids[0]] *= 1.5
        changed = metrics.report()
        assert simulations.call_count == 2
        assert original != changed


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
        ({"mse": -1.0, "n": 3, "k": 1}, "non-negative MSE"),
        ({"mse": 1.0, "n": 0, "k": 1}, "positive number of points"),
    ],
)
def test_aic_invalid(kwargs: dict, match: str) -> None:
    """The AIC needs a non-negative MSE and data points."""
    with pytest.raises(ValueError, match=match):
        aic_from_mse(**kwargs)


def test_aic_of_a_perfect_fit() -> None:
    """A perfect fit has the best possible AIC, minus infinity."""
    assert aic_from_mse(mse=0.0, n=3, k=1) == float("-inf")
    assert aic([0.0, 0.0, 0.0], k=1) == float("-inf")


def test_r_squared() -> None:
    """The coefficient of determination of a prediction."""
    assert r_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    # the mean of the data explains nothing
    assert r_squared([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == pytest.approx(0.0)
    # a prediction worse than the mean is negative
    assert r_squared([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) < 0.0


def test_r_squared_without_variance(caplog: pytest.LogCaptureFixture) -> None:
    """R² is undefined for data without variance, which is worth a warning."""
    assert np.isnan(r_squared([2.0, 2.0], [1.0, 3.0]))
    assert "without variance" in caplog.text


def test_r_squared_of_one_point(caplog: pytest.LogCaptureFixture) -> None:
    """R² of one point, e.g. a value per simulation of one group, is NaN silently."""
    assert np.isnan(r_squared([2.0], [1.0]))
    assert caplog.records == []


def test_r_squared_length_mismatch() -> None:
    """As many predictions as data points are required."""
    with pytest.raises(ValueError, match="as many predictions"):
        r_squared([1.0, 2.0], [1.0])


def test_within_twofold() -> None:
    """The percentage of the positive points whose prediction is within twofold.

    The ratios 0.4 and 2.1 are outside, 0.5, 1 and 2 inside, the bounds count.
    A point with a non-positive data value or prediction has no ratio and is
    left out, so 3 of the 5 points with a ratio are inside.
    """
    data = [1.0, 2.0, 4.0, 0.5, 10.0, 3.0, 0.0, -1.0]
    predicted = [0.4, 1.0, 4.0, 1.0, 21.0, -3.0, 1.0, -2.0]
    assert within_fold(data, predicted) == pytest.approx(60.0)
    # the default is twofold, a threefold criterion takes in 0.4 and 2.1
    assert within_fold(data, predicted, fold=2.0) == pytest.approx(60.0)
    assert within_fold(data, predicted, fold=3.0) == pytest.approx(100.0)
    # every point inside, and none
    assert within_fold([1.0, 2.0], [2.0, 1.0]) == pytest.approx(100.0)
    assert within_fold([1.0, 2.0], [3.0, 0.5]) == pytest.approx(0.0)


def test_within_twofold_without_a_ratio() -> None:
    """Without a point with positive data and prediction the criterion is NaN."""
    assert np.isnan(within_fold([0.0, -1.0, 2.0], [1.0, 1.0, 0.0]))
    assert np.isnan(within_fold([], []))
    with pytest.raises(ValueError, match="as many predictions"):
        within_fold([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="fold factor"):
        within_fold([1.0], [1.0], fold=1.0)


def test_metrics_require_initialized_problem(
    op_hctz_pk: OptimizationProblem,
) -> None:
    """The metrics need the resolved data of an initialized problem."""
    pset = ParameterSet(sid="s", values=dict.fromkeys(op_hctz_pk.pids, 1.0))
    with pytest.raises(ValueError, match="not initialized"):
        FitMetrics(problem=op_hctz_pk, parameter_set=pset)


def test_metrics_datapoints(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The data points carry the data and the predictions."""
    op_hctz_pk.initialize(fit_settings)
    metrics = FitMetrics(
        problem=op_hctz_pk, parameter_set=op_hctz_pk.parameter_set_model()
    )
    dp = metrics.datapoints_df()

    assert list(dp.columns) == [
        "experiment",
        "mapping",
        "kind",
        "x",
        "DV",
        "PRED",
        "IPRED",
        "RES",
        "IRES",
        "NRES",
        "IWRES",
    ]
    assert len(dp) == sum(len(y) for y in op_hctz_pk.y_references)

    # without individual parameters PRED is IPRED
    assert np.allclose(dp.PRED, dp.IPRED)
    assert np.allclose(dp.RES, dp.IRES)
    # the residuals follow the convention data - prediction
    assert np.allclose(dp.IRES, dp.DV - dp.IPRED)


def test_metrics_population_and_individual(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """PRED and IPRED differ when a population parameter set is given."""
    op_hctz_pk.initialize(fit_settings)
    model_set = op_hctz_pk.parameter_set_model()
    other_set = ParameterSet(
        sid="individual",
        values={pid: value * 1.5 for pid, value in model_set.values.items()},
    )

    metrics = FitMetrics(
        problem=op_hctz_pk,
        parameter_set=other_set,
        population_parameter_set=model_set,
    )
    dp = metrics.datapoints_df()
    assert not np.allclose(dp.PRED, dp.IPRED)
    assert np.allclose(dp.RES, dp.DV - dp.PRED)
    assert np.allclose(dp.IRES, dp.DV - dp.IPRED)


def test_metrics_mappings_and_summary(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The metrics are reported per mapping and over all data points."""
    op_hctz_iv.initialize(fit_settings)
    metrics = FitMetrics(
        problem=op_hctz_iv, parameter_set=op_hctz_iv.parameter_set_model()
    )

    mappings = metrics.mappings_df()
    assert list(mappings.columns) == [
        "experiment",
        "mapping",
        "kind",
        "n",
        "MSE",
        "RMSE",
        "NRMSE",
        "RMSE_w",
        "R2",
        "within_2fold",
    ]
    assert len(mappings) == len(op_hctz_iv.mapping_keys)
    assert (mappings.n > 0).all()
    assert np.allclose(mappings.RMSE, np.sqrt(mappings.MSE))

    summary = metrics.summary()
    assert set(summary) == {
        "parameter_set",
        "kind",
        "n",
        "k",
        "cost",
        "MSE",
        "RMSE",
        "NRMSE",
        "RMSE_w",
        "R2",
        "within_2fold",
        "AIC",
        "BIC",
    }
    # the information criteria are over all data points and not per mapping,
    # a criterion of a subset with the parameters of the whole fit says nothing
    assert "AIC" not in mappings.columns
    assert "BIC" not in mappings.columns
    # the BIC charges `ln(n)` per parameter, the AIC `2`
    assert summary["BIC"] > summary["AIC"]
    assert summary["kind"] == "all"
    assert summary["n"] == int(mappings.n.sum())
    assert summary["k"] == len(op_hctz_iv.parameters)
    assert summary["RMSE"] == pytest.approx(np.sqrt(summary["MSE"]))
    assert summary["AIC"] == pytest.approx(
        aic_from_mse(mse=summary["MSE"], n=summary["n"], k=summary["k"])
    )

    # the twofold criterion of the same data points and predictions
    datapoints = metrics.datapoints_df()
    assert summary["within_2fold"] == pytest.approx(
        within_fold(datapoints.DV, datapoints.IPRED)
    )
    assert 0.0 <= summary["within_2fold"] <= 100.0
    for _, row in mappings.iterrows():
        of_mapping = datapoints[
            (datapoints.experiment == row["experiment"])
            & (datapoints.mapping == row["mapping"])
        ]
        expected = within_fold(of_mapping.DV, of_mapping.IPRED)
        assert row["within_2fold"] == pytest.approx(expected, nan_ok=True)

    # the cost is the objective of the optimization
    assert summary["cost"] == pytest.approx(
        op_hctz_iv.cost_least_square(np.log10(op_hctz_iv.xmodel))
    )
    assert metrics.report()


def test_normalized_and_weighted_residuals(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """NRES is scale free and IWRES is the residual of the cost."""
    op_hctz_pk.initialize(fit_settings)
    metrics = FitMetrics(
        problem=op_hctz_pk, parameter_set=op_hctz_pk.parameter_set_model()
    )
    dp = metrics.datapoints_df()

    # the normalized residual is the residual over the mean of its curve
    for _mapping, of_mapping in dp.groupby(["experiment", "mapping"]):
        assert np.allclose(of_mapping.NRES, of_mapping.IRES / of_mapping.DV.mean())

    # the weighted residuals are what the cost sums, on the training data
    training = dp[dp.kind == MappingKind.TRAINING.value]
    cost = 0.5 * float(np.sum(np.square(training.IWRES)))
    assert cost == pytest.approx(metrics.cost())
    summary = metrics.summary(kind=MappingKind.TRAINING)
    assert summary["cost"] == pytest.approx(0.5 * summary["n"] * summary["RMSE_w"] ** 2)
    assert summary["NRMSE"] == pytest.approx(rmse(training.NRES))

    # the outliers are curves of small values missed by a factor: their absolute
    # RMSE is below the one of the training data, their NRMSE is far above
    outliers = metrics.summary(kind=MappingKind.OUTLIER)
    assert outliers["RMSE"] < summary["RMSE"]
    assert outliers["NRMSE"] > 5 * summary["NRMSE"]


def test_the_documentation_describes_the_twofold_criterion() -> None:
    """The metrics part of the documentation of fitting names the criterion."""
    page = (Path(__file__).parents[2] / "docs" / "fitting.md").read_text(
        encoding="utf-8"
    )
    metrics = page.split("\n## Metrics\n", 1)[1].split("\n## ", 1)[0]
    assert "within_2fold" in metrics
    assert "within twofold" in metrics
    assert "within_fold" in metrics
