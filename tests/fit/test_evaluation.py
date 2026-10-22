"""The criteria and the aggregates of the evaluation of a fit (#325, #326)."""

import numpy as np
import pandas as pd
import pytest

from sbmlsim.fit.evaluation import (
    DEFAULT_CRITERIA,
    Fold,
    Uncertainty,
    afe,
    criteria_metrics,
    gmfe,
    guest_limits,
    within_guest,
)


def _points(
    dv: list[float], ipred: list[float], sd: list[float] | None = None
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DV": dv,
            "IPRED": ipred,
            "DV_SD": sd if sd is not None else [np.nan] * len(dv),
        }
    )


def test_fold_counts_the_points_with_a_ratio() -> None:
    points = _points([1.0, 1.0, 1.0, 0.0, -1.0], [2.0, 0.49, 1.3, 1.0, 1.0])
    metrics = criteria_metrics(points)
    assert metrics["within_2fold"] == pytest.approx(200 / 3)
    assert metrics["within_1.25fold"] == pytest.approx(0.0)


def test_a_point_without_a_ratio_is_not_applicable() -> None:
    accepted = Fold(2.0).accepted(_points([1.0, 0.0, 1.0], [1.0, 1.0, np.inf]))
    assert accepted.iloc[0] == 1.0
    assert np.isnan(accepted.iloc[1])
    assert np.isnan(accepted.iloc[2])


def test_gmfe_and_afe() -> None:
    points = _points([1.0, 1.0], [2.0, 0.5])
    assert gmfe(points) == pytest.approx(2.0)
    assert afe(points) == pytest.approx(1.0)
    assert afe(_points([1.0, 1.0], [2.0, 2.0])) == pytest.approx(2.0)
    assert np.isnan(gmfe(_points([0.0], [1.0])))
    assert np.isnan(afe(_points([0.0], [1.0])))


def test_uncertainty_needs_a_standard_deviation() -> None:
    points = _points([10.0, 10.0, 10.0], [11.0, 13.0, 30.0], [1.0, 1.0, np.nan])
    metrics = criteria_metrics(points)
    assert metrics["within_1sd"] == pytest.approx(50.0)
    assert metrics["within_2sd"] == pytest.approx(50.0)
    assert np.isnan(criteria_metrics(_points([1.0], [1.0]))["within_2sd"])


def test_the_keys_of_the_metrics() -> None:
    assert list(criteria_metrics(_points([1.0], [1.0]))) == [
        "GMFE",
        "AFE",
        "within_1.25fold",
        "within_1.5fold",
        "within_2fold",
        "within_1sd",
        "within_2sd",
    ]
    assert [criterion.name for criterion in DEFAULT_CRITERIA] == [
        "within_1.25fold",
        "within_1.5fold",
        "within_2fold",
        "within_1sd",
        "within_2sd",
    ]
    assert Uncertainty(2.0).name == "within_2sd"


def test_no_points_give_nan() -> None:
    metrics = criteria_metrics(_points([], []))
    assert all(np.isnan(value) for value in metrics.values())


def test_guest_as_published() -> None:
    low, high = guest_limits([1.0, 2.0, 0.5, 10.0])
    np.testing.assert_allclose(high, [1.0, 3.0, 0.75, 19.0])
    np.testing.assert_allclose(low, [1.0, 4 / 3, 1 / 3, 10 / 1.9])
    assert within_guest([2.0, 2.0], [2.5, 3.5]) == pytest.approx(50.0)
    assert within_guest([0.5], [0.6]) == pytest.approx(100.0)
