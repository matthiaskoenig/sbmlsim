"""The criteria and the aggregates of the evaluation of a fit (#325, #326)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from sbmlsim.fit.evaluation import (
    DEFAULT_CRITERIA,
    MIN_POINTS_SUBSET_AGREEMENT,
    Fold,
    Grouping,
    Subset,
    Uncertainty,
    afe,
    criteria_metrics,
    gmfe,
    group_metrics,
    guest_limits,
    subset_agreement,
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


def _table() -> pd.DataFrame:
    """Points of three mappings of two studies with metadata."""
    return pd.DataFrame(
        {
            "experiment": ["A", "A", "A", "B", "B"],
            "mapping": ["fm_p", "fm_p", "fm_u", "fm_p", "fm_p"],
            "kind": ["training", "training", "training", "validation", "training"],
            "DV": [1.0, 2.0, 1.0, 1.0, 4.0],
            "IPRED": [1.0, 1.0, 3.0, 1.5, 4.0],
            "IRES": [0.0, 1.0, -2.0, -0.5, 0.0],
            "NRES": [0.0, 0.5, -2.0, -0.5, 0.0],
            "DV_SD": [0.1, np.nan, 1.0, 1.0, 1.0],
            "tissue": ["PLASMA", "PLASMA", "URINE", "PLASMA", "PLASMA"],
            "health": ["HEALTHY", "HEALTHY", "HEALTHY", "", "IMPAIRED"],
        }
    )


def _no_mapping(experiment: str, mapping: str) -> Any:
    return mapping


def test_a_grouping_by_a_column() -> None:
    groups = Grouping.by("tissue").groups(_table(), _no_mapping)
    assert [label for label, _points in groups] == ["PLASMA", "URINE"]
    assert [len(points) for _label, points in groups] == [4, 1]
    assert Grouping.by("tissue").name == "tissue"


def test_a_grouping_by_two_columns() -> None:
    grouping = Grouping.by("tissue", "health")
    assert grouping.name == "tissue_health"
    labels = [label for label, _points in grouping.groups(_table(), _no_mapping)]
    assert labels == [
        "PLASMA, HEALTHY",
        "URINE, HEALTHY",
        "PLASMA, not reported",
        "PLASMA, IMPAIRED",
    ]


def test_a_grouping_by_subsets_may_overlap() -> None:
    grouping = Grouping.of_subsets(
        "studies",
        Subset("A", lambda key, mapping: key in ("fm_p", "fm_u")),
        Subset("plasma", lambda key, mapping: key == "fm_p"),
    )
    groups = dict(grouping.groups(_table(), _no_mapping))
    assert len(groups["A"]) == 5
    assert len(groups["plasma"]) == 4


def test_an_unknown_column_names_the_columns() -> None:
    with pytest.raises(ValueError, match=r"'route'.*'tissue'"):
        Grouping.by("route").groups(_table(), _no_mapping)


def test_a_grouping_has_columns_or_subsets() -> None:
    with pytest.raises(ValueError, match="columns or subsets"):
        Grouping(name="none")
    with pytest.raises(ValueError, match="columns or subsets"):
        Grouping(
            name="both",
            columns=("tissue",),
            subsets=(Subset("A", lambda key, mapping: True),),
        )


def test_the_metrics_of_a_grouping() -> None:
    metrics = group_metrics(_table(), Grouping.by("tissue"), _no_mapping)
    assert list(metrics.columns[:7]) == [
        "grouping",
        "group",
        "kind",
        "n",
        "n_ratio",
        "n_sd",
        "NRMSE",
    ]
    assert list(metrics.columns[7:]) == list(criteria_metrics(_table()))
    rows = list(zip(metrics.group, metrics.kind, metrics.n, strict=True))
    assert rows == [
        ("PLASMA", "training", 3),
        ("PLASMA", "validation", 1),
        ("URINE", "training", 1),
    ]


def test_a_group_of_one_point_has_metrics() -> None:
    table = _table().iloc[[2]]
    metrics = group_metrics(table, Grouping.by("tissue"), _no_mapping)
    assert metrics.n.tolist() == [1]
    assert np.isfinite(metrics.NRMSE.iloc[0])


def test_subset_agreement() -> None:
    ratios = np.linspace(-0.3, 0.4, 12)
    points = _points([1.0] * 12, list(10**ratios))
    agreement = subset_agreement(points)
    assert agreement is not None
    bias, half, bias_ci, limit_ci = agreement
    sd = np.std(ratios, ddof=1)
    t = scipy.stats.t.ppf(0.975, 11)
    assert bias == pytest.approx(np.mean(ratios))
    assert half == pytest.approx(1.96 * sd)
    assert bias_ci == pytest.approx(t * sd / np.sqrt(12))
    assert limit_ci == pytest.approx(t * np.sqrt(3 * sd**2 / 12))
    # the points without a ratio do not count
    assert subset_agreement(_points([1.0] * 9 + [0.0] * 3, [2.0] * 12)) is None
    assert MIN_POINTS_SUBSET_AGREEMENT == 10


def test_group_metrics_has_its_denominators() -> None:
    metrics = group_metrics(_table(), Grouping.by("tissue"), _no_mapping)
    assert list(metrics.columns[:7]) == [
        "grouping",
        "group",
        "kind",
        "n",
        "n_ratio",
        "n_sd",
        "NRMSE",
    ]
    plasma = metrics[(metrics.group == "PLASMA") & (metrics.kind == "training")]
    assert plasma[["n", "n_ratio", "n_sd"]].to_numpy().tolist() == [[3, 3, 2]]


def test_group_metrics_has_no_rmse_over_units() -> None:
    """The RMSE of a group mixes the units of its observables, NRMSE is scale free."""
    metrics = group_metrics(_table(), Grouping.by("tissue"), _no_mapping)
    assert "RMSE" not in metrics.columns
    table = _table()
    table.loc[0, "NRES"] = np.nan
    metrics = group_metrics(table, Grouping.by("tissue"), _no_mapping)
    first = metrics[(metrics.group == "PLASMA") & (metrics.kind == "training")]
    assert np.isnan(first.NRMSE.iloc[0])


def test_not_reported_values_are_one_group() -> None:
    table = _table()
    table.loc[4, "health"] = "NR"
    labels = [label for label, _ in Grouping.by("health").groups(table, _no_mapping)]
    assert labels == ["HEALTHY", "not reported"]


def test_within_guest_checks_its_input() -> None:
    assert within_guest([2.0, 2.0], [2.5, np.nan]) == pytest.approx(100.0)
    assert np.isnan(within_guest([2.0], [np.nan]))
    with pytest.raises(ValueError, match="positive"):
        within_guest([0.0], [1.0])
    with pytest.raises(ValueError, match="lengths"):
        within_guest([2.0, 2.0], [2.0])
