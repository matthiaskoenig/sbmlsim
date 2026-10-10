"""The report of a fit with values per simulation and values over a dimension."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from sbmlsim.fit import FitMappingCollection, FitParameter, FitSettings, ParameterSet
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType, ResidualType
from sbmlsim.fit.report import FitReport
from tests.fit.scalar_experiment import (
    CMAX,
    TRUE_V,
    DoseStudy,
    IndividualsStudy,
    MixedStudy,
    OneDoseStudy,
    RowsStudy,
    ScalarStudy,
)


def _problem(
    collections: list[FitMappingCollection], tmp_path: Path
) -> OptimizationProblem:
    return OptimizationProblem(
        "scalar_report",
        collections,
        [
            FitParameter(
                pid="V", lower_bound=1.0, upper_bound=100.0, start_value=30.0, unit="l"
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
    )


@pytest.fixture
def report(tmp_path: Path) -> FitReport:
    """The report of the true parameters, a perfect fit, and of parameters near them."""
    problem = _problem(
        [
            FitMappingCollection(experiment=ScalarStudy, mappings=["fm_cmax"]),
            FitMappingCollection(experiment=DoseStudy, mappings=["fm_dose"]),
        ],
        tmp_path,
    )
    settings = FitSettings(parameter_scale=ParameterScaleType.LINEAR)
    return FitReport(
        problem,
        settings,
        [
            ParameterSet(sid="true", values={"V": TRUE_V}),
            ParameterSet(sid="near", values={"V": 1.05 * TRUE_V}),
        ],
        mapping_figures=True,
    )


def test_the_report_files_of_both_kinds(report: FitReport, tmp_path: Path) -> None:
    """The html and the two figures of both mappings are written."""
    results_dir = report.create(output_dir=tmp_path / "out", name="scalar")
    assert (results_dir / "index.html").exists()
    plots = results_dir / "plots"
    for mapping in report.problem.mapping_keys:
        sid = report.problem.experiment_keys[report.problem.mapping_keys.index(mapping)]
        assert (plots / f"{sid}_{mapping}.svg").exists()
        assert (plots / f"fit_{sid}_{mapping}.svg").exists()
    assert (plots / "goodness_of_fit.svg").exists()
    assert (plots / "bland_altman.svg").exists()
    html = (results_dir / "index.html").read_text()
    assert "fm_cmax" in html
    assert "fm_dose" in html


def test_the_points_of_scalar_mappings(report: FitReport) -> None:
    """A scalar point has no x, a point over a dimension has the dose."""
    pset = report.reference_set
    points = report.points(pset)
    assert len(points) == 1 + 3
    assert np.isfinite(points["DV"]).all()
    assert np.isfinite(points["IPRED"]).all()
    scalar = points[points["mapping"] == "fm_cmax"]
    assert scalar["x"].isna().all()
    dose = points[points["mapping"] == "fm_dose"]
    assert np.isfinite(dose["x"]).all()

    # the data is the simulation of the true parameters
    np.testing.assert_allclose(points["DV"], points["IPRED"])

    frame = report._datapoints_df(pset)
    assert len(frame) == 4
    assert frame[frame["mapping"] == "fm_cmax"]["x_ref"].isna().all()


def test_a_perfect_fit_has_the_best_information_criteria(report: FitReport) -> None:
    """The AIC and BIC of an MSE of 0 are minus infinity, the R² of one row NaN."""
    metrics = report.metrics_df().set_index("parameter_set")
    assert metrics.loc["true", "MSE"] == 0.0
    assert metrics.loc["true", "AIC"] == float("-inf")
    assert metrics.loc["true", "BIC"] == float("-inf")
    assert np.isfinite(metrics.loc["near", ["AIC", "BIC"]].astype(float)).all()

    mappings = report.metrics_mappings_df()
    true = mappings[mappings["parameter_set"] == "true"].set_index("mapping")
    assert np.isnan(true.loc["fm_cmax", "R2"])
    assert true.loc["fm_dose", "R2"] == 1.0


def test_metrics_of_mappings_with_one_key_in_two_experiments(tmp_path: Path) -> None:
    """The mappings `fm_cmax` of two studies have their own n, MSE and R²."""
    problem = _problem(
        [
            FitMappingCollection(experiment=ScalarStudy, mappings=["fm_cmax"]),
            FitMappingCollection(experiment=IndividualsStudy, mappings=["fm_cmax"]),
        ],
        tmp_path,
    )
    report = FitReport(
        problem,
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
        ParameterSet(sid="true", values={"V": TRUE_V}),
        mapping_figures=False,
    )
    mappings = report.metrics_mappings_df().set_index("experiment")
    assert list(mappings["mapping"]) == ["fm_cmax", "fm_cmax"]

    assert mappings.loc["ScalarStudy", "n"] == 1
    assert mappings.loc["ScalarStudy", "MSE"] == 0.0
    assert np.isnan(mappings.loc["ScalarStudy", "R2"])

    # the individuals scatter around the one simulated cmax
    individuals = CMAX[1] * np.array([0.9, 1.0, 1.2])
    residuals = individuals - CMAX[1]
    assert mappings.loc["IndividualsStudy", "n"] == 3
    assert mappings.loc["IndividualsStudy", "MSE"] == pytest.approx(
        np.mean(residuals**2)
    )
    sst = np.sum((individuals - individuals.mean()) ** 2)
    assert mappings.loc["IndividualsStudy", "R2"] == pytest.approx(
        1 - np.sum(residuals**2) / sst
    )

    # the table and the card of every mapping
    results_dir = report.create(output_dir=tmp_path / "out", name="shared")
    table = pd.read_csv(results_dir / "metrics_mappings.tsv", sep="\t")
    assert sorted(table["n"]) == [1, 3]
    html = (results_dir / "index.html").read_text()
    assert html.count("fm_cmax") >= 2


@pytest.fixture
def figures(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Figure]]:
    """The figures a report saves, by the name of their file."""
    saved: dict[str, Figure] = {}

    def save(self: FitReport, fig: Figure, path: Path) -> None:
        saved[path.stem] = fig

    monkeypatch.setattr(FitReport, "_save_mpl_figure", save)
    yield saved
    for fig in saved.values():
        plt.close(fig)


def test_the_figures_of_rows_and_of_a_dimension_of_one_value(
    tmp_path: Path, figures: dict[str, Figure]
) -> None:
    """Every marker is inside, the simulation is next to the rows, one dose has room."""
    problem = _problem(
        [
            FitMappingCollection(experiment=RowsStudy, mappings=["fm_rows"]),
            FitMappingCollection(experiment=OneDoseStudy, mappings=["fm_one"]),
        ],
        tmp_path,
    )
    report = FitReport(
        problem,
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
        ParameterSet(sid="near", values={"V": 1.2 * TRUE_V}),
    )
    report.plot_fit(tmp_path)
    report.plot_fit_residual(tmp_path)

    rows = CMAX[1] * np.array([0.9, 1.0, 1.2])
    for ax in figures["RowsStudy_fm_rows"].axes:
        bottom, top = ax.get_ylim()
        assert bottom < rows.min() and top > rows.max()
        if ax.get_yscale() == "log":
            # the top marker has room above it, also when the rows span less
            # than a decade
            assert top > 1.2 * rows.max()
        assert ax.get_ylabel() == "pk.cmax [mg/l]"
        [simulation] = [line for line in ax.get_lines() if line.get_label() == "near"]
        # next to every row, not on top of it
        x_sim = np.asarray(simulation.get_xdata(), dtype=float)
        assert np.all((x_sim > np.arange(3)) & (x_sim < np.arange(3) + 0.5))
    for ax in figures["fit_RowsStudy_fm_rows"].axes[:2]:
        [prediction] = [line for line in ax.get_lines() if line.get_label() == "near"]
        x_pred = np.asarray(prediction.get_xdata(), dtype=float)
        assert np.all((x_pred > np.arange(3)) & (x_pred < np.arange(3) + 0.5))

    for name in ("OneDoseStudy_fm_one", "fit_OneDoseStudy_fm_one"):
        for ax in figures[name].axes:
            left, right = ax.get_xlim()
            assert left < 100.0 < right
    assert figures["OneDoseStudy_fm_one"].axes[0].get_xlabel() == "PODOSE [mg]"


def test_a_mapping_without_positive_values_has_a_linear_log_panel(
    tmp_path: Path, figures: dict[str, Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log panel of the data figure stays linear, without a warning."""
    problem = _problem(
        [FitMappingCollection(experiment=RowsStudy, mappings=["fm_rows"])], tmp_path
    )
    report = FitReport(
        problem,
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
        ParameterSet(sid="near", values={"V": TRUE_V}),
    )
    original = FitReport.residual_data

    def negative(self: FitReport, pset: ParameterSet) -> dict[str, Any]:
        data = original(self, pset)
        data["y_obs"] = [-np.abs(y) for y in data["y_obs"]]
        return data

    monkeypatch.setattr(FitReport, "residual_data", negative)
    monkeypatch.setattr(
        problem, "y_references", [-np.abs(y) for y in problem.y_references]
    )
    report.plot_fit(tmp_path)
    assert [ax.get_yscale() for ax in figures["RowsStudy_fm_rows"].axes] == [
        "linear",
        "linear",
    ]


def test_the_cost_axis_of_a_perfect_fit_starts_at_zero(
    tmp_path: Path, figures: dict[str, Figure]
) -> None:
    """A cost of zero has no logarithm, its linear axis has no negative costs."""
    problem = _problem(
        [FitMappingCollection(experiment=ScalarStudy, mappings=["fm_cmax"])], tmp_path
    )
    report = FitReport(
        problem,
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
        ParameterSet(sid="true", values={"V": TRUE_V}),
    )
    report.plot_cost_bar(tmp_path / "cost.png")
    ax = figures["cost"].axes[0]
    assert ax.get_xscale() == "linear"
    left, right = ax.get_xlim()
    assert left == 0.0 and right > 0.0


def test_the_residual_figure_of_a_dimension_draws_what_the_legend_lists(
    tmp_path: Path, figures: dict[str, Figure]
) -> None:
    """The line of a dimension of one value has the marker which shows it."""
    problem = _problem(
        [FitMappingCollection(experiment=OneDoseStudy, mappings=["fm_one"])], tmp_path
    )
    report = FitReport(
        problem,
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
        ParameterSet(sid="near", values={"V": 1.2 * TRUE_V}),
    )
    report.plot_fit_residual(tmp_path)
    for ax in figures["fit_OneDoseStudy_fm_one"].axes[:2]:
        [line] = [line for line in ax.get_lines() if line.get_label() == "near"][:1]
        assert line.get_marker() not in (None, "None", "")


def test_normalized_residuals_are_marked_in_the_residual_figure(
    tmp_path: Path, figures: dict[str, Figure]
) -> None:
    """Dimensionless residuals do not read as a quantity of the data unit."""
    problem = _problem(
        [FitMappingCollection(experiment=MixedStudy, mappings=["fm_tc"])], tmp_path
    )
    for residual, normalized in (
        (ResidualType.ABSOLUTE, False),
        (ResidualType.NORMALIZED, True),
    ):
        report = FitReport(
            problem,
            FitSettings(parameter_scale=ParameterScaleType.LINEAR, residual=residual),
            ParameterSet(sid="near", values={"V": 1.2 * TRUE_V}),
        )
        figures.clear()
        report.plot_fit_residual(tmp_path)
        ax = figures["fit_MixedStudy_fm_tc"].axes[0]
        _, labels = ax.get_legend_handles_labels()
        assert ("near normalized residuals" in labels) is normalized
        assert ("near residuals" in labels) is not normalized
        assert ("residuals [-]" in ax.get_ylabel()) is normalized


def test_the_x_limits_ignore_a_measurement_at_infinity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A steady state after the end is not a position on the axis."""
    problem = _problem(
        [FitMappingCollection(experiment=MixedStudy, mappings=["fm_tc"])], tmp_path
    )
    report = FitReport(
        problem,
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
        ParameterSet(sid="near", values={"V": TRUE_V}),
    )
    monkeypatch.setattr(
        FitReport, "_x_positions", lambda self, k: np.array([0.0, 2.0, np.inf])
    )
    fig, ax = plt.subplots()
    report._set_x_limits(ax, 0, [np.array([0.0, 2.0])])
    left, right = ax.get_xlim()
    plt.close(fig)
    assert np.isfinite(left) and np.isfinite(right) and left < 0.0 and right > 2.0
