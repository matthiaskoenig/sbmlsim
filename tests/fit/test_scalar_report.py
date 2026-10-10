"""The report of a fit with values per simulation and values over a dimension."""

from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from sbmlsim.fit import FitMappingCollection, FitParameter, FitSettings, ParameterSet
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.report import FitReport
from tests.fit.scalar_experiment import (
    CMAX,
    TRUE_V,
    DoseStudy,
    IndividualsStudy,
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
