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
from sbmlsim.fit.options import ParameterScaleType
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
    """The html and the figure of both mappings are written."""
    results_dir = report.create(output_dir=tmp_path / "out", name="scalar")
    assert (results_dir / "index.html").exists()
    plots = results_dir / "plots"
    for mapping in report.problem.mapping_keys:
        sid = report.problem.experiment_keys[report.problem.mapping_keys.index(mapping)]
        assert (plots / f"{sid}_{mapping}.svg").exists()
        assert not (plots / f"fit_{sid}_{mapping}.svg").exists()
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
    report.plot_mapping(tmp_path)

    rows = CMAX[1] * np.array([0.9, 1.0, 1.2])
    # the curve, linear and logarithmic
    for ax in figures["RowsStudy_fm_rows"].axes[:2]:
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
    for ax in figures["OneDoseStudy_fm_one"].axes[:2]:
        left, right = ax.get_xlim()
        assert left < 100.0 < right
    assert figures["OneDoseStudy_fm_one"].axes[0].get_xlabel() == "PODOSE [mg]"


def _legend_rows(fig: Figure) -> list[list[str]]:
    """Get the labels of the one legend of a figure in its rows, as they are read.

    The blank entries which fill a row are left out.
    """
    fig.draw_without_rendering()
    [legend] = fig.legends
    # the centers of the texts of a row differ by a little, mathtext is taller
    entries = []
    for text in legend.get_texts():
        if text.get_text():
            extent = text.get_window_extent()
            center = 0.5 * (extent.y0 + extent.y1)
            entries.append((-center, extent.x0, extent.height, text.get_text()))
    rows: list[list[tuple[float, str]]] = []
    top = -np.inf
    for y, x, height, label in sorted(entries):
        if not rows or y - top > 0.5 * height:
            rows.append([])
            top = y
        rows[-1].append((x, label))
    return [[label for _, label in sorted(row)] for row in rows]


def _inside(inner: Any, outer: Any) -> bool:
    """Check whether a box lies inside another one."""
    return bool(
        outer.x0 <= inner.x0
        and inner.x1 <= outer.x1
        and outer.y0 <= inner.y0
        and inner.y1 <= outer.y1
    )


def test_the_figures_of_a_mapping_have_one_title_and_one_legend(
    tmp_path: Path, figures: dict[str, Figure]
) -> None:
    """The title is above the panels and the legend below them, never inside.

    A title per panel overlaps the next one for a long key, a legend inside
    a panel covers the data. The legend has the data first and every
    parameter set in a row of its own.
    """
    problem = _problem(
        [FitMappingCollection(experiment=DoseStudy, mappings=["fm_dose"])], tmp_path
    )
    report = FitReport(
        problem,
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
        [
            ParameterSet(sid="true", values={"V": TRUE_V}),
            ParameterSet(sid="near", values={"V": 1.05 * TRUE_V}),
        ],
    )
    report.plot_mapping(tmp_path)
    fig = figures["DoseStudy_fm_dose"]
    title = fig.get_suptitle().splitlines()
    assert title[0] == "DoseStudy fm_dose [training]"
    assert title[1].startswith("pk.cmax [mg/l]")
    assert all(ax.get_title() == "" for ax in fig.axes)
    assert all(ax.get_legend() is None for ax in fig.axes)
    # the data first, the entries of a set together, the lines of neither
    # set last
    read = [label for row in _legend_rows(fig) for label in row if label]
    assert read[0].startswith("data")
    for sid in ("true", "near"):
        at = [k for k, label in enumerate(read) if label.split(" ")[0] == sid]
        assert at == list(range(at[0], at[0] + len(at))), read
    assert read[-2:] == ["prediction = measurement", "2-fold"]

    figures.clear()
    report.show_titles = False
    report.plot_mapping(tmp_path)
    assert figures["DoseStudy_fm_dose"].get_suptitle() == ""


@pytest.mark.parametrize("n_sets", [1, 2, 4])
@pytest.mark.parametrize("length", [8, 60, 120])
def test_the_legend_of_a_mapping_fits_into_its_figure(
    tmp_path: Path, figures: dict[str, Figure], n_sets: int, length: int
) -> None:
    """The legend is inside the figure and clear of the panels for long set ids.

    Two or three columns of long ids are wider than the figure: the legend
    takes fewer columns, and a figure is widened for an id which is wider
    than it in one column. The entries of a set stay together.
    """
    problem = _problem(
        [FitMappingCollection(experiment=DoseStudy, mappings=["fm_dose"])], tmp_path
    )
    sids = [f"set{j}_".ljust(length, "x") for j in range(n_sets)]
    report = FitReport(
        problem,
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
        [
            ParameterSet(sid=sid, values={"V": TRUE_V * (1.0 + 0.05 * j)})
            for j, sid in enumerate(sids)
        ],
    )
    report.plot_mapping(tmp_path)
    for name in ("DoseStudy_fm_dose",):
        fig = figures[name]
        rows = _legend_rows(fig)
        [legend] = fig.legends
        extent = legend.get_window_extent()
        assert _inside(extent, fig.bbox), name
        assert not any(ax.get_window_extent().overlaps(extent) for ax in fig.axes)
        assert _inside(fig.texts[0].get_window_extent(), fig.bbox)
        assert rows[0][0].startswith("data")
        read = [label for row in rows for label in row]
        for sid in sids:
            at = [k for k, label in enumerate(read) if label.split(" ")[0] == sid]
            assert at == list(range(at[0], at[0] + len(at))), (name, read)


def test_a_long_title_of_a_mapping_is_wrapped_between_its_words(
    tmp_path: Path, figures: dict[str, Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An identifier stays whole, a figure is widened for one wider than it."""
    problem = _problem(
        [FitMappingCollection(experiment=DoseStudy, mappings=["fm_dose"])], tmp_path
    )
    report = FitReport(
        problem,
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
        ParameterSet(sid="true", values={"V": TRUE_V}),
    )
    identifier = "fm_" + "-".join(["dose"] * 40)
    monkeypatch.setattr(
        FitReport, "mapping_title", lambda self, k: f"DoseStudy {identifier} [kind]"
    )
    report.plot_mapping(tmp_path)
    fig = figures["DoseStudy_fm_dose"]
    assert fig.get_suptitle().split("\n")[:3] == ["DoseStudy", identifier, "[kind]"]
    fig.draw_without_rendering()
    assert fig.get_figwidth() > 10.0
    assert _inside(fig.texts[0].get_window_extent(), fig.bbox)
    assert _inside(fig.legends[0].get_window_extent(), fig.bbox)


def test_the_log_panels_of_a_mapping_leave_out_zeros(
    tmp_path: Path, figures: dict[str, Figure]
) -> None:
    """The zero residuals of a perfect fit and negative ones are not in a log panel.

    A logarithmic axis clips them to the bottom of the panel, a line drops
    there; the error bars stay, also where they reach zero.
    """
    problem = _problem(
        [FitMappingCollection(experiment=DoseStudy, mappings=["fm_dose"])], tmp_path
    )
    report = FitReport(
        problem,
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
        [
            ParameterSet(sid="true", values={"V": TRUE_V}),
            ParameterSet(sid="near", values={"V": 1.05 * TRUE_V}),
        ],
    )
    report.plot_mapping(tmp_path)
    axes = figures["DoseStudy_fm_dose"].axes
    linear, log = axes[0], axes[1]
    assert (linear.get_yscale(), log.get_yscale()) == ("linear", "log")
    lines = {line.get_label(): line for line in linear.get_lines()}
    for masked in log.get_lines():
        y_log = np.asarray(masked.get_ydata(), dtype=float)
        assert not np.any(y_log <= 0.0)
        label = str(masked.get_label())
        if label.startswith("_"):
            continue
        y = np.asarray(lines[label].get_ydata(), dtype=float)
        np.testing.assert_array_equal(np.where(y > 0, y, np.nan), y_log)
    assert axes[1].collections, "the error bars of the data"


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
        report.problem,
        "y_references",
        [-np.abs(y) for y in report.problem.y_references],
    )
    report.plot_mapping(tmp_path)
    assert [ax.get_yscale() for ax in figures["RowsStudy_fm_rows"].axes[:2]] == [
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


def test_the_figure_of_a_dimension_draws_what_the_legend_lists(
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
    report.plot_mapping(tmp_path)
    for ax in figures["OneDoseStudy_fm_one"].axes[:2]:
        [line] = [line for line in ax.get_lines() if line.get_label() == "near"][:1]
        assert line.get_marker() not in (None, "None", "")


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
