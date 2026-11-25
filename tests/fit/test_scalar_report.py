"""The report of a fit with values per simulation and values over a dimension."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from sbmlsim.fit import FitMappingCollection, FitParameter, FitSettings, ParameterSet
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.report import MAPPING_SIZE, FitReport
from sbmlsim.plot import Chart
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
    html = (results_dir / "index.html").read_text(encoding="utf-8")
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
    html = (results_dir / "index.html").read_text(encoding="utf-8")
    assert html.count("fm_cmax") >= 2


@pytest.fixture
def figures(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Chart]]:
    """The figures a report saves, by the name of their file, as charts."""
    saved: dict[str, Chart] = {}

    def save(self: FitReport, chart: Chart, path: Path) -> None:
        saved[path.stem] = chart

    monkeypatch.setattr(FitReport, "_save_chart", save)
    yield saved


def _plots(chart: Chart) -> list[dict[str, Any]]:
    """Get the plots of the spec of a chart."""
    return chart.spec()["plots"]


def _curves(plot: dict[str, Any]) -> list[dict[str, Any]]:
    """Get the curves of a plot of a spec."""
    return [e for e in plot["elements"] if e["kind"] == "curve"]


def _labelled(plot: dict[str, Any], label: str) -> list[dict[str, Any]]:
    """Get the curves of a plot of a spec with a label."""
    return [e for e in _curves(plot) if e.get("label") == label]


def _geometry(chart: Chart) -> dict[str, Any]:
    """Get the geometry of a chart as the renderer draws it."""
    pytest.importorskip("sbmlsim_core")
    from sbmlsim.engine import import_core

    return import_core().figure_geometry(chart.to_json())


def _overlaps(a: dict[str, float], b: dict[str, float]) -> bool:
    """Check whether two rectangles of a geometry overlap."""
    return (
        a["x"] < b["x"] + b["width"]
        and b["x"] < a["x"] + a["width"]
        and a["y"] < b["y"] + b["height"]
        and b["y"] < a["y"] + a["height"]
    )


def test_the_figures_of_rows_and_of_a_dimension_of_one_value(
    tmp_path: Path, figures: dict[str, Chart]
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
    for plot in _plots(figures["RowsStudy_fm_rows"])[:2]:
        if plot["y"]["scale"] == "log":
            # the top marker has room above it, also when the rows span less
            # than a decade
            assert plot["y"]["min"] < rows.min()
            assert plot["y"]["max"] > 1.2 * rows.max()
        assert plot["y"]["label"] == "pk.cmax [mg/l]"
        # the rows have no x, no ticks
        assert not plot["x"]["ticks_visible"]
        [simulation] = _labelled(plot, "near")
        # next to every row, not on top of it
        x_sim = np.asarray(simulation["x"], dtype=float)
        assert np.all((x_sim > np.arange(3)) & (x_sim < np.arange(3) + 0.5))
        assert plot["x"]["min"] < 0.0 and plot["x"]["max"] > 2.5
    for plot in _plots(figures["OneDoseStudy_fm_one"])[:2]:
        assert plot["x"]["min"] < 100.0 < plot["x"]["max"]
    assert _plots(figures["OneDoseStudy_fm_one"])[0]["x"]["label"] == "PODOSE [mg]"


def test_the_figures_of_a_mapping_have_one_title_and_legends_right_of_rows(
    tmp_path: Path, figures: dict[str, Chart]
) -> None:
    """The title is above the panels and the legends right of the rows.

    A title per panel overlaps the next one for a long key, a legend inside
    a panel covers the data. The legend of the curves has the data first and
    then every parameter set, the legend of the agreement its lines.
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
    chart = figures["DoseStudy_fm_dose"]
    assert chart.title is not None
    title = chart.title.splitlines()
    assert title[0] == "DoseStudy fm_dose [training]"
    assert title[1].startswith("pk.cmax [mg/l]")
    plots = _plots(chart)
    assert all("title" not in plot for plot in plots)
    assert (chart.legend_scope, chart.legend_position) == ("row", "outside")
    entries = _geometry(chart)["legend_entries"]
    curves = [e["legend"] for e in entries if e["owner"] == "row:0"]
    assert curves == ["data", "true", "near"]
    lines = [e["legend"] for e in entries if e["owner"] == "row:1"]
    assert "prediction = measurement" in lines and "2-fold" in lines

    figures.clear()
    report.show_titles = False
    report.plot_mapping(tmp_path)
    assert figures["DoseStudy_fm_dose"].title is None


@pytest.mark.parametrize("n_sets", [1, 2, 4])
@pytest.mark.parametrize("length", [8, 60, 120])
def test_the_legend_of_a_mapping_fits_into_its_figure(
    tmp_path: Path, figures: dict[str, Chart], n_sets: int, length: int
) -> None:
    """The legend is inside the figure and clear of the panels for long set ids.

    The legends are right of the rows, which the renderer widens the figure
    for, so the panels keep their size.
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
    geometry = _geometry(figures["DoseStudy_fm_dose"])
    entries = geometry["legend_entries"]
    panels = [plot["rect"] for plot in geometry["plots"]]
    for entry in entries:
        rect = entry["rect"]
        assert rect["x"] + rect["width"] <= geometry["width"] + 0.5, entry
        assert rect["y"] + rect["height"] <= geometry["height"] + 0.5, entry
        assert not any(_overlaps(rect, panel) for panel in panels), entry
    curves = [e["legend"] for e in entries if e["owner"] == "row:0"]
    assert curves == ["data", *sids]


def test_a_long_title_of_a_mapping_is_wrapped_between_its_words(
    tmp_path: Path, figures: dict[str, Chart], monkeypatch: pytest.MonkeyPatch
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
    chart = figures["DoseStudy_fm_dose"]
    assert chart.title is not None
    assert chart.title.split("\n")[:3] == ["DoseStudy", identifier, "[kind]"]
    assert chart.width > MAPPING_SIZE[0]


def test_the_log_panels_of_a_mapping_leave_out_zeros(
    tmp_path: Path, figures: dict[str, Chart]
) -> None:
    """The zero residuals of a perfect fit and negative ones are not in a log panel.

    A logarithmic axis cannot show them, a line has a gap there; the error
    bars stay.
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
    linear, log = _plots(figures["DoseStudy_fm_dose"])[:2]
    assert (linear["y"]["scale"], log["y"]["scale"]) == ("linear", "log")
    for masked, curve in zip(_curves(log), _curves(linear), strict=True):
        y_log = np.array([np.nan if y is None else y for y in masked["y"]])
        assert not np.any(y_log <= 0.0)
        if "yerr" in curve:
            assert "yerr" in masked, "the error bars of the data"
            continue
        y = np.array([np.nan if v is None else v for v in curve["y"]])
        np.testing.assert_array_equal(np.where(y > 0, y, np.nan), y_log)


def test_a_mapping_without_positive_values_has_a_linear_log_panel(
    tmp_path: Path, figures: dict[str, Chart], monkeypatch: pytest.MonkeyPatch
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
        data["y_obsip"] = [-np.abs(y) for y in data["y_obsip"]]
        return data

    monkeypatch.setattr(FitReport, "residual_data", negative)
    monkeypatch.setattr(
        report.problem,
        "y_references",
        [-np.abs(y) for y in report.problem.y_references],
    )
    report.plot_mapping(tmp_path)
    plots = _plots(figures["RowsStudy_fm_rows"])[:2]
    assert [plot["y"]["scale"] for plot in plots] == ["linear", "linear"]


def test_the_cost_axis_of_a_perfect_fit_starts_at_zero(
    tmp_path: Path, figures: dict[str, Chart]
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
    x = _plots(figures["cost"])[0]["x"]
    assert x["scale"] == "linear"
    assert x["min"] == 0.0 and x["max"] > 0.0


def test_the_figure_of_a_dimension_draws_what_the_legend_lists(
    tmp_path: Path, figures: dict[str, Chart]
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
    for plot in _plots(figures["OneDoseStudy_fm_one"])[:2]:
        [line] = _labelled(plot, "near")
        assert line["style"]["marker"]["type"] != "none"


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
    limits = report._x_limits(0, [np.array([0.0, 2.0])])
    assert limits is not None
    left, right = limits
    assert np.isfinite(left) and np.isfinite(right) and left < 0.0 and right > 2.0
