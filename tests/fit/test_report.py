"""Test the report of a fit.

The report is separate from the optimization: it is created from the definition
of the problem, the settings and one or more parameter sets.
"""

import dataclasses
import json
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pypdf import PdfReader

import sbmlsim.fit.report as report_module
from examples.hctz_fitting.experiments.studies import Weir1998
from sbmlsim import __version__
from sbmlsim.fit import FitSettings, MappingKind, ParameterSet, ParameterSets
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.evaluation import MIN_POINTS_SUBSET_AGREEMENT, Grouping, Subset
from sbmlsim.fit.fisher import fisher_information
from sbmlsim.fit.methods import ReportContext
from sbmlsim.fit.metrics import fold_counts
from sbmlsim.fit.objects import EVALUATED_KINDS, FitMappingCollection, FitParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.report import (
    MAX_GROUP_ROWS,
    ROW_HEIGHT,
    STUDY_COLORS,
    FitReport,
    category_color,
)
from sbmlsim.fit.report_document import fit_document
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization
from sbmlsim.plot import Chart, ChartAxis
from sbmlsim.plot.colors import normalize_color


def _plots(chart: Chart) -> list[dict[str, Any]]:
    """Get the plots of the spec of a chart."""
    return chart.spec()["plots"]


def _of_kind(plot: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """Get the elements of a kind of a plot of a spec."""
    return [e for e in plot["elements"] if e["kind"] == kind]


def _labels(plot: dict[str, Any]) -> list[str]:
    """Get the labels of the elements of a plot of a spec."""
    return [e["label"] for e in plot["elements"] if "label" in e]


def _texts(plot: dict[str, Any]) -> list[str]:
    """Get the texts of a plot of a spec, e.g. the lines of a box of metrics."""
    return [e["text"] for e in _of_kind(plot, "text")]


def _error_bars(plot: dict[str, Any]) -> list[dict[str, Any]]:
    """Get the error bars of a plot: curves with errors or of segments.

    Symmetric errors are the errors of a curve, asymmetric ones segments of a
    line without markers broken by gaps, see `FitReport._draw_errors`.
    """
    return [
        e
        for e in _of_kind(plot, "curve")
        if "xerr" in e
        or "yerr" in e
        or (e["style"]["marker"]["type"] == "none" and None in e["y"])
    ]


def _n_bars(element: dict[str, Any]) -> int:
    """Get the number of the bars of an element of `_error_bars`."""
    if "xerr" in element or "yerr" in element:
        return len(element["x"])
    return sum(value is not None for value in element["y"]) // 2


def _fit(
    problem: OptimizationProblem, settings: FitSettings, size: int = 1
) -> OptimizationResult:
    """Run a small fit of the problem.

    A few steps of every run: a report shows a fit, whether it converged or
    not, see the fixture `short_fit`.
    """
    return run_optimization(
        problem=problem,
        settings=settings,
        size=size,
        n_cores=1,
        serial=True,
        seed=1234,
        max_nfev=3,
    )


def _assert_report_files(results_dir: Path) -> None:
    """Check the files every report writes."""
    assert (results_dir / "index.html").exists()
    assert (results_dir / "report.txt").exists()
    assert (results_dir / "parameters.json").exists()
    assert list((results_dir / "plots").glob("*.svg"))


def test_report_from_optimization_result(
    tmp_path: Path, op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The report of a fit shows the fitted parameters alone."""
    opt_result = _fit(op_hctz_iv, fit_settings)
    report = FitReport.from_optimization_result(
        problem=op_hctz_iv,
        opt_result=opt_result,
        mapping_figures=False,
    )

    # the values the model started from are not reported
    assert len(report.parameter_sets) == 1
    assert report.reference_set.sid != "model"

    results_dir = report.create(output_dir=tmp_path, name="fit")
    _assert_report_files(results_dir)
    # the plots of the runs need the result of an optimization
    assert (results_dir / "plots" / "traces.svg").exists()
    assert (results_dir / "optimization_result.json").exists()

    # a single set is not compared against another one
    assert not (results_dir / "plots" / "cost_scatter.svg").exists()


def test_report_from_optimization_result_with_model(
    tmp_path: Path, op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """`with_model` compares the fit against the values the model started from."""
    opt_result = _fit(op_hctz_iv, fit_settings)
    report = FitReport.from_optimization_result(
        problem=op_hctz_iv,
        opt_result=opt_result,
        with_model=True,
        mapping_figures=False,
    )

    # the model values are the reference, the fit is the second set
    assert len(report.parameter_sets) == 2
    assert report.reference_set.sid == "model"
    # no parameter set is black, which is the color of the reference data
    assert "black" not in {report.color(pset) for pset in report.parameter_sets}

    results_dir = report.create(output_dir=tmp_path, name="fit_with_model")
    _assert_report_files(results_dir)
    assert (results_dir / "plots" / "cost_scatter.svg").exists()


def test_the_report_shows_the_target_of_a_versioned_parameter(
    tmp_path: Path, definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """A versioned fit says which entity a parameter writes.

    `definition_hctz_iv.parameters[0].pid` is already `Ka_dis_hctz` (see
    `examples/hctz_fitting/fitting/parameters.py`), so renaming the parameter
    while pointing its `target` at that same entity is what actually produces
    a `target_id != pid` row; setting `.target` alone would be a no-op.
    """
    definition = dataclasses.replace(
        definition_hctz_iv,
        parameters=[
            FitParameter("Ka_po", 1.0, 1e-4, 10.0, "1/hr", target="Ka_dis_hctz"),
        ],
    )
    problem = definition.problem(opid="hctz_iv_versioned")
    problem.initialize(fit_settings)
    report = FitReport(
        problem=problem,
        settings=fit_settings,
        parameter_sets=problem.parameter_set_model(),
        mapping_figures=False,
    )
    html = (report.create(tmp_path, name="versioned") / "index.html").read_text(
        encoding="utf-8"
    )
    assert ">target</th>" in html
    assert "Ka_dis_hctz" in html


def test_the_report_of_an_ordinary_fit_has_no_target_column(
    tmp_path: Path, op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A fit whose parameters are not versioned gets no target column."""
    op_hctz_iv.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_iv,
        settings=fit_settings,
        parameter_sets=op_hctz_iv.parameter_set_model(),
        mapping_figures=False,
    )
    html = (report.create(tmp_path, name="ordinary") / "index.html").read_text(
        encoding="utf-8"
    )
    assert ">target</th>" not in html
    # a fit without networks has no section of them, and no line in its place
    assert "Networks" not in html
    assert "  </div>\n\n  <h3>Settings</h3>" in html


def test_the_report_shows_every_data_path(
    tmp_path: Path, op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A problem with several data paths shows the paths, one per line."""
    data_path = op_hctz_iv.data_path
    assert isinstance(data_path, Path)
    problem = OptimizationProblem(
        opid=op_hctz_iv.opid,
        mapping_collections=op_hctz_iv.mapping_collections,
        fit_parameters=op_hctz_iv.parameters,
        base_path=op_hctz_iv.base_path,
        data_path=[data_path, data_path.parent],
    )
    problem.initialize(fit_settings)
    report = FitReport(
        problem=problem,
        settings=fit_settings,
        parameter_sets=problem.parameter_set_model(),
        mapping_figures=False,
    )
    info = report.fit_info()
    assert info["data path"] == f"{data_path}\n{data_path.parent}"
    assert info["base path"] == str(op_hctz_iv.base_path)
    html = (report.create(tmp_path, name="paths") / "index.html").read_text(
        encoding="utf-8"
    )
    assert "PosixPath" not in html and "WindowsPath" not in html


def test_the_experiments_are_those_with_fit_mappings(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """An experiment whose mappings are all excluded is not listed."""
    excluded = FitMappingCollection(experiment=Weir1998, kind=MappingKind.EXCLUDED)
    problem = OptimizationProblem(
        opid=op_hctz_iv.opid,
        mapping_collections=[*op_hctz_iv.mapping_collections, excluded],
        fit_parameters=op_hctz_iv.parameters,
        base_path=op_hctz_iv.base_path,
        data_path=op_hctz_iv.data_path,
    )
    problem.initialize(fit_settings)
    report = FitReport(
        problem=problem,
        settings=fit_settings,
        parameter_sets=problem.parameter_set_model(),
        mapping_figures=False,
    )
    assert report.fit_info()["experiments"] == "Beermann1976"


def test_a_metric_which_is_not_defined_is_a_dash(
    tmp_path: Path, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The cost of the validation data is not defined, it is `-` and not `nan`."""
    op_hctz_pk.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    context = report.html_context(tmp_path, "report")
    names = [column["name"] for column in context["metrics_columns"]]
    rows = {row[names.index("kind")]: row for row in context["metrics"]}
    assert rows["validation"][names.index("cost")] == "-"
    assert rows["training"][names.index("cost")] != "-"
    cells = [cell for row in context["metrics"] for cell in row]
    cells += [
        value for row in context["mapping_metrics"] for value in map(str, row.values())
    ]
    assert "nan" not in cells


def test_the_subsets_of_the_data_points(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The panels are `all` and every kind the problem has, in that order."""
    op_hctz_pk.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    # one panel per kind, there is no panel over all data points
    assert report.point_kinds() == ["training", "validation", "outlier"]

    points = report.points(report.reference_set)
    assert set(points.kind) == {"training", "validation", "outlier"}
    assert len(report._of_kind(points, "outlier")) > 0
    assert sum(
        len(report._of_kind(points, kind)) for kind in report.point_kinds()
    ) == len(points)

    # the studies are colored, every study has its own color
    assert report.studies() == ["Beermann1976", "Patel1984", "Weir1998"]
    colors = {study: report.study_color(study) for study in report.studies()}
    assert len(set(colors.values())) == len(colors)
    assert "black" not in set(colors.values())


def test_the_subsets_of_a_problem_with_one_kind(
    definition_hctz_pk: FitDefinition, fit_settings: FitSettings
) -> None:
    """A problem with a single kind has one panel, not `all` and the kind."""
    training = [
        collection
        for collection in definition_hctz_pk.collections()
        if collection.kind is MappingKind.TRAINING
    ]
    problem = definition_hctz_pk.problem(opid="training", mapping_collections=training)
    problem.initialize(fit_settings)
    report = FitReport(
        problem=problem,
        settings=fit_settings,
        parameter_sets=problem.parameter_set_model(),
        mapping_figures=False,
    )
    assert report.point_kinds() == ["training"]


def test_the_limits_of_agreement_are_the_training_data(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The Bland-Altman limits come from the training data, not from a panel.

    They are drawn in every panel, so they must not be the agreement of the
    panel: the outliers are far away by definition and would widen them.
    """
    op_hctz_pk.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    pset = report.reference_set
    bias, half = report.agreement(pset)
    assert half > 0.0

    # the same numbers as the training data of the report
    points = report.points(pset)
    training = report._of_kind(points, MappingKind.TRAINING.value)
    _mean, difference, _mask = report._log_ratio(training)
    assert bias == pytest.approx(float(np.mean(difference)))
    assert half == pytest.approx(1.96 * float(np.std(difference, ddof=1)))

    # and not the agreement of all data points, which the outliers widen
    _mean, all_difference, _mask = report._log_ratio(points)
    assert 1.96 * float(np.std(all_difference, ddof=1)) > half


def test_the_title_of_a_panel_counts_the_points_drawn() -> None:
    """A panel of the data points states how many of its points it draws.

    The axes of both figures are logarithmic, so a point with a value or a
    prediction which is zero or negative is not drawn; the title then states
    the points drawn of the points of the kind, the denominator of the
    percentage within twofold.
    """
    positive = pd.DataFrame(
        {"kind": ["training"] * 3, "DV": [1.0, 2.0, 3.0], "IPRED": [1.0, 2.0, 3.0]}
    )
    assert FitReport._panel_title("training", [positive]) == "training (n=3)"
    with_zero = positive.assign(IPRED=[1.0, 0.0, -3.0])
    assert FitReport._panel_title("training", [with_zero]) == "training (n=1 of 3)"
    # of several sets the one with the most points drawn, of the reference set
    assert FitReport._panel_title("training", [with_zero, positive]) == "training (n=3)"
    assert FitReport._panel_title("validation", [positive]) == "validation (n=0)"


def test_the_legend_lists_the_studies_before_the_lines(
    tmp_path: Path,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    saved_figures: dict[str, Chart],
) -> None:
    """The legend of the panels lists the points first and the lines last.

    The band and the lines of twofold are labelled in the last panel, and a
    study whose points are only in a later panel, e.g. a study with validation
    data alone, is still listed with the other studies and not after the
    lines: the renderer lists the entries of a row in the order of its panels.
    """
    pytest.importorskip("sbmlsim_core")
    from sbmlsim.engine import import_core

    report = _initialized_report(op_hctz_pk, fit_settings)
    for name in ("goodness_of_fit", "bland_altman"):
        getattr(report, f"plot_{name}")(tmp_path / f"{name}.svg")
        chart = saved_figures[name]
        assert chart.legend_scope == "row" and chart.legend_position == "outside"
        plots = _plots(chart)
        for plot in plots[:-1]:
            assert "2-fold" not in _labels(plot)
        entries = import_core().figure_geometry(chart.to_json())["legend_entries"]
        legends = [entry["legend"] for entry in entries]
        studies = report.studies()
        assert legends[: len(studies)] == studies, legends
        assert "2-fold" in legends and "prediction = measurement" in legends


def test_both_figures_draw_the_same_band(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The goodness of fit and the Bland-Altman plot show one agreement.

    The band is a horizontal line in the one figure and a line parallel to
    the diagonal in the other, which on logarithmic axes is the same thing,
    so both are drawn from `agreement` and carry the same legend entries.
    """
    op_hctz_pk.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    pset = report.reference_set
    identity, bias_label, limits_label = report._band_labels(pset)
    assert identity == "prediction = measurement"
    assert bias_label.startswith("bias ")
    assert limits_label.startswith("LoA ")

    # the labels carry the numbers of `agreement`, so both figures state them
    bias, half = report.agreement(pset)
    assert f"{10**bias:.2f}x" in bias_label
    assert f"{10 ** (bias - half):.2f}-{10 ** (bias + half):.2f}x" in limits_label


def test_the_goodness_of_fit_and_altman_plots(
    tmp_path: Path, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Both figures are created and reported with a panel per subset."""
    op_hctz_pk.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    results_dir = report.create(output_dir=tmp_path, name="subsets")

    for name in ["goodness_of_fit", "bland_altman"]:
        assert (results_dir / "plots" / f"{name}.svg").exists()
    # the relative residuals are gone, the Bland-Altman plot is the same
    # information with a reference to read it against
    assert not (results_dir / "plots" / "residual_scatter.svg").exists()

    html = (results_dir / "index.html").read_text(encoding="utf-8")
    assert "plots/goodness_of_fit.svg" in html
    assert "plots/bland_altman.svg" in html
    assert "residual_scatter" not in html
    # the figures are inline with their spec, as in the report of experiments
    for name in ["goodness_of_fit", "bland_altman", "cost_bar"]:
        assert f'<figure data-figure="{name}">' in html, name
        assert f'data-figure-spec="{name}"' in html, name
        assert (results_dir / "plots" / f"{name}.json").exists(), name
    assert "<svg" in html and '<img loading="lazy"' not in html
    # the script of the report beside its page makes the figures interactive
    assert html.count('<script src="figures.js" defer></script>') == 1
    assert "hydrateFigures" in (results_dir / "figures.js").read_text(encoding="utf-8")
    # the metrics of the report cover the outliers
    metrics = (results_dir / "metrics.tsv").read_text(encoding="utf-8")
    assert MappingKind.OUTLIER.value in metrics
    # both figures cover a full row of the report, the other plots do not
    context = report.html_context(results_dir, name="subsets")
    plots = context["result_plots"] + context["mapping_plots"]
    wide = {plot["src"]: plot["wide"] for plot in plots}
    assert wide["plots/goodness_of_fit.svg"]
    assert wide["plots/bland_altman.svg"]
    assert not wide["plots/residual_boxplot.svg"]
    assert 'class="card wide"' in html


def test_the_key_metrics_of_the_panels(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Every panel carries the key metrics of its points."""
    op_hctz_pk.initialize(fit_settings)
    pset = op_hctz_pk.parameter_set_model()
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=pset,
        mapping_figures=False,
    )

    kind = MappingKind.TRAINING.value
    summary = report.metrics(pset).summary(kind=MappingKind.TRAINING)
    gof = report.panel_metrics(kind, "goodness_of_fit")
    assert f"R² = {summary['R2']:.3f}" in gof
    assert f"NRMSE = {summary['NRMSE']:.3g}" in gof
    # a single set has no prefix
    assert not gof.startswith(pset.sid)

    # the agreement of the training panel is the band, so its bias is the bias
    # of the band and its points are inside the limits
    bias, _half = report.agreement(pset)
    altman = report.panel_metrics(kind, "bland_altman")
    assert f"bias = {10**bias:.2f}x" in altman
    assert "in LoA = " in altman
    inside = int(altman.split("in LoA = ")[1].rstrip("%"))
    assert inside >= 90

    with pytest.raises(ValueError, match="Unknown plot"):
        report.panel_metrics(kind, "other")


def test_the_twofold_criterion_of_a_report(fitted_report) -> None:
    """The percentage of the points within twofold is a metric of the report.

    It is a column of the metrics per kind and per mapping, of their TSV files
    and of their tables, and the goodness of fit states it in the box of a
    panel and in its caption.
    """
    report, plots_dir = fitted_report
    results_dir = plots_dir.parent
    metrics = report.metrics_df()
    mappings = report.metrics_mappings_df()
    assert "within_2fold" in metrics.columns
    assert "within_2fold" in mappings.columns
    assert list(metrics.columns).index("within_2fold") == (
        list(metrics.columns).index("R2") + 1
    )
    assert (results_dir / "metrics_observables.tsv").exists()
    for name in ["metrics.tsv", "metrics_mappings.tsv", "metrics_observables.tsv"]:
        header = (results_dir / name).read_text(encoding="utf-8").splitlines()[0]
        assert "within_2fold" in header.split("\t")

    pset = report.reference_set
    training = report.metrics(pset).summary(kind=MappingKind.TRAINING)
    percent = f"{training['within_2fold']:.1f}%"
    box = report.panel_metrics(MappingKind.TRAINING.value, "goodness_of_fit")
    assert f"2-fold = {percent}" in box

    # the statement names the points it counts, the points with a ratio
    statement = report.twofold_statement()
    assert statement.startswith("Within twofold of the data are ")
    points = report.points(pset)
    for kind in report.point_kinds():
        of_kind = points[points.kind == kind]
        inside, n_ratio = fold_counts(of_kind.DV, of_kind.IPRED)
        value = report.metrics(pset).summary(kind=MappingKind(kind))["within_2fold"]
        assert value == pytest.approx(100.0 * inside / n_ratio)
        assert (
            f"{inside} of the {n_ratio} {kind} data points ({value:.1f}%)" in statement
        )
    assert "with a positive value and prediction" in statement

    caption = report.plot_caption("goodness_of_fit")
    assert caption == f"{report.PLOT_CAPTIONS['goodness_of_fit']}. {statement}"
    # the other figures keep their caption
    assert report.plot_caption("waterfall") == report.PLOT_CAPTIONS["waterfall"]

    context = report.html_context(results_dir, name="twofold")
    names = [column["name"] for column in context["metrics_columns"]]
    assert "2-fold [%]" in names
    rows = {row[names.index("kind")]: row for row in context["metrics"]}
    assert rows["training"][names.index("2-fold [%]")] == f"{percent[:-1]}"
    assert "2-fold [%]" in context["mapping_metrics_columns"]
    assert "2-fold [%]" in context["numeric_columns"]
    assert context["hints"]["2-fold [%]"] == report.HINTS["within_2fold"]
    assert all("within_2fold" in row for row in context["mapping_metrics"])
    (plot,) = [
        p
        for p in context["result_plots"]
        if p["src"].endswith(f"goodness_of_fit.{report.image_format}")
    ]
    assert plot["caption"] == caption

    html = (results_dir / "index.html").read_text(encoding="utf-8")
    assert "2-fold [%]" in html
    assert statement in html


def test_the_key_metrics_of_several_sets(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """With several sets the metrics of a set follow a line of its id."""
    op_hctz_pk.initialize(fit_settings)
    model = op_hctz_pk.parameter_set_model()
    other = ParameterSet(
        sid="other",
        values={pid: value * 1.5 for pid, value in model.values.items()},
    )
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=ParameterSets([model, other]),
        mapping_figures=False,
    )
    text = report.panel_metrics(MappingKind.TRAINING.value, "goodness_of_fit")
    lines = text.splitlines()
    # a line of the id of every set and two lines of its metrics
    assert len(lines) == 6
    assert lines[0] == f"{model.sid}:"
    assert lines[3] == "other:"


def test_the_figures_of_every_mapping(
    tmp_path: Path,
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
) -> None:
    """A report draws the figure of every fit mapping by default.

    They are almost the whole cost of a report, so the tests which only read
    its tables create it with `mapping_figures=False`; this is the test that
    the default still draws them and that the cards carry them.
    """
    op_hctz_iv.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_iv,
        settings=fit_settings,
        parameter_sets=op_hctz_iv.parameter_set_model(),
    )
    results_dir = report.create(output_dir=tmp_path, name="figures")

    plots = results_dir / "plots"
    for k, mapping_id in enumerate(op_hctz_iv.mapping_keys):
        sid = op_hctz_iv.experiment_keys[k]
        assert (plots / f"{sid}_{mapping_id}.svg").exists()
        assert (plots / f"{sid}_{mapping_id}.json").exists()
        # one figure per mapping, the residual figure is gone
        assert not (plots / f"fit_{sid}_{mapping_id}.svg").exists()

    html = (results_dir / "index.html").read_text(encoding="utf-8")
    assert "Data, simulation and agreement" in html


def test_a_report_without_the_figures_of_the_mappings(
    tmp_path: Path, op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """`mapping_figures=False` keeps the tables and drops the mapping figures."""
    op_hctz_iv.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_iv,
        settings=fit_settings,
        parameter_sets=op_hctz_iv.parameter_set_model(),
        mapping_figures=False,
    )
    results_dir = report.create(output_dir=tmp_path, name="tables")
    plots = results_dir / "plots"

    for k, mapping_id in enumerate(op_hctz_iv.mapping_keys):
        sid = op_hctz_iv.experiment_keys[k]
        assert not (plots / f"{sid}_{mapping_id}.svg").exists()
        assert not (plots / f"fit_{sid}_{mapping_id}.svg").exists()

    # the overview figures and everything which is read as a table stay
    assert (plots / "goodness_of_fit.svg").exists()
    assert (plots / "bland_altman.svg").exists()
    _assert_report_files(results_dir)
    html = (results_dir / "index.html").read_text(encoding="utf-8")
    for mapping_id in op_hctz_iv.mapping_keys:
        assert mapping_id in html


def test_report_without_optimization(
    tmp_path: Path, op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A report is created from stored parameters, without running a fit."""
    opt_result = _fit(op_hctz_iv, fit_settings)
    parameters_path = tmp_path / "parameters.json"
    opt_result.parameter_sets(size=1).to_json(path=parameters_path)

    # a fresh problem, nothing was optimized on it
    problem = OptimizationProblem(
        opid=op_hctz_iv.opid,
        mapping_collections=op_hctz_iv.mapping_collections,
        fit_parameters=op_hctz_iv.parameters,
        base_path=op_hctz_iv.base_path,
        data_path=op_hctz_iv.data_path,
    )
    report = FitReport(
        problem=problem,
        settings=fit_settings,
        parameter_sets=ParameterSets.from_json(parameters_path),
        mapping_figures=False,
    )

    results_dir = report.create(output_dir=tmp_path, name="report")
    _assert_report_files(results_dir)
    # no optimization, so no plots of the runs
    assert not (results_dir / "plots" / "traces.svg").exists()
    assert not (results_dir / "optimization_result.json").exists()


def test_report_multiple_parameter_sets(
    tmp_path: Path, op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Several parameter sets are compared in a single report."""
    opt_result = _fit(op_hctz_iv, fit_settings, size=2)
    sets = ParameterSets(
        [
            op_hctz_iv.parameter_set_model(),
            *opt_result.parameter_sets(size=2),
        ]
    )
    report = FitReport(
        problem=op_hctz_iv,
        settings=fit_settings,
        parameter_sets=sets,
        mapping_figures=False,
    )
    assert len(report.parameter_sets) == 3

    results_dir = report.create(output_dir=tmp_path, name="compare")
    _assert_report_files(results_dir)
    # the costs of the sets are compared against the reference set
    assert (results_dir / "plots" / "cost_scatter.svg").exists()

    # every set is a column of the parameter table
    df = report.parameter_sets.to_df()
    assert len(df.columns) == 2 + 3


def test_report_single_set(
    tmp_path: Path, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A report of a single set has nothing to compare it against."""
    op_hctz_pk.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    results_dir = report.create(output_dir=tmp_path, name="model")
    _assert_report_files(results_dir)
    assert not (results_dir / "plots" / "cost_scatter.svg").exists()


def test_report_requires_a_set(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A report without a parameter set is an error."""
    with pytest.raises(ValueError, match="At least one"):
        FitReport(
            problem=op_hctz_pk,
            settings=fit_settings,
            parameter_sets=[],
            mapping_figures=False,
        )


def test_report_parameter_set_of_other_problem(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A set which does not cover the parameters of the problem is reported."""
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=ParameterSet(sid="other", values={"unknown": 1.0}),
        mapping_figures=False,
    )
    with pytest.raises(KeyError, match="does not contain"):
        report.residual_data(report.reference_set)


def test_run_plots_require_a_result(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The traces and the waterfall plot need an optimization result."""
    op_hctz_pk.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    with pytest.raises(ValueError, match="OptimizationResult"):
        _ = report.opt_result_required


def _html_of(results_dir: Path) -> str:
    """Read the HTML report."""
    return (results_dir / "index.html").read_text(encoding="utf-8")


def test_html_report_sections(
    tmp_path: Path, op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The report has the three sections and its search."""
    opt_result = _fit(op_hctz_iv, fit_settings)
    report = FitReport.from_optimization_result(
        problem=op_hctz_iv,
        opt_result=opt_result,
        mapping_figures=False,
    )
    results_dir = report.create(output_dir=tmp_path, name="fit")
    html = _html_of(results_dir)

    for token in [
        'id="overview"',
        'id="results"',
        'id="mappings"',
        'id="search"',
        'id="lightbox"',
        'data-kind="training"',
    ]:
        assert token in html, token

    # one card per fit mapping
    assert html.count('class="card mapping"') == len(op_hctz_iv.mapping_keys)
    # the tables can be sorted
    assert "data-sort=" in html


def test_html_report_is_offline(
    tmp_path: Path, op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The report needs no network, it is read from a file and archived."""
    opt_result = _fit(op_hctz_iv, fit_settings)
    report = FitReport.from_optimization_result(
        problem=op_hctz_iv,
        opt_result=opt_result,
        mapping_figures=False,
    )
    html = _html_of(report.create(output_dir=tmp_path, name="fit"))
    assert "http://" not in html
    assert "https://" not in html


def test_html_report_references_exist(
    tmp_path: Path, op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Every file the report links to was written."""
    opt_result = _fit(op_hctz_iv, fit_settings)
    report = FitReport.from_optimization_result(
        problem=op_hctz_iv,
        opt_result=opt_result,
        mapping_figures=False,
    )
    results_dir = report.create(output_dir=tmp_path, name="fit")

    references = re.findall(r'(?:src|href)="([^"#:]+)"', _html_of(results_dir))
    assert references
    missing = [ref for ref in references if not (results_dir / ref).exists()]
    assert missing == []


def test_html_report_without_optimization(
    tmp_path: Path, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A report of stored parameters has no plots of the runs."""
    op_hctz_pk.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    html = _html_of(report.create(output_dir=tmp_path, name="model"))
    assert "traces" not in html
    assert "optimization_result.json" not in html
    # the sections which do not depend on a fit are there
    assert 'id="overview"' in html
    assert 'id="mappings"' in html


def test_html_context(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The context of the template carries the parts of the report."""
    op_hctz_pk.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    context = report.html_context(results_dir=Path("nowhere"), name="report")

    # the heading is the id of the fit, not the directory of the report
    assert context["fit_id"] == op_hctz_pk.opid
    assert context["opid"] == ""
    # the data the model does not describe is not part of the report
    assert context["kinds"] == [kind.value for kind in EVALUATED_KINDS]
    assert MappingKind.EXCLUDED.value not in context["kinds"]
    assert len(context["parameters"]) == len(op_hctz_pk.parameters)
    assert len(context["mappings"]) == len(op_hctz_pk.mapping_keys)
    assert context["settings"]["residual"] == fit_settings.residual.name
    assert context["data_total"]["total"] == len(op_hctz_pk.mapping_keys)
    # every mapping carries its kind and its metrics
    for mapping in context["mappings"]:
        assert mapping["kind"] in {kind.value for kind in MappingKind}
        assert set(mapping["metrics"]) == {"n", "RMSE", "R²", "2-fold", "GMFE"}


def test_html_report_is_well_formed(
    tmp_path: Path, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The tags of the report are balanced."""
    op_hctz_pk.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    html = _html_of(report.create(output_dir=tmp_path, name="model"))

    void = {"img", "br", "hr", "meta", "link", "input", "source", "col"}
    stack: list[str] = []
    errors: list[str] = []

    class Checker(HTMLParser):
        def handle_starttag(self, tag: str, attrs: object) -> None:
            if tag not in void:
                stack.append(tag)

        def handle_endtag(self, tag: str) -> None:
            if tag in void:
                return
            if not stack or stack[-1] != tag:
                errors.append(tag)
            else:
                stack.pop()

    Checker(convert_charrefs=True).feed(html)
    assert errors == []
    assert stack == []


def test_the_report_explains_its_values(
    tmp_path: Path, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Every value of the report says what it means, as a tooltip."""
    op_hctz_pk.initialize(fit_settings)
    parameter_set = op_hctz_pk.parameter_set_model()
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=[parameter_set],
        mapping_figures=False,
    )
    report.create(tmp_path, name="hints")
    html = (tmp_path / "hints" / "index.html").read_text(encoding="utf-8")

    # the metrics of the summary, i.e. what the tables of the report show
    for column in ["MSE", "RMSE", "R2", "AIC", "BIC"]:
        assert f'{column}<span class="hint"' in html, column
    # and the figures say what to look for in them
    assert html.count('class="hint"') > 5
    assert "Bayesian information criterion" in html


def test_the_report_of_the_fisher_information(
    tmp_path: Path, op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The Fisher information is a section with its table and correlations."""
    op_hctz_iv.initialize(fit_settings)
    parameter_set = op_hctz_iv.parameter_set_model()
    fisher = fisher_information(op_hctz_iv, fit_settings, parameter_set)
    report = FitReport(
        problem=op_hctz_iv,
        settings=fit_settings,
        parameter_sets=[parameter_set],
        fisher=fisher,
        mapping_figures=False,
    )
    results_dir = report.create(tmp_path, name="fisher")

    # the information is stored next to the report
    assert (results_dir / "fisher.json").exists()
    assert (results_dir / "fisher.tsv").exists()

    html = (results_dir / "index.html").read_text(encoding="utf-8")
    assert "Fisher information" in html
    assert "Correlation" in html
    for pid in op_hctz_iv.pids:
        assert pid in html
    # the intravenous data does not determine every parameter, and the report
    # says so instead of showing errors which cannot be read
    assert not fisher.is_identifiable
    assert "does not have full rank" in html
    # the columns of text are left aligned, the numbers right aligned
    section = html[html.index('id="identifiability"') :]
    assert '<th data-sort="text">unit' in section
    assert '<th class="num" data-sort="num">se' in section
    assert "<td>dimensionless</td>" in section or "<td>1/" in section


def test_the_settings_state_the_absolute_tolerance(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The settings name the absolute tolerance, the report lists no states."""
    op_hctz_pk.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    context = report.html_context(Path("nowhere"), "report")
    assert "concentration" in context["settings"]["absolute tolerance"]
    assert "tolerances" not in context


# ---------------------------------------------------------------------------
# the report document: the two tabs of the HTML, Typst, PDF and LaTeX
# ---------------------------------------------------------------------------
#: the figures of the report document, which LaTeX includes as PNG
DOCUMENT_FIGURES = [
    "traces",
    "waterfall",
    "goodness_of_fit",
    "bland_altman",
    "profiles",
]


def test_report_writes_all_formats(report_dir: Path) -> None:
    for name in [
        "index.html",
        "report.typ",
        "report.pdf",
        "report.tex",
        "references.bib",
    ]:
        assert (report_dir / name).exists(), name
    html = (report_dir / "index.html").read_text(encoding="utf-8")
    assert 'data-tab="report"' in html and 'data-tab="mappings"' in html
    assert "<math" in html


def test_a_failed_pdf_is_stated_with_its_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fitted_report
) -> None:
    """The report states why the PDF failed, every other output is written.

    The fit is kept: the report has its Typst source and no PDF.
    """

    def broken(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("typst could not compile 'report.typ': boom")

    monkeypatch.setattr(report_module, "compile_pdf", broken)
    report, _ = fitted_report
    out = report.create(output_dir=tmp_path, name="failed")
    assert not (out / "report.pdf").exists()
    html = _html_of(out)
    assert "PDF could not be created" in html
    assert "typst could not compile &#39;report.typ&#39;: boom" in html
    # the page does not link the PDF which is not there
    assert 'href="report.pdf"' not in html
    for name in [
        "report.typ",
        "report.tex",
        "report_standalone.tex",
        "references.bib",
        "references.csl",
        "parameters.json",
        "optimization_result.json",
    ]:
        assert (out / name).exists(), name


def test_the_report_tab_holds_the_document(report_dir: Path, fitted) -> None:
    """The page opens on the report: the document, then the overview."""
    html = _html_of(report_dir)
    nav = html[html.index('<nav class="sections">') : html.index("</nav>")]
    assert nav.index('data-tab="report"') < nav.index('data-tab="mappings"')
    assert "Overview" not in nav

    # the report tab is the one shown, the document comes first in it
    report = html.index('id="report"')
    mappings = html.index('id="mappings"')
    assert '<div id="report" class="tab-panel active"' in html
    assert report < html.index("<h2>Methods</h2>") < html.index('id="overview"')
    assert html.index('id="overview"') < html.index('id="results"') < mappings
    assert html.index('id="identifiability"') < mappings
    assert html.index('<section id="references">') < html.index('id="overview"')
    # the mappings tab holds every fit mapping and nothing of the report
    assert html.index('class="card mapping"') > mappings
    problem, _ = fitted
    assert html.count('class="card mapping"') == len(problem.mapping_keys)
    assert "PDF could not be created" not in html

    # the document numbers its parts and links its references in the page
    assert "Table 1." in html and "Figure 1." in html
    assert '<span class="eqno">(1)</span>' in html
    for label in re.findall(r'<a href="#([^"]+)">', html):
        assert f'id="{label}"' in html, label
    # its figures open full size as the others do, the downloads are linked
    assert '<figure id="fig:traces' in html
    for name in ["report.pdf", "report.typ", "report.tex", "references.bib"]:
        assert f'href="{name}"' in html, name


def test_the_document_survives_the_escaping_of_the_page(report_dir: Path) -> None:
    """The MathML and the tables of the document are markup, not text."""
    html = _html_of(report_dir)
    assert "&lt;math" not in html and "&lt;table" not in html
    assert '<math display="block">' in html
    # the namespace of MathML is the one of HTML, the page names no URL
    assert "http" not in html.split("<main>")[1]


def _pdf_text(path: Path) -> str:
    """Text of a PDF, its lines joined by spaces."""
    text = " ".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(text.split())


def test_the_pdf_is_the_report(report_dir: Path) -> None:
    """The PDF has the methods, the results and the versioned references."""
    text = _pdf_text(report_dir / "report.pdf")
    for token in ["Methods", "Results", "Bibliography", "Bland"]:
        assert token in text, token
    # the version of the provenance of the fit, here the installed sbmlsim
    assert f"Version {__version__}." in text


def test_latex_includes_the_png_figures(report_dir: Path) -> None:
    """LaTeX includes a PNG of every figure, at 300 dpi."""
    plots_dir = report_dir / "plots"
    for key in DOCUMENT_FIGURES:
        assert (plots_dir / f"{key}.svg").exists(), key
        with Image.open(plots_dir / f"{key}.png") as image:
            assert image.format == "PNG"
            assert image.info["dpi"] == pytest.approx((300, 300), abs=0.1)
    # the figures of the mappings and the other figures are not copied
    pngs = {path.stem for path in plots_dir.glob("*.png")}
    assert pngs == set(DOCUMENT_FIGURES)

    tex = (report_dir / "report.tex").read_text(encoding="utf-8")
    included = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", tex)
    assert len(included) == len(DOCUMENT_FIGURES)
    for path in included:
        assert path.endswith(".png") and (report_dir / path).exists(), path
    assert "\\bibliography{references}" in tex
    standalone = (report_dir / "report_standalone.tex").read_text(encoding="utf-8")
    assert standalone.startswith("\\documentclass")
    assert tex.strip() in standalone
    bib = (report_dir / "references.bib").read_text(encoding="utf-8")
    assert f"note = {{Version {__version__}}}" in bib


def _non_ascii(text: str) -> set[str]:
    """Characters of a LaTeX source which pdflatex cannot typeset.

    ASCII and the latin letters of T1 with utf8 input (umlauts of the names of
    authors) are typeset, everything else is not.
    """
    return {c for c in text if ord(c) > 127 and not (c.isalpha() and ord(c) < 0x180)}


def test_the_latex_has_no_unicode_symbols(report_dir: Path) -> None:
    """The LaTeX of a report has its symbols as commands, pdflatex reads it."""
    for name in ["report.tex", "report_standalone.tex"]:
        tex = (report_dir / name).read_text(encoding="utf-8")
        assert not _non_ascii(tex), (name, _non_ascii(tex))
    standalone = (report_dir / "report_standalone.tex").read_text(encoding="utf-8")
    assert "\\usepackage[T1]{fontenc}" in standalone
    assert "\\usepackage[utf8]{inputenc}" in standalone
    tex = (report_dir / "report.tex").read_text(encoding="utf-8")
    assert "\\chi" in tex and "\\alpha" in tex


@pytest.mark.skipif(
    shutil.which("pdflatex") is None or shutil.which("bibtex") is None,
    reason="pdflatex and bibtex are not installed",
)
def test_the_standalone_latex_compiles(report_dir: Path, tmp_path: Path) -> None:
    """The standalone LaTeX compiles with pdflatex and bibtex."""
    work = tmp_path / "latex"
    shutil.copytree(report_dir, work)

    def run(*command: str) -> None:
        done = subprocess.run(
            command, cwd=work, capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stdout[-3000:]

    run("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report_standalone")
    run("bibtex", "report_standalone")
    for _ in range(2):
        run(
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "report_standalone",
        )
    assert (work / "report_standalone.pdf").read_bytes().startswith(b"%PDF")


def test_a_png_report_has_no_copies(tmp_path: Path, fitted) -> None:
    """The figures of a PNG report are the figures LaTeX includes."""
    problem, result = fitted
    report = FitReport.from_optimization_result(
        problem=problem, opt_result=result, image_format="png", mapping_figures=False
    )
    out = report.create(output_dir=tmp_path)
    plots = sorted(path.name for path in (out / "plots").iterdir())
    assert "traces.png" in plots
    assert not [name for name in plots if name.endswith(".svg")]
    assert "plots/traces.png" in (out / "report.tex").read_text(encoding="utf-8")


def test_the_document_of_a_report(fitted_report) -> None:
    """The document of a report is its methods and its results."""
    report, plots_dir = fitted_report
    document = report.document(plots_dir)
    assert document == fit_document(report, ReportContext(), plots_dir)
    assert document.title == report.opt_result.sid
    assert [block.key for block in document.blocks] == ["methods", "results"]


def test_the_context_of_a_report(tmp_path: Path, fitted) -> None:
    """The context gives the report its title and the first sentence."""
    problem, result = fitted
    context = ReportContext(
        title="Pharmacokinetics of HCTZ",
        description="The absorption and the elimination of HCTZ.",
    )
    report = FitReport.from_optimization_result(
        problem=problem, opt_result=result, context=context, mapping_figures=False
    )
    assert report.context is context
    out = report.create(output_dir=tmp_path)
    html = _html_of(out)
    assert "Methods: Pharmacokinetics of HCTZ" in html
    assert "The absorption and the elimination of HCTZ." in html
    assert "Pharmacokinetics of HCTZ" in (out / "report.typ").read_text(
        encoding="utf-8"
    )


def _tabs(html: str) -> tuple[str, str]:
    """Get the panels of the tabs Report and Fit mappings of a report page."""
    report = html.index('<div id="report" class="tab-panel')
    mappings = html.index('<section id="mappings" class="tab-panel')
    end = html.index("</section>", html.rindex('class="card mapping"'))
    return html[report:mappings], html[mappings:end]


def test_the_report_tab_has_no_absolute_tolerances(report_dir: Path) -> None:
    """The overview has no table of the tolerances, the methods state them."""
    html = _html_of(report_dir)
    report, _ = _tabs(html)
    assert "Absolute tolerances" not in html
    assert "reference volume" not in html
    # the methods and the settings name the absolute tolerance of the fit
    assert "the absolute tolerance" in report
    assert "<td>absolute tolerance</td>" in report


def test_the_figures_and_metrics_of_the_mappings_are_in_their_tab(
    report_dir: Path,
) -> None:
    """The figures and the metrics of every mapping are in the mappings tab.

    The figures of the cost and weight and of the residuals of every mapping,
    then the table of their metrics, come before the cards of the mappings;
    the table filters with the chips and the search, and the report tab and
    its document have neither.
    """
    html = _html_of(report_dir)
    report, mappings = _tabs(html)
    assert "Contribution of the fit mappings" not in html
    for token in [
        "Cost, weight and residuals of the fit mappings",
        "plots/cost_bar.svg",
        "plots/residual_boxplot.svg",
        "Metrics of the fit mappings",
    ]:
        assert token not in report, token
        assert token in mappings, token
    cards = mappings.index('class="card mapping"')
    assert (
        mappings.index("plots/cost_bar.svg")
        < mappings.index("plots/residual_boxplot.svg")
        < mappings.index("Metrics of the fit mappings")
        < cards
    )
    # the rows of the metrics filter by kind and by the search
    metrics = mappings[mappings.index("Metrics of the fit mappings") : cards]
    assert '<table data-filter="row">' in metrics
    assert '<tr data-kind="training">' in metrics
    # the figures of the results stay in the report tab
    for name in ["goodness_of_fit", "bland_altman"]:
        assert f"plots/{name}.svg" in report, name
    # no document format refers to the moved figures, every reference resolves
    for name in ["report.typ", "report.tex", "report_standalone.tex"]:
        text = (report_dir / name).read_text(encoding="utf-8")
        assert "cost_bar" not in text and "residual_boxplot" not in text, name
    for label in re.findall(r'<a href="#([^"]+)">', html):
        assert f'id="{label}"' in html, label


def test_the_figures_of_the_mappings_grow_with_the_mappings(
    report_dir: Path, fitted
) -> None:
    """A row per fit mapping keeps its label readable, the first one on top."""
    problem, _ = fitted
    n = len(problem.mapping_keys)
    for name in ["cost_bar", "residual_boxplot"]:
        spec = json.loads(
            (report_dir / "plots" / f"{name}.json").read_text(encoding="utf-8")
        )
        # a row per mapping and room for the axes, in pixels
        assert spec["height"] >= 108.0 + ROW_HEIGHT * n - 0.5, name
        labels = [
            f"{e}|{m}"
            for e, m in zip(problem.experiment_keys, problem.mapping_keys, strict=True)
        ]
        assert spec["plots"][0]["y"]["categories"] == labels, name
        assert spec["plots"][0]["y"]["reverse"], name
    # a long id of a mapping breaks in the table, which stays in its card
    html = _html_of(report_dir)
    assert html.count('<td class="mono wrap">') == n


@pytest.mark.parametrize("plot", ["goodness_of_fit", "bland_altman"])
def test_the_panel_text_of_a_kind_is_the_text_of_its_points(
    fitted_report: tuple[FitReport, Path], plot: str
) -> None:
    """The box of a panel is computed from the points it draws."""
    report, _plots = fitted_report
    kind = MappingKind.TRAINING.value
    points = {
        pset.sid: report._of_kind(report.points(pset), kind)
        for pset in report.parameter_sets
    }
    assert report.panel_metrics(kind, plot) == report._panel_text(points, plot)


@pytest.fixture
def saved_figures(monkeypatch: pytest.MonkeyPatch) -> dict[str, Chart]:
    """The figures a report saves, by the name of their file, as charts."""
    saved: dict[str, Chart] = {}

    def save(self: FitReport, chart: Chart, path: Path) -> None:
        saved[path.stem] = chart

    monkeypatch.setattr(FitReport, "_save_chart", save)
    return saved


def _mapping_report(
    problem: OptimizationProblem, settings: FitSettings, n_sets: int = 1
) -> FitReport:
    problem.initialize(settings)
    model = problem.parameter_set_model()
    sets = [model]
    if n_sets > 1:
        sets.append(
            ParameterSet(
                sid="other",
                values={pid: 1.2 * value for pid, value in model.values.items()},
            )
        )
    return FitReport(
        problem=problem, settings=settings, parameter_sets=ParameterSets(sets)
    )


def test_the_figure_of_a_mapping_has_the_curve_and_the_agreement(
    tmp_path: Path,
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    saved_figures: dict[str, Chart],
) -> None:
    """The curve linear and logarithmic, the goodness of fit and the Bland-Altman plot."""
    report = _mapping_report(op_hctz_iv, fit_settings)
    report.plot_mapping(tmp_path)
    assert len(saved_figures) == len(op_hctz_iv.mapping_keys)
    k = 0
    name = f"{op_hctz_iv.experiment_keys[k]}_{op_hctz_iv.mapping_keys[k]}"
    chart = saved_figures[name]
    curve, curve_log, gof, altman = _plots(chart)
    assert curve["y"]["scale"] == "linear"
    assert curve_log["y"]["scale"] == "log"
    assert (gof["x"]["scale"], gof["y"]["scale"]) == ("log", "log")
    assert (altman["x"]["scale"], altman["y"]["scale"]) == ("log", "linear")
    box = "\n".join(_texts(gof))
    assert "GMFE" in box and "2-fold" in box and "cost" in box
    # the box is a white polygon behind its lines, inside the panel
    (polygon,) = _of_kind(gof, "polygon")
    assert polygon["style"]["fill"]["color"] == "#ffffff"
    low, high = gof["x"]["min"], gof["x"]["max"]
    assert all(low < x < high for x, _y in polygon["paths"][0])
    # the title has the kind, also of training data, and the observable
    assert chart.title is not None
    title = chart.title.splitlines()
    assert title[0].endswith(f"[{op_hctz_iv.mapping_kinds[k].value}]")
    assert title[1].startswith(report._y_label(k))
    # the legends are right of the rows: the curves and the agreement
    assert (chart.legend_scope, chart.legend_position) == ("row", "outside")
    assert curve["legend_order"][0] == "data"
    # the data is labelled in words, the prediction at the data are hollow
    # markers of the color of the set
    labels = _labels(curve)
    assert labels[0].startswith("data")
    hollow = [
        e
        for e in _of_kind(curve, "curve")
        if e["style"]["marker"].get("fill") == "none"
    ]
    assert hollow and all(e["legend"] == report.reference_set.sid for e in hollow)
    # the agreement has its identity and twofold
    assert "prediction = measurement" in _labels(gof)
    assert "2-fold" in _labels(altman)


def test_the_box_of_a_mapping_without_errors_has_no_uncertainty(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    report = _mapping_report(op_hctz_iv, fit_settings)
    for k in range(len(op_hctz_iv.mapping_keys)):
        if op_hctz_iv.y_errors[k] is None:
            assert "2 SD = -" in report.mapping_text(k)
        else:
            assert "2 SD = -" not in report.mapping_text(k)


def test_the_figure_of_a_mapping_compares_sets(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    report = _mapping_report(op_hctz_iv, fit_settings, n_sets=2)
    text = report.mapping_text(0).splitlines()
    sids = [pset.sid for pset in report.parameter_sets]
    assert [line[:-1] for line in text if line.endswith(":")] == sids


def test_the_data_label_of_a_mapping_has_its_error_and_count(
    op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    report = _mapping_report(op_hctz_iv, fit_settings)
    for k in range(len(op_hctz_iv.mapping_keys)):
        label = report._data_label(k)
        assert label.startswith("data")
        assert ("mean ± SD" in label) == (op_hctz_iv.y_errors[k] is not None)
        assert ("N = " in label) == (op_hctz_iv.y_counts[k] is not None)


def test_a_report_without_groupings_is_unchanged(
    fitted_report: tuple[FitReport, Path],
) -> None:
    """No figure, no table and no section of an evaluation without groupings."""
    report, plots = fitted_report
    assert report.groupings == ()
    assert not list(plots.glob("goodness_of_fit_*"))
    assert not list(plots.glob("criteria_*"))
    results_dir = plots.parent
    assert not (results_dir / "metrics_groups.tsv").exists()
    html = (results_dir / "index.html").read_text(encoding="utf-8")
    assert 'id="evaluation"' not in html


def test_a_report_of_a_grouping(
    tmp_path: Path, op_hctz_iv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The figures of a grouping, its metrics and its section."""
    op_hctz_iv.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_iv,
        settings=fit_settings,
        parameter_sets=op_hctz_iv.parameter_set_model(),
        mapping_figures=False,
        groupings=[Grouping.by("tissue")],
    )
    results_dir = report.create(output_dir=tmp_path, name="grouping")
    plots = results_dir / "plots"
    for name in ("goodness_of_fit_tissue", "bland_altman_tissue", "criteria_tissue"):
        assert (plots / f"{name}.svg").exists(), name
    table = pd.read_csv(results_dir / "metrics_groups.tsv", sep="\t")
    assert list(table.columns[:4]) == ["parameter_set", "grouping", "group", "kind"]
    assert set(table.grouping) == {"tissue"}
    html = (results_dir / "index.html").read_text(encoding="utf-8")
    assert 'id="evaluation"' in html
    assert "goodness_of_fit_tissue" in html


def test_the_panels_of_a_grouping(
    tmp_path: Path,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    saved_figures: dict[str, Chart],
) -> None:
    """A row per group and a column per kind, an empty panel says so."""
    op_hctz_pk.initialize(fit_settings)
    grouping = Grouping.by("tissue")
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
        groupings=[grouping],
    )
    report.plot_grouping(grouping, tmp_path)
    points = report.points(report.reference_set)
    groups = grouping.groups(points, report._fit_mapping)
    kinds = report.point_kinds()
    plots = _plots(saved_figures["goodness_of_fit_tissue"])
    assert len(plots) == len(groups) * len(kinds)
    for (label, of_group), row in zip(
        groups, range(0, len(plots), len(kinds)), strict=True
    ):
        for column, kind in enumerate(kinds):
            plot = plots[row + column]
            assert (plot["row"], plot["column"]) == (row // len(kinds), column)
            assert plot["title"].startswith(f"{label}: {kind}")
            if of_group[of_group.kind == kind].empty:
                assert _texts(plot) == ["no data"]
    assert len(_plots(saved_figures["bland_altman_tissue"])) == len(plots)


def test_the_bland_altman_plot_draws_twofold(
    tmp_path: Path,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    saved_figures: dict[str, Chart],
) -> None:
    """The lines of twofold, as in the goodness of fit, and the agreement of a panel.

    The validation panel has the agreement of its points, the training panel
    the band of the training data alone.
    """
    report = _mapping_report(op_hctz_pk, fit_settings)
    report.plot_bland_altman(tmp_path / "bland_altman.svg")
    plots = _plots(saved_figures["bland_altman"])
    kinds = report.point_kinds()
    for plot, kind in zip(plots, kinds, strict=True):
        levels = {round(e["value"], 6) for e in _of_kind(plot, "rule")}
        assert round(np.log10(2.0), 6) in levels
        assert round(-np.log10(2.0), 6) in levels
        labels = _labels(plot)
        points = report._of_kind(report.points(report.reference_set), kind)
        n_points = int(np.sum(report._log_ratio(points)[2]))
        own = kind == MappingKind.VALIDATION.value
        own = own and n_points >= MIN_POINTS_SUBSET_AGREEMENT
        assert ("this panel bias" in labels) is own, kind
        assert ("this panel LoA" in labels) is own, kind
    assert MappingKind.VALIDATION.value in kinds


def test_error_bars(
    tmp_path: Path,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    saved_figures: dict[str, Chart],
) -> None:
    """With `error_bars` the points of the data with an SD have error bars."""
    op_hctz_pk.initialize(fit_settings)
    assert any(errors is not None for errors in op_hctz_pk.y_errors)
    for error_bars in (False, True):
        report = FitReport(
            problem=op_hctz_pk,
            settings=fit_settings,
            parameter_sets=op_hctz_pk.parameter_set_model(),
            mapping_figures=False,
            error_bars=error_bars,
        )
        saved_figures.clear()
        report.plot_goodness_of_fit(tmp_path / "goodness_of_fit.svg")
        report.plot_bland_altman(tmp_path / "bland_altman.svg")
        for name in ("goodness_of_fit", "bland_altman"):
            plot = _plots(saved_figures[name])[0]
            assert bool(_error_bars(plot)) is error_bars, (name, error_bars)


def test_color_by_a_column(
    tmp_path: Path,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    saved_figures: dict[str, Chart],
) -> None:
    """`color_by` gives the points of one value of a column one colour."""
    op_hctz_pk.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
        color_by="tissue",
    )
    report.plot_goodness_of_fit(tmp_path / "goodness_of_fit.svg")
    plot = _plots(saved_figures["goodness_of_fit"])[0]
    tissues = set(report.points(report.reference_set).tissue)
    labelled = {
        e["label"]: e["style"]["color"]
        for e in _of_kind(plot, "curve")
        if e.get("label") in tissues
    }
    assert labelled
    assert len(set(labelled.values())) == len(labelled)


def test_the_legend_of_a_grouping_has_the_band(
    tmp_path: Path,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    saved_figures: dict[str, Chart],
) -> None:
    """The band is in the legend also when the last panel has no points."""
    op_hctz_pk.initialize(fit_settings)
    grouping = Grouping.of_subsets(
        "subsets",
        Subset("all", _select_all),
        Subset("none", _select_none),
    )
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    report.plot_grouping(grouping, tmp_path)
    for name in ("goodness_of_fit_subsets", "bland_altman_subsets"):
        labels = [
            label for plot in _plots(saved_figures[name]) for label in _labels(plot)
        ]
        assert "prediction = measurement" in labels, name
        assert "2-fold" in labels, name
        assert any(label.startswith("LoA") for label in labels), name


def _select_all(key: str, mapping: Any) -> bool:
    return True


def _select_none(key: str, mapping: Any) -> bool:
    return False


def test_the_box_of_a_panel_has_the_criteria(
    fitted_report: tuple[FitReport, Path],
) -> None:
    """The box of the goodness of fit states GMFE, AFE and the points within 2 SD."""
    report, _plots = fitted_report
    box = report.panel_metrics(MappingKind.TRAINING.value, "goodness_of_fit")
    assert "GMFE = " in box and "AFE = " in box and "2 SD = " in box


def test_error_bars_only_for_points_with_a_ratio(
    tmp_path: Path,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    saved_figures: dict[str, Chart],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A point which is not drawn on the logarithmic axes has no error bar."""
    op_hctz_pk.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
        error_bars=True,
    )
    # a standard deviation of the points which are zero, e.g. at the start of
    # an amount in the urine
    points = FitReport.points

    def with_sd(self: FitReport, pset: ParameterSet) -> pd.DataFrame:
        df = points(self, pset)
        df.loc[df.DV <= 0, "DV_SD"] = 1e-3
        return df

    monkeypatch.setattr(FitReport, "points", with_sd)
    assert (report.points(report.reference_set).DV <= 0).any()
    report.plot_goodness_of_fit(tmp_path / "goodness_of_fit.svg")
    bars = [
        bar
        for plot in _plots(saved_figures["goodness_of_fit"])
        for bar in _error_bars(plot)
    ]
    assert bars
    for bar in bars:
        assert all(x > 0 for x in bar["x"] if x is not None)
        assert all(y > 0 for y in bar["y"] if y is not None)


def test_long_set_ids_stay_in_their_box(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """With several sets the id of a set is a line of its own in a box."""
    op_hctz_pk.initialize(fit_settings)
    model = op_hctz_pk.parameter_set_model()
    sets = [
        ParameterSet(sid="PK_20261022_232242__64ca9_0", values=dict(model.values)),
        ParameterSet(
            sid="PK_20261022_232242__64ca9_1",
            values={pid: 1.1 * value for pid, value in model.values.items()},
        ),
    ]
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=ParameterSets(sets),
        mapping_figures=False,
    )
    for plot in ("goodness_of_fit", "bland_altman"):
        lines = report.panel_metrics(MappingKind.TRAINING.value, plot).splitlines()
        ids = [line for line in lines if line.endswith(":")]
        assert len(ids) == 2
        # not wider than the one line of a set of before, which fit its panel
        assert max(len(line) for line in lines) <= 56, lines
    text = report.mapping_text(0).splitlines()
    assert sum(line.endswith(":") for line in text) == 2


def test_the_curve_of_a_mapping_draws_only_the_errors_of_the_data(
    tmp_path: Path,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    saved_figures: dict[str, Chart],
) -> None:
    """A filled error of the weights has no error bar in the curve."""
    report = _mapping_report(op_hctz_pk, fit_settings)
    filled = [
        k
        for k, measured in enumerate(op_hctz_pk.y_errors_measured)
        if measured is not None and not measured.all()
    ]
    assert filled
    k = filled[0]
    report.plot_mapping(tmp_path)
    name = f"{op_hctz_pk.experiment_keys[k]}_{op_hctz_pk.mapping_keys[k]}"
    curve = _plots(saved_figures[name])[0]
    bars = [bar for bar in _error_bars(curve) if bar["legend"] == "data"]
    measured = op_hctz_pk.y_errors_measured[k]
    assert measured is not None
    # a point whose error the weights filled has no bar
    assert sum(_n_bars(bar) for bar in bars) == int(measured.sum())


def test_an_error_to_zero_reaches_the_bottom_of_the_log_panel(
    tmp_path: Path,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    saved_figures: dict[str, Chart],
) -> None:
    """A standard deviation as large as its mean keeps its bar in the log panel.

    The lower end of the bar has no logarithm, so the bar goes to the bottom
    of the axis and keeps its upper half, as in the linear panel; e.g. the
    data of Patel1984 `fm_25_tab`.
    """
    report = _mapping_report(op_hctz_pk, fit_settings)
    wide = []
    for k, errors in enumerate(op_hctz_pk.y_errors):
        measured = op_hctz_pk.y_errors_measured[k]
        if errors is None or measured is None:
            continue
        y = np.asarray(op_hctz_pk.y_references[k], dtype=float)
        if np.any(measured & (y > 0) & (np.asarray(errors, dtype=float) >= y)):
            wide.append(k)
    assert wide
    k = wide[0]
    report.plot_mapping(tmp_path)
    name = f"{op_hctz_pk.experiment_keys[k]}_{op_hctz_pk.mapping_keys[k]}"
    linear, log = _plots(saved_figures[name])[:2]
    assert log["y"]["scale"] == "log"
    bars = {
        key: [bar for bar in _error_bars(plot) if bar["legend"] == "data"]
        for key, plot in (("linear", linear), ("log", log))
    }
    y = np.asarray(op_hctz_pk.y_references[k], dtype=float)
    measured = op_hctz_pk.y_errors_measured[k]
    assert measured is not None
    assert sum(_n_bars(bar) for bar in bars["log"]) == int(np.sum(measured & (y > 0)))
    lows = [
        value
        for bar in bars["log"]
        if "yerr" not in bar
        for value in bar["y"][0::3]
        if value is not None
    ]
    # the bars of the points whose error reaches zero end at the bottom
    clipped = int(
        np.sum(measured & (y > 0) & (np.asarray(op_hctz_pk.y_errors[k]) >= y))
    )
    bottom = log["y"]["min"]
    assert sum(low == pytest.approx(bottom) for low in lows) == clipped


def test_a_box_of_metrics_stays_in_its_panel() -> None:
    """A box of a long text is clamped to its panel, a short one fits its text."""
    pytest.importorskip("sbmlsim_core")
    chart = Chart(400, 300, columns=2)
    plots = [
        chart.plot(0, column, x=ChartAxis(min=0.0, max=1.0), y=ChartAxis(min=0, max=1))
        for column in range(2)
    ]
    FitReport._metrics_boxes(chart, [(plots[0], "x" * 400), (plots[1], "n = 4")])
    long, short = (_of_kind(plot, "polygon")[0] for plot in _plots(chart))
    xs = [x for x, _y in long["paths"][0]]
    assert min(xs) > 0.0 and max(xs) < 1.0
    assert max(x for x, _y in short["paths"][0]) < 0.5
    assert [_texts(plot) for plot in _plots(chart)] == [["x" * 400], ["n = 4"]]


def test_a_box_of_several_lines_is_one_text() -> None:
    """The lines of a box are one text of several lines, the box their height."""
    pytest.importorskip("sbmlsim_core")
    chart = Chart(400, 300)
    plot = chart.plot(x=ChartAxis(min=0.0, max=1.0), y=ChartAxis(min=0, max=1))
    FitReport._metrics_boxes(chart, [(plot, "GMFE = 1.2\nn = 4\n2 SD = 80 %")])
    (spec,) = _plots(chart)
    assert _texts(spec) == ["GMFE = 1.2\nn = 4\n2 SD = 80 %"]
    (box,) = _of_kind(spec, "polygon")
    ys = [y for _x, y in box["paths"][0]]
    (text,) = _of_kind(spec, "text")
    # the text is centred in its box
    assert text["y"] == pytest.approx((min(ys) + max(ys)) / 2)


def _initialized_report(
    problem: OptimizationProblem, settings: FitSettings, **kwargs: Any
) -> FitReport:
    problem.initialize(settings)
    return FitReport(
        problem=problem,
        settings=settings,
        parameter_sets=problem.parameter_set_model(),
        mapping_figures=False,
        **kwargs,
    )


def test_the_columns_of_a_report_are_checked(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A typo in a column fails when the report is created, not when it is written."""
    with pytest.raises(ValueError, match=r"color_by 'tisue'.*'tissue'"):
        _initialized_report(op_hctz_pk, fit_settings, color_by="tisue")
    with pytest.raises(ValueError, match=r"'tisue'.*'tissue'"):
        _initialized_report(op_hctz_pk, fit_settings, groupings=[Grouping.by("tisue")])
    with pytest.raises(ValueError, match=r"tissue.*twice"):
        _initialized_report(
            op_hctz_pk,
            fit_settings,
            groupings=[Grouping.by("tissue"), Grouping.by("tissue")],
        )


def test_the_metrics_of_no_grouping(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    report = _initialized_report(op_hctz_pk, fit_settings)
    metrics = report.metrics_groups_df()
    assert metrics.empty
    assert list(metrics.columns[:4]) == ["parameter_set", "grouping", "group", "kind"]


def test_the_criteria_of_an_empty_subset(
    tmp_path: Path,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    saved_figures: dict[str, Chart],
) -> None:
    report = _initialized_report(op_hctz_pk, fit_settings)
    grouping = Grouping.of_subsets(
        "subsets", Subset("all", _select_all), Subset("none", _select_none)
    )
    report.plot_grouping(grouping, tmp_path)
    plot = _plots(saved_figures["criteria_subsets"])[0]
    assert plot["y"]["categories"] == ["all", "none"]
    assert "no data" in _texts(plot)


def test_the_figures_of_many_groups_are_pages(
    tmp_path: Path,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    saved_figures: dict[str, Chart],
) -> None:
    """A figure has at most `MAX_GROUP_ROWS` rows, more groups are more pages."""
    report = _initialized_report(op_hctz_pk, fit_settings)
    grouping = Grouping.by("mapping")
    n_groups = len(grouping.groups(report.points(report.reference_set), _no_lookup))
    assert n_groups > MAX_GROUP_ROWS
    report.plot_grouping(grouping, tmp_path)
    pages = -(-n_groups // MAX_GROUP_ROWS)
    names = ["goodness_of_fit_mapping"] + [
        f"goodness_of_fit_mapping_{page}" for page in range(2, pages + 1)
    ]
    assert [name for name in saved_figures if name.startswith("goodness")] == names
    n_kinds = len(report.point_kinds())
    rows = [len(_plots(saved_figures[name])) // n_kinds for name in names]
    assert sum(rows) == n_groups
    assert max(rows) <= MAX_GROUP_ROWS


def _no_lookup(experiment: str, mapping: str) -> Any:
    return None


def test_the_evaluation_section_has_a_tab_per_grouping(
    tmp_path: Path, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    report = _initialized_report(
        op_hctz_pk,
        fit_settings,
        groupings=[Grouping.by("tissue"), Grouping.by("route")],
    )
    results_dir = report.create(output_dir=tmp_path, name="tabs")
    html = (results_dir / "index.html").read_text(encoding="utf-8")
    assert html.count('data-grouping="tissue"') == 2
    assert html.count('data-grouping="route"') == 2
    assert "2-fold [%]" in html.split('id="evaluation"')[1]


def test_a_panel_of_one_point_has_no_nan(
    fitted_report: tuple[FitReport, Path],
) -> None:
    report, _plots = fitted_report
    points = report.points(report.reference_set).iloc[[1]]
    for plot in ("goodness_of_fit", "bland_altman"):
        text = report._panel_text({report.reference_set.sid: points}, plot)
        assert "nan" not in text, text


# ----------------------------------------------------------------------------
# the legend and the colors of the summary figures with many studies (#511)
# ----------------------------------------------------------------------------
def _legend_figure(report: FitReport, labels: list[str]) -> Chart:
    """Get a figure of the panels of a report whose points carry one label each."""
    chart, kinds = report._panels("legend")
    for column in range(len(kinds)):
        plot = chart.plot(0, column, legend_order=labels)
        for index, label in enumerate(labels):
            plot.curve(
                [1.0],
                [1.0],
                line="none",
                marker="circle",
                color=normalize_color(category_color(index, len(labels))),
                label=label,
                legend=label,
            )
        plot.curve([0.0, 1.0], [0.0, 1.0], label="identity", legend="identity")
    return chart


def test_the_colors_of_few_studies_are_the_categorical_ones() -> None:
    """A report of at most as many studies as colors keeps the colors it had."""
    assert [category_color(i, 5) for i in range(5)] == list(STUDY_COLORS[:5])
    assert category_color(0, len(STUDY_COLORS)) == STUDY_COLORS[0]


@pytest.mark.parametrize("total", [len(STUDY_COLORS) + 1, 20, 35, 64])
def test_every_study_has_a_color_of_its_own(total: int) -> None:
    """The colors repeated every nine studies, which told them apart no more (#511)."""
    colors = [category_color(index, total) for index in range(total)]

    assert len(set(colors)) == total


def test_the_figure_legend_shows_every_entry(fitted_report: Any) -> None:
    """A legend of many studies shows every entry instead of being cut (#511).

    The legend of a fit of 35 studies ran off the bottom of the figure: it
    showed 29 of them and cut the rest and the entries of the lines with them.
    The legend of a row is right of its panels, and the renderer grows the
    figure for it.
    """
    pytest.importorskip("sbmlsim_core")
    from sbmlsim.engine import import_core

    labels = [f"Study{index:02d}" for index in range(35)]
    report, _plots_dir = fitted_report
    chart = _legend_figure(report, labels)

    geometry = import_core().figure_geometry(chart.to_json())
    entries = geometry["legend_entries"]
    assert [entry["legend"] for entry in entries] == [*labels, "identity"]
    for entry in entries:
        rect = entry["rect"]
        assert rect["y"] >= 0.0
        assert rect["y"] + rect["height"] <= geometry["height"] + 0.5
        assert rect["x"] + rect["width"] <= geometry["width"] + 0.5
    # the panels keep their room, right of them is the legend
    panels = [plot["rect"] for plot in geometry["plots"]]
    assert max(p["x"] + p["width"] for p in panels) <= min(
        entry["rect"]["x"] for entry in entries
    )


def test_the_figures_are_saved_with_their_spec(
    tmp_path: Path, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A figure of the report document is saved with its spec and as a PNG."""
    report = _initialized_report(op_hctz_pk, fit_settings)
    chart = report.plot_goodness_of_fit(tmp_path / "goodness_of_fit.svg")
    assert isinstance(chart, Chart)
    assert (
        (tmp_path / "goodness_of_fit.svg")
        .read_text(encoding="utf-8")
        .startswith("<svg")
    )
    spec = json.loads((tmp_path / "goodness_of_fit.json").read_text(encoding="utf-8"))
    assert spec == json.loads(chart.to_json())
    with Image.open(tmp_path / "goodness_of_fit.png") as image:
        assert image.info["dpi"] == pytest.approx((300, 300), abs=0.1)
    # a figure of the mappings tab has no PNG
    report.plot_cost_bar(tmp_path / "cost_bar.svg")
    assert not (tmp_path / "cost_bar.png").exists()
    with pytest.raises(ValueError, match="svg or png"):
        report.plot_cost_bar(tmp_path / "cost_bar.pdf")
    # a report of another format is refused when it is created
    with pytest.raises(ValueError, match="svg or png, not 'pdf'"):
        _initialized_report(op_hctz_pk, fit_settings, image_format="pdf")


def test_the_study_colors_and_set_markers_are_of_the_renderer(
    tmp_path: Path,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    saved_figures: dict[str, Chart],
) -> None:
    """A study has its color and a set its marker in every agreement figure."""
    report = _mapping_report(op_hctz_pk, fit_settings, n_sets=2)
    report.plot_goodness_of_fit(tmp_path / "goodness_of_fit.svg")
    report.plot_bland_altman(tmp_path / "bland_altman.svg")
    for name in ("goodness_of_fit", "bland_altman"):
        for plot in _plots(saved_figures[name]):
            for element in _of_kind(plot, "curve"):
                label = element.get("label", "")
                for k, pset in enumerate(report.parameter_sets):
                    if not label.endswith(f" ({pset.sid})"):
                        continue
                    study = label.removesuffix(f" ({pset.sid})")
                    style = element["style"]
                    assert style["color"] == normalize_color(
                        STUDY_COLORS[report.studies().index(study)]
                    )
                    assert style["marker"]["type"] == ("circle", "square")[k]


def test_fit_report_imports_no_matplotlib(tmp_path: Path) -> None:
    """Building a report with every figure imports no matplotlib.

    The report of a fit of two runs with the profiles of its parameters and
    the figures of the mappings: traces, waterfall, the agreement, the costs,
    the mappings and the profiles.
    """
    script = f"""
import sys
from pathlib import Path

from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS
from sbmlsim.fit import FitSettings
from sbmlsim.fit.identifiability import ProfileSettings, profile_likelihood
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.runner import run_optimization

problem = FIT_DEFINITIONS["IV"].problem(opid="iv")
settings = FitSettings(absolute_tolerance=1e-6, relative_tolerance=1e-6)
result = run_optimization(
    problem=problem, settings=settings, size=2, n_cores=1, serial=True,
    seed=1234, max_nfev=2,
)
problem.initialize(settings)
identifiability = profile_likelihood(
    problem=problem,
    settings=settings,
    parameter_set=result.parameter_set(),
    profile_settings=ProfileSettings(
        reoptimize=False, initial_step=0.5, min_step=0.1, max_step=2.0, max_points=4
    ),
    serial=True,
    show_progress=False,
)
report = FitReport.from_optimization_result(
    problem=problem,
    opt_result=result,
    identifiability=identifiability,
    mapping_figures=True,
)
out = report.create(output_dir=Path({str(tmp_path)!r}), name="report")
plots = out / "plots"
for name in ("traces", "waterfall", "goodness_of_fit", "cost_bar", "profiles"):
    assert (plots / f"{{name}}.svg").exists(), name
assert list(plots.glob("profile_*.svg"))
assert len(list(plots.glob("*_fm_*.svg"))) == len(problem.mapping_keys)
print("matplotlib" in sys.modules)
"""
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip().splitlines()[-1] == "False", done.stdout[-2000:]
