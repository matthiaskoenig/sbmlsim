"""Test the report of a fit.

The report is separate from the optimization: it is created from the definition
of the problem, the settings and one or more parameter sets.
"""

import dataclasses
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from PIL import Image
from pypdf import PdfReader

from examples.hctz_fitting.experiments.studies import Weir1998
from sbmlsim import __version__
from sbmlsim.fit import FitSettings, MappingKind, ParameterSet, ParameterSets
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.evaluation import Grouping
from sbmlsim.fit.fisher import fisher_information
from sbmlsim.fit.methods import ReportContext
from sbmlsim.fit.metrics import fold_counts
from sbmlsim.fit.objects import EVALUATED_KINDS, FitMappingCollection, FitParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.report_document import fit_document
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization


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
    html = (report.create(tmp_path, name="versioned") / "index.html").read_text()
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
    html = (report.create(tmp_path, name="ordinary") / "index.html").read_text()
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
    html = (report.create(tmp_path, name="paths") / "index.html").read_text()
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


def test_the_legend_lists_the_studies_before_the_lines() -> None:
    """The legend of the panels lists the points first and the lines last.

    The band and the lines of twofold are labelled on the last panel, and a
    study whose points are only in that panel, e.g. a study with validation
    data alone, is still listed with the other studies and not after them.
    """
    from matplotlib.figure import Figure

    fig = Figure()
    first, last = fig.subplots(nrows=1, ncols=2)
    first.plot([1.0], [1.0], marker="o", linestyle="none", label="StudyA")
    last.plot([1.0, 2.0], [1.0, 2.0], linestyle="-.", label="bias 1.00x")
    last.plot([1.0, 2.0], [2.0, 4.0], linestyle=":", label="2-fold")
    last.plot([1.0], [1.0], marker="o", linestyle="none", label="StudyB")
    FitReport._set_figure_legend(fig, [first, last])
    (legend,) = fig.legends
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["StudyA", "StudyB", "bias 1.00x", "2-fold"]


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
    """With several sets the box has a line per set, prefixed with its id."""
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
    assert len(lines) == 2
    assert lines[0].startswith(f"{model.sid}: ")
    assert lines[1].startswith("other: ")


def test_the_figures_of_every_mapping(
    tmp_path: Path,
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    no_pyplot: None,
) -> None:
    """A report draws the figure of every fit mapping by default.

    They are almost the whole cost of a report, so the tests which only read
    its tables create it with `mapping_figures=False`; this is the test that
    the default still draws them and that the cards carry them. No figure is
    drawn with pyplot, whose backend may be interactive (#317).
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
    context = report.html_context(results_dir=Path("nowhere"), name="the_fit")

    assert context["fit_id"] == "the_fit"
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
    html = (tmp_path / "hints" / "index.html").read_text()

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

    html = (results_dir / "index.html").read_text()
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
    html = (report_dir / "index.html").read_text()
    assert 'data-tab="report"' in html and 'data-tab="mappings"' in html
    assert "<math" in html


def test_pdf_failure_keeps_the_fit(monkeypatch, tmp_path, fitted) -> None:
    import sbmlsim.fit.report as report_module

    def broken(*args, **kwargs):
        raise RuntimeError("typst could not compile 'report.typ': boom")

    monkeypatch.setattr(report_module, "compile_pdf", broken)
    problem, result = fitted
    out = FitReport.from_optimization_result(problem=problem, opt_result=result).create(
        output_dir=tmp_path
    )
    assert (out / "report.typ").exists() and not (out / "report.pdf").exists()
    assert "PDF could not be created" in (out / "index.html").read_text()


def test_a_failed_pdf_is_stated_with_its_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fitted_report
) -> None:
    """The report states why the PDF failed, every other output is written."""
    import sbmlsim.fit.report as report_module

    def broken(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("typst could not compile 'report.typ': boom")

    monkeypatch.setattr(report_module, "compile_pdf", broken)
    report, _ = fitted_report
    out = report.create(output_dir=tmp_path, name="failed")
    html = _html_of(out)
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

    tex = (report_dir / "report.tex").read_text()
    included = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", tex)
    assert len(included) == len(DOCUMENT_FIGURES)
    for path in included:
        assert path.endswith(".png") and (report_dir / path).exists(), path
    assert "\\bibliography{references}" in tex
    standalone = (report_dir / "report_standalone.tex").read_text()
    assert standalone.startswith("\\documentclass")
    assert tex.strip() in standalone
    bib = (report_dir / "references.bib").read_text()
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
        tex = (report_dir / name).read_text()
        assert not _non_ascii(tex), (name, _non_ascii(tex))
    standalone = (report_dir / "report_standalone.tex").read_text()
    assert "\\usepackage[T1]{fontenc}" in standalone
    assert "\\usepackage[utf8]{inputenc}" in standalone
    tex = (report_dir / "report.tex").read_text()
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
    assert "plots/traces.png" in (out / "report.tex").read_text()


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
    assert "Pharmacokinetics of HCTZ" in (out / "report.typ").read_text()


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
        text = (report_dir / name).read_text()
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
        svg = (report_dir / "plots" / f"{name}.svg").read_text()
        match = re.search(r'height="([0-9.]+)pt"', svg)
        assert match is not None, name
        height = float(match.group(1))
        # 0.17 inches per mapping and 1.5 inches for the axes, in points
        assert height >= (1.5 + 0.17 * n) * 72 - 0.5, name
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
def saved_figures(monkeypatch: pytest.MonkeyPatch) -> dict[str, Figure]:
    """The figures a report saves, by the name of their file."""
    saved: dict[str, Figure] = {}

    def save(self: FitReport, fig: Figure, path: Path) -> None:
        saved[path.stem] = fig

    monkeypatch.setattr(FitReport, "_save_mpl_figure", save)
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
    saved_figures: dict[str, Figure],
) -> None:
    """The curve linear and logarithmic, the goodness of fit and the Bland-Altman plot."""
    report = _mapping_report(op_hctz_iv, fit_settings)
    report.plot_mapping(tmp_path)
    assert len(saved_figures) == len(op_hctz_iv.mapping_keys)
    k = 0
    name = f"{op_hctz_iv.experiment_keys[k]}_{op_hctz_iv.mapping_keys[k]}"
    fig = saved_figures[name]
    curve, curve_log, gof, altman = fig.axes[:4]
    assert curve.get_yscale() == "linear"
    assert curve_log.get_yscale() == "log"
    assert (gof.get_xscale(), gof.get_yscale()) == ("log", "log")
    assert (altman.get_xscale(), altman.get_yscale()) == ("log", "linear")
    box = "\n".join(text.get_text() for text in gof.texts)
    assert "GMFE" in box and "2-fold" in box and "cost" in box
    # the title has the kind, also of training data, and the observable
    title = fig.get_suptitle().splitlines()
    assert title[0].endswith(f"[{op_hctz_iv.mapping_kinds[k].value}]")
    assert title[1].startswith(report._y_label(k))
    # the data is labelled in words, the prediction at the data are open markers
    labels = [str(line.get_label()) for line in curve.get_lines()]
    assert not any(label.startswith("reference_data") for label in labels)
    open_markers = [
        line for line in curve.get_lines() if line.get_markerfacecolor() == "none"
    ]
    assert open_markers


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
    assert [line.split(":")[0] for line in text[::2]] == sids


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
    saved_figures: dict[str, Figure],
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
    fig = saved_figures["goodness_of_fit_tissue"]
    assert len(fig.axes) == len(groups) * len(kinds)
    for (label, of_group), row in zip(
        groups, range(0, len(fig.axes), len(kinds)), strict=True
    ):
        for column, kind in enumerate(kinds):
            ax = fig.axes[row + column]
            assert ax.get_title().startswith(f"{label}: {kind}")
            if of_group[of_group.kind == kind].empty:
                assert any(text.get_text() == "no data" for text in ax.texts)
    assert len(saved_figures["bland_altman_tissue"].axes) == len(fig.axes)
