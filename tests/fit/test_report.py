"""Test the report of a fit.

The report is separate from the optimization: it is created from the definition
of the problem, the settings and one or more parameter sets.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest

from sbmlsim.fit import FitSettings, MappingKind, ParameterSet, ParameterSets
from sbmlsim.fit.fisher import fisher_information
from sbmlsim.fit.objects import EVALUATED_KINDS
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization


def _fit(
    problem: OptimizationProblem, settings: FitSettings, size: int = 1
) -> OptimizationResult:
    """Run a small fit of the problem."""
    return run_optimization(
        problem=problem,
        settings=settings,
        size=size,
        n_cores=1,
        serial=True,
        seed=1234,
    )


def _assert_report_files(results_dir: Path) -> None:
    """Check the files every report writes."""
    assert (results_dir / "index.html").exists()
    assert (results_dir / "report.txt").exists()
    assert (results_dir / "parameters.json").exists()
    assert list((results_dir / "plots").glob("*.svg"))


def test_report_from_optimization_result(
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The report of a fit shows the fitted parameters alone."""
    opt_result = _fit(op_hctz_pkiv, fit_settings)
    report = FitReport.from_optimization_result(
        problem=op_hctz_pkiv, opt_result=opt_result
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
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """`with_model` compares the fit against the values the model started from."""
    opt_result = _fit(op_hctz_pkiv, fit_settings)
    report = FitReport.from_optimization_result(
        problem=op_hctz_pkiv, opt_result=opt_result, with_model=True
    )

    # the model values are the reference, the fit is the second set
    assert len(report.parameter_sets) == 2
    assert report.reference_set.sid == "model"
    # no parameter set is black, which is the color of the reference data
    assert "black" not in {report.color(pset) for pset in report.parameter_sets}

    results_dir = report.create(output_dir=tmp_path, name="fit_with_model")
    _assert_report_files(results_dir)
    assert (results_dir / "plots" / "cost_scatter.svg").exists()


def test_the_subsets_of_the_data_points(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The panels are `all` and every kind the problem has, in that order."""
    op_hctz_pk.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
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
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A problem with a single kind has one panel, not `all` and the kind."""
    op_hctz_pkiv.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        parameter_sets=op_hctz_pkiv.parameter_set_model(),
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
    wide = {plot["src"]: plot["wide"] for plot in context["result_plots"]}
    assert wide["plots/goodness_of_fit.svg"]
    assert wide["plots/bland_altman.svg"]
    assert not wide["plots/cost_bar.svg"]
    assert 'class="card wide"' in html


def test_the_key_metrics_of_the_panels(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Every panel carries the key metrics of its points."""
    op_hctz_pk.initialize(fit_settings)
    pset = op_hctz_pk.parameter_set_model()
    report = FitReport(problem=op_hctz_pk, settings=fit_settings, parameter_sets=pset)

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
    )
    text = report.panel_metrics(MappingKind.TRAINING.value, "goodness_of_fit")
    lines = text.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith(f"{model.sid}: ")
    assert lines[1].startswith("other: ")


def test_report_without_optimization(
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A report is created from stored parameters, without running a fit."""
    opt_result = _fit(op_hctz_pkiv, fit_settings)
    parameters_path = tmp_path / "parameters.json"
    opt_result.parameter_sets(size=1).to_json(path=parameters_path)

    # a fresh problem, nothing was optimized on it
    problem = OptimizationProblem(
        opid=op_hctz_pkiv.opid,
        mapping_collections=op_hctz_pkiv.mapping_collections,
        fit_parameters=op_hctz_pkiv.parameters,
        base_path=op_hctz_pkiv.base_path,
        data_path=op_hctz_pkiv.data_path,
    )
    report = FitReport(
        problem=problem,
        settings=fit_settings,
        parameter_sets=ParameterSets.from_json(parameters_path),
    )

    results_dir = report.create(output_dir=tmp_path, name="report")
    _assert_report_files(results_dir)
    # no optimization, so no plots of the runs
    assert not (results_dir / "plots" / "traces.svg").exists()
    assert not (results_dir / "optimization_result.json").exists()


def test_report_multiple_parameter_sets(
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Several parameter sets are compared in a single report."""
    opt_result = _fit(op_hctz_pkiv, fit_settings, size=2)
    sets = ParameterSets(
        [
            op_hctz_pkiv.parameter_set_model(),
            *opt_result.parameter_sets(size=2),
        ]
    )
    report = FitReport(problem=op_hctz_pkiv, settings=fit_settings, parameter_sets=sets)
    assert len(report.parameter_sets) == 3

    results_dir = report.create(output_dir=tmp_path, name="compare")
    _assert_report_files(results_dir)
    # the costs of the sets are compared against the reference set
    assert (results_dir / "plots" / "cost_scatter.svg").exists()

    # every set is a column of the parameter table
    df = report.parameter_sets.to_df()
    assert len(df.columns) == 2 + 3


def test_report_single_set(
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A report of a single set has nothing to compare it against."""
    op_hctz_pkiv.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        parameter_sets=op_hctz_pkiv.parameter_set_model(),
    )
    results_dir = report.create(output_dir=tmp_path, name="model")
    _assert_report_files(results_dir)
    assert not (results_dir / "plots" / "cost_scatter.svg").exists()


def test_report_requires_a_set(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A report without a parameter set is an error."""
    with pytest.raises(ValueError, match="At least one"):
        FitReport(problem=op_hctz_pkiv, settings=fit_settings, parameter_sets=[])


def test_report_parameter_set_of_other_problem(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A set which does not cover the parameters of the problem is reported."""
    report = FitReport(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        parameter_sets=ParameterSet(sid="other", values={"unknown": 1.0}),
    )
    with pytest.raises(KeyError, match="does not contain"):
        report.residual_data(report.reference_set)


def test_run_plots_require_a_result(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The traces and the waterfall plot need an optimization result."""
    op_hctz_pkiv.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        parameter_sets=op_hctz_pkiv.parameter_set_model(),
    )
    with pytest.raises(ValueError, match="OptimizationResult"):
        _ = report.opt_result_required


def _html_of(results_dir: Path) -> str:
    """Read the HTML report."""
    return (results_dir / "index.html").read_text(encoding="utf-8")


def test_html_report_sections(
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The report has the three sections and its search."""
    opt_result = _fit(op_hctz_pkiv, fit_settings)
    report = FitReport.from_optimization_result(
        problem=op_hctz_pkiv, opt_result=opt_result
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
    assert html.count('class="card mapping"') == len(op_hctz_pkiv.mapping_keys)
    # the tables can be sorted
    assert "data-sort=" in html


def test_html_report_is_offline(
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The report needs no network, it is read from a file and archived."""
    opt_result = _fit(op_hctz_pkiv, fit_settings)
    report = FitReport.from_optimization_result(
        problem=op_hctz_pkiv, opt_result=opt_result
    )
    html = _html_of(report.create(output_dir=tmp_path, name="fit"))
    assert "http://" not in html
    assert "https://" not in html


def test_html_report_references_exist(
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Every file the report links to was written."""
    opt_result = _fit(op_hctz_pkiv, fit_settings)
    report = FitReport.from_optimization_result(
        problem=op_hctz_pkiv, opt_result=opt_result
    )
    results_dir = report.create(output_dir=tmp_path, name="fit")

    references = re.findall(r'(?:src|href)="([^"#:]+)"', _html_of(results_dir))
    assert references
    missing = [ref for ref in references if not (results_dir / ref).exists()]
    assert missing == []


def test_html_report_without_optimization(
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A report of stored parameters has no plots of the runs."""
    op_hctz_pkiv.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        parameter_sets=op_hctz_pkiv.parameter_set_model(),
    )
    html = _html_of(report.create(output_dir=tmp_path, name="model"))
    assert "traces" not in html
    assert "optimization_result.json" not in html
    # the sections which do not depend on a fit are there
    assert 'id="overview"' in html
    assert 'id="mappings"' in html


def test_html_context(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The context of the template carries the parts of the report."""
    op_hctz_pkiv.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        parameter_sets=op_hctz_pkiv.parameter_set_model(),
    )
    context = report.html_context(results_dir=Path("nowhere"), name="the_fit")

    assert context["fit_id"] == "the_fit"
    # the data the model does not describe is not part of the report
    assert context["kinds"] == [kind.value for kind in EVALUATED_KINDS]
    assert MappingKind.EXCLUDED.value not in context["kinds"]
    assert len(context["parameters"]) == len(op_hctz_pkiv.parameters)
    assert len(context["mappings"]) == len(op_hctz_pkiv.mapping_keys)
    assert context["settings"]["residual"] == fit_settings.residual.name
    assert context["data_total"]["total"] == len(op_hctz_pkiv.mapping_keys)
    # every mapping carries its kind and its metrics
    for mapping in context["mappings"]:
        assert mapping["kind"] in {kind.value for kind in MappingKind}
        assert set(mapping["metrics"]) == {"n", "RMSE", "R²"}


def test_html_report_is_well_formed(
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The tags of the report are balanced."""
    op_hctz_pkiv.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        parameter_sets=op_hctz_pkiv.parameter_set_model(),
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
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Every value of the report says what it means, as a tooltip."""
    op_hctz_pkiv.initialize(fit_settings)
    parameter_set = op_hctz_pkiv.parameter_set_model()
    report = FitReport(
        problem=op_hctz_pkiv, settings=fit_settings, parameter_sets=[parameter_set]
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
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The Fisher information is a section with its table and correlations."""
    op_hctz_pkiv.initialize(fit_settings)
    parameter_set = op_hctz_pkiv.parameter_set_model()
    fisher = fisher_information(op_hctz_pkiv, fit_settings, parameter_set)
    report = FitReport(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        parameter_sets=[parameter_set],
        fisher=fisher,
    )
    results_dir = report.create(tmp_path, name="fisher")

    # the information is stored next to the report
    assert (results_dir / "fisher.json").exists()
    assert (results_dir / "fisher.tsv").exists()

    html = (results_dir / "index.html").read_text()
    assert "Fisher information" in html
    assert "Correlation" in html
    for pid in op_hctz_pkiv.pids:
        assert pid in html
    # the intravenous data does not determine every parameter, and the report
    # says so instead of showing errors which cannot be read
    assert not fisher.is_identifiable
    assert "does not have full rank" in html
