"""Test the report of a fit.

The report is separate from the optimization: it is created from the definition
of the problem, the settings and one or more parameter sets.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from sbmlsim.fit import FitSettings, ParameterSet, ParameterSets
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
    """The report of a fit compares the fit against the model values."""
    opt_result = _fit(op_hctz_pkiv, fit_settings)
    report = FitReport.from_optimization_result(
        problem=op_hctz_pkiv, opt_result=opt_result
    )

    # the model values are the reference, the fit is the second set
    assert len(report.parameter_sets) == 2
    assert report.reference_set.sid == "model"

    results_dir = report.create(output_dir=tmp_path, name="fit")
    _assert_report_files(results_dir)
    # the plots of the runs need the result of an optimization
    assert (results_dir / "plots" / "traces.svg").exists()
    assert (results_dir / "optimization_result.json").exists()


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
    assert context["kinds"] == ["training", "validation", "outlier"]
    assert len(context["parameters"]) == len(op_hctz_pkiv.parameters)
    assert len(context["mappings"]) == len(op_hctz_pkiv.mapping_keys)
    assert context["settings"]["residual"] == fit_settings.residual.name
    assert context["data_total"]["total"] == len(op_hctz_pkiv.mapping_keys)
    # every mapping carries its kind and its metrics
    for mapping in context["mappings"]:
        assert mapping["kind"] in {"training", "validation", "outlier"}
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
