"""Test the report of a fit.

The report is separate from the optimization: it is created from the definition
of the problem, the settings and one or more parameter sets.
"""

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
        fit_experiments=op_hctz_pkiv.fit_experiments,
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
