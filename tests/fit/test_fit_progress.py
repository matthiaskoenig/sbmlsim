"""The progress of the runs of a fit and of the points of their jacobians."""

from __future__ import annotations

from typing import Any

import pytest

from sbmlsim import progress
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, OptimizationAlgorithmType
from sbmlsim.fit.runner import run_optimization


def _run(problem: OptimizationProblem, settings: FitSettings, **kwargs: Any) -> None:
    """Run a fit of two runs which stops after three evaluations each."""
    run_optimization(
        problem=problem,
        settings=settings,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        size=2,
        serial=True,
        max_nfev=3,
        **kwargs,
    )


def test_a_fit_reports_its_runs(
    recorder, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The runs of a fit are the steps of an activity with their estimate."""
    with progress.reporter(recorder):
        _run(op_hctz_pk, fit_settings)
    fits = [e for e in recorder.events if e.kind == "fit"]
    assert fits
    assert (fits[-1].completed, fits[-1].total) == (2, 2)
    assert fits[0].label == "optimizing"


def test_a_fit_without_progress_reports_nothing(
    recorder, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """`show_progress=False` reports no activity."""
    with progress.reporter(recorder):
        _run(op_hctz_pk, fit_settings, show_progress=False)
    assert recorder.events == []


@pytest.mark.slow
def test_a_fit_reports_the_points_of_its_jacobians(
    recorder, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A jacobian on the workers of a fit reports its points, nested in the fit."""
    with progress.reporter(recorder):
        run_optimization(
            problem=op_hctz_pk,
            settings=fit_settings,
            algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
            size=1,
            n_cores=2,
            max_nfev=3,
        )
    kinds = {(e.kind, e.depth) for e in recorder.events}
    assert ("fit", 0) in kinds
    assert ("jacobian", 1) in kinds


def test_an_identifiability_reports_its_scans_and_their_points(
    recorder, fitted
) -> None:
    """The scans are the steps of the analysis, the points those of a scan."""
    import copy

    from sbmlsim.fit.identifiability import ProfileSettings, profile_likelihood

    problem, result = fitted
    problem = copy.deepcopy(problem)
    settings = result.settings_stored
    problem.initialize(settings)
    parameter_set = result.parameter_set()
    with progress.reporter(recorder):
        profile_likelihood(
            problem=problem,
            settings=settings,
            parameter_set=parameter_set,
            profile_settings=ProfileSettings(reoptimize=False, max_points=2),
            pids=[problem.pids[0]],
            serial=True,
        )
    analysis = [e for e in recorder.events if e.kind == "identifiability"]
    assert analysis
    assert (analysis[-1].completed, analysis[-1].total) == (2, 2)
    assert analysis[0].label == "profiling"
    scans = [e for e in recorder.events if e.kind == "scan"]
    assert scans
    assert {e.depth for e in scans} == {1}
    assert any(e.label.endswith("up") for e in scans)


def test_a_fit_report_reports_a_figure_per_mapping(recorder, fitted, tmp_path) -> None:
    """The figures of the fit mappings are the long part of a report."""
    from sbmlsim.fit.report import FitReport

    problem, result = fitted
    report = FitReport.from_optimization_result(
        problem=problem, opt_result=result, mapping_figures=True
    )
    with progress.reporter(recorder):
        report.create(output_dir=tmp_path)
    figures = [e for e in recorder.events if e.kind == "report"]
    assert figures
    assert figures[0].label == "mapping figures"
    assert figures[-1].completed == figures[-1].total == len(problem.mapping_keys)
