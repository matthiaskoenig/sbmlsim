"""Fixtures for the parameter fitting tests.

The HCTZ example is the reference fitting problem, `op_hctz_pk` is its
smallest subset: a single simulation experiment with four fit mappings.

Building a problem selects the fit mappings of the experiments, which takes
0.1 to 0.2 s. A session builds each problem once and a test gets a deep copy
of it, 0.1 ms: a test is free to initialize or to change its problem, since
the copy is its own. With pytest-xdist every worker has its own session and
builds the problems once.
"""

import copy
from pathlib import Path
from typing import Any

import pytest

from examples.hctz_fitting import DATA_PATH, HCTZ_PATH
from examples.hctz_fitting.experiments.metadata import HCTZMappingMetaData, Route
from examples.hctz_fitting.experiments.studies import Beermann1976
from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS
from sbmlsim.fit import FitMapping, FitMappingCollection, FitSettings
from sbmlsim.fit.cli import FitDefinition, run_fit
from sbmlsim.fit.helpers import select_mapping_collections
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.result import OptimizationResult


@pytest.fixture(scope="session")
def _op_hctz_pk_template() -> OptimizationProblem:
    """Build the uninitialized problem of the pharmacokinetics data once.

    Only the copies in `op_hctz_pk` are handed to a test, so this problem is
    never initialized.
    """
    return FIT_DEFINITIONS["PK"].problem(opid="hctz_pk")


@pytest.fixture
def op_hctz_pk(_op_hctz_pk_template: OptimizationProblem) -> OptimizationProblem:
    """Get the uninitialized problem of the pharmacokinetics data.

    It has training, validation and outlier fit mappings.
    """
    return copy.deepcopy(_op_hctz_pk_template)


@pytest.fixture
def definition_hctz_pk() -> FitDefinition:
    """Get the definition of the pharmacokinetics fit."""
    return FIT_DEFINITIONS["PK"]


def _is_iv(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Accept the iv data."""
    metadata = fit_mapping.metadata
    return isinstance(metadata, HCTZMappingMetaData) and metadata.route is Route.IV


def is_oral(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Select the oral mappings of the HCTZ problem.

    A module level function, not a lambda, because a selector is pickled with
    the `FitParameter` that carries it and the workers of a parallel fit
    unpickle the parameters.
    """
    metadata = fit_mapping.metadata
    return isinstance(metadata, HCTZMappingMetaData) and metadata.route is Route.PO


def is_intravenous(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Select the intravenous mappings of the HCTZ problem.

    A module level function, not a lambda, for the same reason as `is_oral`.
    """
    metadata = fit_mapping.metadata
    return isinstance(metadata, HCTZMappingMetaData) and metadata.route is Route.IV


def _collections_iv() -> dict[str, list[FitMappingCollection]]:
    """Select the iv data of Beermann1976, the rest is excluded."""
    return select_mapping_collections(
        experiment_classes=[Beermann1976],
        base_path=HCTZ_PATH,
        data_path=DATA_PATH,
        filters=[_is_iv],
        print_info=False,
    )


def _definition_iv() -> FitDefinition:
    """Create the definition of the fit of the iv data of Beermann1976."""
    pk = FIT_DEFINITIONS["PK"]
    return FitDefinition(
        mapping_collections=_collections_iv,
        parameters=pk.parameters,
        base_path=pk.base_path,
        data_path=pk.data_path,
        settings=pk.settings,
    )


@pytest.fixture
def definition_hctz_iv() -> FitDefinition:
    """Get the definition of the fit of the iv data of Beermann1976."""
    return _definition_iv()


@pytest.fixture(scope="session")
def _op_hctz_iv_template() -> OptimizationProblem:
    """Build the uninitialized problem of the iv data once.

    Only the copies in `op_hctz_iv` are handed to a test, so this problem is
    never initialized.
    """
    return _definition_iv().problem(opid="hctz_iv")


@pytest.fixture
def op_hctz_iv(_op_hctz_iv_template: OptimizationProblem) -> OptimizationProblem:
    """Get the uninitialized problem of the iv data, four training mappings."""
    return copy.deepcopy(_op_hctz_iv_template)


@pytest.fixture
def fit_settings() -> FitSettings:
    """Get the default settings of an optimization."""
    return FitSettings(
        residual=ResidualType.NORMALIZED,
        weighting_curves=(WeightingCurvesType.POINTS,),
        weighting_points=WeightingPointsType.ERROR_WEIGHTING,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-6,
    )


@pytest.fixture
def short_fit() -> dict[str, Any]:
    """Get the arguments of a least squares fit which stops after a few steps.

    A test of what a fit produces, rather than of where it converges, does not
    wait for the convergence: a fit of the HCTZ problem to convergence takes up
    to 40 s, a step takes a fraction of a second.
    """
    return {"max_nfev": 3}


@pytest.fixture(scope="session")
def _fitted_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[OptimizationProblem, OptimizationResult]:
    """Run one quick fit of the pharmacokinetics data for the whole session.

    Two runs of one evaluation each with the seed 1234: the result has the
    runs, the optimizer and the provenance of a fit, not an optimum.
    """
    runs = run_fit(
        FIT_DEFINITIONS["PK"],
        size=2,
        n_cores=1,
        seed=1234,
        output_dir=tmp_path_factory.mktemp("fitted"),
        max_nfev=1,
    )
    (run,) = runs.values()
    return run.problem, run.result


@pytest.fixture
def fitted(
    _fitted_template: tuple[OptimizationProblem, OptimizationResult],
) -> tuple[OptimizationProblem, OptimizationResult]:
    """Get the initialized problem and the result of a quick fit.

    The result is a copy of its own, a test may change it. The problem is the
    one of the session, which a test must not change.
    """
    problem, result = _fitted_template
    return problem, copy.deepcopy(result)


@pytest.fixture(scope="session")
def fitted_report(tmp_path_factory: pytest.TempPathFactory) -> tuple[FitReport, Path]:
    """Create the report of a quick fit of a single run for the whole session.

    One run of one evaluation with the seed 1234, without an identifiability
    analysis and without the figures of the fit mappings: the report has no
    waterfall plot, no profiles and no Fisher information. The report is
    created, so its plots exist in the returned directory. A test must not
    change the report; it copies it to change it.
    """
    output_dir = tmp_path_factory.mktemp("fitted_report")
    runs = run_fit(
        FIT_DEFINITIONS["PK"],
        size=1,
        n_cores=1,
        seed=1234,
        output_dir=output_dir,
        max_nfev=1,
    )
    (run,) = runs.values()
    report = FitReport.from_optimization_result(
        problem=run.problem, opt_result=run.result, mapping_figures=False
    )
    results_dir = report.create(output_dir=output_dir / "report")
    return report, results_dir / "plots"
