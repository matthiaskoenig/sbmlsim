"""Fixtures for the parameter fitting tests.

The HCTZ example is the reference fitting problem, `op_hctz_pk` is its
smallest subset: a single simulation experiment with four fit mappings.
"""

import pytest

from examples.hctz_fitting import DATA_PATH, HCTZ_PATH
from examples.hctz_fitting.experiments.metadata import HCTZMappingMetaData, Route
from examples.hctz_fitting.experiments.studies import Beermann1976
from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS
from sbmlsim.fit import FitMapping, FitMappingCollection, FitSettings
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.helpers import select_mapping_collections
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)


@pytest.fixture
def op_hctz_pk() -> OptimizationProblem:
    """Get the uninitialized problem of the pharmacokinetics data.

    It has training, validation and outlier fit mappings.
    """
    return FIT_DEFINITIONS["PK"].problem(opid="hctz_pk")


@pytest.fixture
def definition_hctz_pk() -> FitDefinition:
    """Get the definition of the pharmacokinetics fit."""
    return FIT_DEFINITIONS["PK"]


def _is_iv(fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
    """Accept the iv data."""
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


@pytest.fixture
def definition_hctz_iv() -> FitDefinition:
    """Get the definition of the fit of the iv data of Beermann1976."""
    pk = FIT_DEFINITIONS["PK"]
    return FitDefinition(
        mapping_collections=_collections_iv,
        parameters=pk.parameters,
        base_path=pk.base_path,
        data_path=pk.data_path,
        settings=pk.settings,
    )


@pytest.fixture
def op_hctz_iv(definition_hctz_iv: FitDefinition) -> OptimizationProblem:
    """Get the uninitialized problem of the iv data, four training mappings."""
    return definition_hctz_iv.problem(opid="hctz_iv")


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
