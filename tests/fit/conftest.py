"""Fixtures for the parameter fitting tests.

The HCTZ example is the reference fitting problem, `op_hctz_pkiv` is its
smallest subset: a single simulation experiment with four fit mappings.
"""

import pytest

from examples.hctz.fitting.fitting import FIT_DEFINITIONS
from sbmlsim.fit import FitSettings
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)


@pytest.fixture
def op_hctz_pkiv() -> OptimizationProblem:
    """Get the uninitialized optimization problem of the iv pharmacokinetics."""
    return FIT_DEFINITIONS["PKIV"].problem(opid="hctz_pkiv")


@pytest.fixture
def definition_hctz_pkiv() -> FitDefinition:
    """Get the definition of the iv pharmacokinetics fit."""
    return FIT_DEFINITIONS["PKIV"]


@pytest.fixture
def op_hctz_pk() -> OptimizationProblem:
    """Get the problem of all pharmacokinetics data.

    It has training, validation and outlier fit mappings.
    """
    return FIT_DEFINITIONS["PK"].problem(opid="hctz_pk")


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
