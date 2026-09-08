"""Fixtures for the parameter fitting tests.

The HCTZ example is the reference fitting problem, `op_hctz_pkiv` is its
smallest subset: a single simulation experiment with four fit mappings.
"""

import pytest

from examples.hctz.fitting.fitting import FitExperimentSubset, op_hctz
from sbmlsim.fit import FitSettings
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)


@pytest.fixture
def op_hctz_pkiv() -> OptimizationProblem:
    """Get the uninitialized optimization problem of the iv pharmacokinetics."""
    return op_hctz(FitExperimentSubset.PKIV)


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
