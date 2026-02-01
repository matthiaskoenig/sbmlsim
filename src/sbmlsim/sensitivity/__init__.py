"""Sensitivity analysis.

This package provides a unified framework for analyzing how uncertainty and
variability in model parameters affect model outputs. It supports multiple
complementary sensitivity analysis strategies, including local, sampling-based,
and global methods, enabling both qualitative and quantitative assessment of
parameter influence.
"""

from .analysis import (
    SensitivityAnalysis,
    SensitivitySimulation,
    SensitivityOutput,
    AnalysisGroup,
)
from .parameters import (
    ParameterType,
    SensitivityParameter,
)
from .sensitivity_fast import FASTSensitivityAnalysis
from .sensitivity_local import LocalSensitivityAnalysis
from .sensitivity_sampling import SamplingSensitivityAnalysis
from .sensitivity_sobol import SobolSensitivityAnalysis
from .sensitivity_morris import MorrisSensitivityAnalysis

__all__ = [
    "ParameterType",
    "SensitivityParameter",
    "SensitivityAnalysis",
    "SensitivitySimulation",
    "SensitivityOutput",
    "AnalysisGroup",
    "SobolSensitivityAnalysis",
    "SamplingSensitivityAnalysis",
    "LocalSensitivityAnalysis",
    "FASTSensitivityAnalysis",
    "MorrisSensitivityAnalysis",
]
