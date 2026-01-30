"""
Sensitivity analysis framework for computational models.

This package provides a unified framework for analyzing how uncertainty and
variability in model parameters affect model outputs. It supports multiple
complementary sensitivity analysis strategies, including local, sampling-based,
and global methods, enabling both qualitative and quantitative assessment of
parameter influence.

Sensitivity analyses are performed by systematically perturbing or sampling
model parameters, executing simulations, and evaluating changes in selected
model outputs. The framework is designed for deterministic simulation models
and integrates sampling, caching, statistical evaluation, and visualization
within a consistent workflow.
"""
from .analysis import (
    SensitivityAnalysis,
    SensitivitySimulation,
    SensitivityOutput,
    AnalysisGroup,
)
from .parameters import (
    SensitivityParameter,
)
from .sensitivity_fast import FASTSensitivityAnalysis
from .sensitivity_local import LocalSensitivityAnalysis
from .sensitivity_sampling import SamplingSensitivityAnalysis
from .sensitivity_sobol import SobolSensitivityAnalysis

__all__ = [
    "SensitivityParameter",
    "SensitivityAnalysis",
    "SensitivitySimulation",
    "SensitivityOutput",
    "AnalysisGroup",
    "SobolSensitivityAnalysis",
    "SamplingSensitivityAnalysis",
    "LocalSensitivityAnalysis",
    "FASTSensitivityAnalysis",
]
