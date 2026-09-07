"""Sensitivity analysis.

This package provides a unified framework for analyzing how uncertainty and
variability in model parameters affect model outputs. It supports multiple
complementary sensitivity analysis strategies, including local, sampling-based,
and global methods. Together, these enable both qualitative and quantitative
assessment of parameter influence on model behavior.

The available sensitivity analysis methods are implemented in the following
modules:

- [`sensitivity.sensitivity_fast`](../api/sensitivity.sensitivity_fast.qmd):
  Global sensitivity analysis using FAST (Fourier Amplitude Sensitivity Test).

- [`sensitivity.sensitivity_local`](../api/sensitivity.sensitivity_local.qmd):
  Local (derivative-based) sensitivity analysis around a nominal parameter set.

- [`sensitivity.sensitivity_morris`](../api/sensitivity.sensitivity_morris.qmd):
  Global screening based on Morris elementary effects.

- [`sensitivity.sensitivity_sampling`](../api/sensitivity.sensitivity_sampling.qmd):
  Sampling-based sensitivity analysis using parameter perturbations and
  statistical summaries.

- [`sensitivity.sensitivity_sobol`](../api/sensitivity.sensitivity_sobol.qmd):
  Variance-based global sensitivity analysis using Sobol indices.

All methods share a common interface and data model. This allows consistent
configuration, execution, and comparison of sensitivity analysis results across
different techniques.
"""

from .analysis import (
    AnalysisGroup,
    SensitivityAnalysis,
    SensitivityOutput,
    SensitivitySimulation,
)
from .parameters import (
    ParameterType,
    SensitivityParameter,
)
from .sensitivity_fast import FASTSensitivityAnalysis
from .sensitivity_local import LocalSensitivityAnalysis
from .sensitivity_morris import MorrisSensitivityAnalysis
from .sensitivity_sampling import SamplingSensitivityAnalysis
from .sensitivity_sobol import SobolSensitivityAnalysis

__all__ = [
    "AnalysisGroup",
    "FASTSensitivityAnalysis",
    "LocalSensitivityAnalysis",
    "MorrisSensitivityAnalysis",
    "ParameterType",
    "SamplingSensitivityAnalysis",
    "SensitivityAnalysis",
    "SensitivityOutput",
    "SensitivityParameter",
    "SensitivitySimulation",
    "SobolSensitivityAnalysis",
]
