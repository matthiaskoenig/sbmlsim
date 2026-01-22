"""Module for sensitivity analysis.

TODO implementation of alternative methods:
    - [ ] FAST
    - [ ] Morris

FIXME: generate simple example => create tests for the sensitivity
FIXME: general documentation
FIXME: add a flag to control resources for parallelization (ncores)

"""

from .sensitivity_sobol import SobolSensitivityAnalysis
from .sensitivity_sampling import SamplingSensitivityAnalysis
from .sensitivity_local import LocalSensitivityAnalysis


