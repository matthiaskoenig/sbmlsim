"""AMICI-generated module for model icg_sd"""

import amici
from pathlib import Path

# Ensure we are binary-compatible, see #556
if '0.16.1' != amici.__version__:
    raise amici.AmiciVersionError(
        f'Cannot use model `icg_sd` in {Path(__file__).parent}, '
        'generated with amici==0.16.1, '
        f'together with amici=={amici.__version__} '
        'which is currently installed. To use this model, install '
        'amici==0.16.1 or re-import the model with the amici '
        'version currently installed.'
    )

from .icg_sd import *
from .icg_sd import getModel as get_model

__version__ = '0.1.0'
