"""Hydrochlorothiazide (HCTZ) physiologically based pharmacokinetics example.

A whole body model of hydrochlorothiazide with the simulation experiments of two
studies and the parameter fitting problem built on them. The example is the
reference problem for `sbmlsim.fit`.
"""

from pathlib import Path

from sbmlsim.resources import HCTZ_SBML

HCTZ_PATH = Path(__file__).parent
#: the model is part of the package, the tutorial of the documentation uses it
MODEL_PATH = HCTZ_SBML
DATA_PATH = HCTZ_PATH / "data"
