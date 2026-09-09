"""Hydrochlorothiazide (HCTZ) physiologically based pharmacokinetics example.

A whole body model of hydrochlorothiazide with the simulation experiments of two
studies and the parameter fitting problem built on them. The example is the
reference problem for `sbmlsim.fit`.
"""

from pathlib import Path

HCTZ_PATH = Path(__file__).parent
MODEL_PATH = HCTZ_PATH / "models" / "hctz_body_flat.xml"
DATA_PATH = HCTZ_PATH / "data"
