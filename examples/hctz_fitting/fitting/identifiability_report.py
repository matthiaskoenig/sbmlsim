"""Identifiability of stored HCTZ parameters by profile likelihood.

The profiles are computed around the parameters a fit wrote, so the analysis
runs without fitting again:

    python -m examples.hctz_fitting.fitting.identifiability_report results/fit/PK/parameters.json

The report with the profiles is written into `results/identifiability`.
"""

import sys
from pathlib import Path

# run as a script (`python examples/hctz_fitting/fitting/identifiability_report.py`, the
# "run file" of an IDE) the repository is not on `sys.path`, so the `examples`
# package is not found
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS
from sbmlsim.fit.cli import identifiability_cli


def main() -> None:
    """Analyse the identifiability of the given parameter set of the HCTZ model."""
    identifiability_cli(FIT_DEFINITIONS)


if __name__ == "__main__":
    main()
