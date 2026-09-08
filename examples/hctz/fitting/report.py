"""Create the report of a HCTZ fit from stored parameters.

Reporting is separate from optimizing: this needs the definition of the fit
problem and one or more sets of parameters, not a fit. The parameters come from
the `parameters.json` a fit wrote, so a report is created again later or for
several fits at once:

    # report the parameters of a finished fit
    python -m examples.hctz.fitting.report results/fit/PK/parameters.json

    # compare the parameters of two fits in one report
    python -m examples.hctz.fitting.report run1/parameters.json run2/parameters.json
"""

from examples.hctz.fitting.fitting import FIT_DEFINITIONS
from sbmlsim.fit.cli import report_cli


def main() -> None:
    """Report the given parameter sets of the HCTZ model."""
    report_cli(FIT_DEFINITIONS)


if __name__ == "__main__":
    main()
