"""Create the report of a HCTZ fit from stored parameters.

Reporting is separate from optimizing: this script needs the definition of the
optimization problem and one or more sets of parameters, not a fit. The
parameters come from the `parameters.json` a fit wrote, so a report can be
created again later, with different settings, or for several sets at once:

    # report the parameters of a finished fit
    python -m examples.hctz.fitting.report results/fit/hctz/parameters.json

    # compare the parameters of two fits in one report
    python -m examples.hctz.fitting.report run1/parameters.json run2/parameters.json
"""

import argparse
from pathlib import Path

from examples.hctz.fitting.fitting import FIT_SETTINGS, FitExperimentSubset, op_hctz
from sbmlsim import log
from sbmlsim.console import console
from sbmlsim.fit import ParameterSet, ParameterSets
from sbmlsim.fit.report import FitReport


def load_parameter_sets(paths: list[Path]) -> ParameterSets:
    """Load the parameter sets of the given JSON files.

    The sets of all files are combined into a single report; a set which occurs
    in more than one file is prefixed with the name of its file to keep the
    identifiers unique.

    Args:
        paths: JSON files written by `ParameterSets.to_json`.

    Returns:
        All parameter sets of the files.
    """
    sets: list[ParameterSet] = []
    sids: set[str] = set()
    for path in paths:
        for pset in ParameterSets.from_json(path):
            if pset.sid in sids:
                pset.sid = f"{path.parent.name}_{pset.sid}"
            sids.add(pset.sid)
            sets.append(pset)

    return ParameterSets(sets)


def main(args: list[str] | None = None) -> None:
    """Create the report of the given parameter sets."""
    parser = argparse.ArgumentParser(
        prog="report_hctz",
        description="Report of the HCTZ fit for one or more parameter sets.",
    )
    parser.add_argument(
        "parameters",
        type=Path,
        nargs="+",
        help="JSON files with the parameter sets to report",
    )
    parser.add_argument(
        "-x",
        "--subset",
        type=FitExperimentSubset,
        choices=list(FitExperimentSubset),
        default=FitExperimentSubset.PK,
        help="subset of the fit experiments the parameters belong to",
    )
    parser.add_argument("-n", "--name", default="report", help="name of the report")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("results") / "report",
        help="directory for the report",
    )
    options = parser.parse_args(args)
    log.enable_rich_logging()

    console.rule(":bar_chart: REPORT HCTZ :bar_chart:", align="left", style="white")

    parameter_sets = load_parameter_sets(options.parameters)
    console.print(f"{'sets':<12}: {[pset.sid for pset in parameter_sets]}")

    # only the definition of the problem is needed, no fit was run here
    problem = op_hctz(options.subset)
    report = FitReport(
        problem=problem,
        settings=FIT_SETTINGS,
        parameter_sets=parameter_sets,
        show_titles=False,
    )
    report.create(output_dir=options.output_dir, name=options.name)


if __name__ == "__main__":
    main()
