"""Identifiability of the HCTZ parameters: global optimization and profiles.

A global optimization finds the best parameters of the model, the profile
likelihood then says how well the data determines every one of them. The fit
runs differential evolution, the profiles are computed around its best
parameter set, and one report carries the fit and the identifiability section
with the profiles:

    python -m examples.hctz_fitting.fitting.identifiability --subset=PK --runs=2 --cores=4

The report is written into `results/identifiability` in the working directory.
The profiles of stored parameters, without fitting again, are computed with
`python -m examples.hctz_fitting.fitting.identifiability_report <parameters.json>`.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# run as a script (`python examples/hctz_fitting/fitting/identifiability.py`, the "run
# file" of an IDE) the repository is not on `sys.path`, so the `examples`
# package is not found
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS
from sbmlsim import log
from sbmlsim.fit.cli import run_fit
from sbmlsim.fit.identifiability import ProfileSettings
from sbmlsim.fit.options import OptimizationAlgorithmType


def main(args: Sequence[str] | None = None) -> Path:
    """Fit the HCTZ model globally and analyse the identifiability of the fit.

    Args:
        args: command line arguments, `sys.argv` by default.

    Returns:
        Path of the directory the report was written to.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "-x",
        "--subset",
        choices=list(FIT_DEFINITIONS),
        default="PK",
        help="fit problem to run",
    )
    parser.add_argument(
        "-r", "--runs", type=int, default=2, help="repeats of the global optimization"
    )
    parser.add_argument(
        "-c", "--cores", type=int, default=2, help="cores for the fit and the scans"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=1234, help="seed of the optimization"
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=50,
        help="generations of the differential evolution",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("results") / "identifiability",
        help="directory for the report",
    )
    options = parser.parse_args(args)
    log.enable_rich_logging()
    definition = FIT_DEFINITIONS[options.subset]

    # 1. global optimization: differential evolution, every repeat with its
    #    own seed derived from the seed of the fit
    runs = run_fit(
        definition=definition,
        opid=options.subset,
        algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
        size=options.runs,
        n_cores=options.cores,
        seed=options.seed,
        output_dir=options.output_dir,
        maxiter=options.maxiter,
        polish=False,
    )
    run = runs[options.subset]

    # 2. profile likelihood around the best parameter set of the fit, two scans
    #    per parameter in parallel
    identifiability = run.identifiability(
        profile_settings=ProfileSettings(alpha=0.95, max_points=30),
        n_cores=options.cores,
    )

    # 3. one report with the fit and the identifiability section
    return run.report(
        output_dir=options.output_dir,
        name=f"{options.subset}_DE",
        identifiability=identifiability,
        show_titles=False,
    )


if __name__ == "__main__":
    main()
