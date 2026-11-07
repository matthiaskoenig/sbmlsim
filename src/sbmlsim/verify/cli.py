"""The command line of `verify`, run as `python -m sbmlsim.run verify`."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Sequence

from sbmlsim.run.cli import EXIT_CONFIGURATION, EXIT_USAGE
from sbmlsim.verify.common import ProjectNotLoaded, UsageError, add_project_arguments
from sbmlsim.verify.compare import compare, ranges
from sbmlsim.verify.definitions import definitions, fits, settings
from sbmlsim.verify.dump import dump
from sbmlsim.verify.files import png, sbml
from sbmlsim.verify.scorecard import scorecard_command

__all__ = [
    "DESCRIPTION",
    "parser",
    "verify",
]

logger = logging.getLogger(__name__)

DESCRIPTION = (
    "Verify that a change of a model changes only what it is meant to change: "
    "dump the results of the experiments of a project and compare two dumps, "
    "and compare fit definitions, settings, PNG figures, SBML models and the "
    "credibility of many models."
)


def parser() -> argparse.ArgumentParser:
    """Create the parser of the command line."""
    p = argparse.ArgumentParser(
        prog="python -m sbmlsim.run verify",
        description=DESCRIPTION,
    )
    sub = p.add_subparsers(required=True)

    def command(
        name: str, function: Callable[[argparse.Namespace], int], help_: str
    ) -> argparse.ArgumentParser:
        s = sub.add_parser(name, help=help_, description=help_)
        s.set_defaults(function=function)
        return s

    s = command("dump", dump, "run the experiments of a project and store everything")
    add_project_arguments(s)
    s.add_argument("--out", required=True, help="pickle the dump is written to")
    s.add_argument(
        "--all",
        action="store_true",
        help="also the experiments defined below the package of the project which are not in its experiments",
    )
    s.add_argument("--order", help="run in the order of a previous dump")
    s.add_argument(
        "--skip", default=[], type=lambda s: s.split(","), help="A,B: leave these out"
    )
    s.add_argument(
        "--atol", help="absolute tolerance, or `model` for that of the project"
    )
    s.add_argument(
        "--rtol", help="relative tolerance, or `model` for that of the project"
    )
    s.add_argument(
        "--reduced-selections",
        action="store_true",
        help="only the selections the experiments need, as a run of the project does",
    )
    s.add_argument(
        "--grid",
        type=_positive_int,
        default=None,
        metavar="N",
        help="output every simulation at N + 1 equidistant times from its start to "
        "its end and the times of its changes instead of the steps of the "
        "integrator, so two dumps share their times; a migration compare dumps "
        "both with the same grid",
    )
    s.add_argument("experiments", nargs="*", help="only these experiment classes")

    s = command("compare", compare, "compare two dumps")
    s.add_argument("old")
    s.add_argument("new")
    s.add_argument(
        "--renames",
        help=(
            "json file of the expected renames and differences (metadata, strings, "
            "ids, sids, figure_fields, expected), applied to the old dump; in a "
            "pattern of expected, * and ? are wildcards and a bracket is literal, "
            "experiment/task/time names the output times only the new dump has"
        ),
    )
    s.add_argument("--threshold", type=float, default=1e-6, help="default 1e-6")
    s.add_argument("--top", type=int, default=20, help="lines per listing")
    s.add_argument("experiments", nargs="*", help="only these experiments")

    s = command("ranges", ranges, "fit mappings with data outside of the simulation")
    s.add_argument("dump")
    s.add_argument("--verbose", action="store_true", help="every mapping")

    s = command("fits", fits, "a minimal fit of every fit definition of a project")
    add_project_arguments(s)
    s.add_argument("--out", required=True, help="output directory of the fits")
    s.add_argument("--max-nfev", type=int, default=3, help="evaluations per start")
    s.add_argument("subsets", nargs="*", help="only these fit definitions")

    s = command("definitions", definitions, "the fit definitions of a project as json")
    add_project_arguments(s)
    s.add_argument(
        "--out", required=True, help="json file the definitions are written to"
    )

    s = command(
        "settings", settings, "the settings a run of a project hands to sbmlsim as json"
    )
    add_project_arguments(s)
    s.add_argument("--out", required=True, help="json file the settings are written to")
    s.add_argument(
        "--experiment",
        help="the experiment class to run, the first of the registry by default",
    )

    s = command("png", png, "compare the PNG figures of two directories")
    s.add_argument("old")
    s.add_argument("new")
    s.add_argument(
        "--sids",
        action="store_true",
        help=(
            "pair an old figure with the new figure of the SId of its file name "
            "(sbmlsim.experiment.to_sid), for a port which renamed the keys"
        ),
    )
    s.add_argument("names", nargs="*", help="only files containing one of these")

    s = command("sbml", sbml, "compare two SBML files or the SBML files of two dirs")
    s.add_argument("old")
    s.add_argument("new")

    s = command(
        "scorecard",
        scorecard_command,
        "the credibility of many models: the scorecard, the qualification of the "
        "ratios across models and the regressions against a baseline",
    )
    s.add_argument(
        "records", nargs="+", help="credibility.json files or directories with them"
    )
    add_project_arguments(s, required=False)
    s.add_argument(
        "--extensions",
        action="append",
        default=[],
        metavar="MODULE:ATTRIBUTE",
        help="an evidence extension or an iterable of them, repeated; "
        "--project adds the extensions of a project",
    )
    s.add_argument(
        "--pool",
        help="the name of the extension which pools a qualification across "
        "the models (needed for --qualification)",
    )
    s.add_argument(
        "--baseline", help="a scorecard TSV to compare with, a regression exits 1"
    )
    s.add_argument("--out", help="TSV file the scorecard is written to")
    s.add_argument("--qualification", help="TSV file the qualification is written to")
    s.add_argument(
        "--field",
        default=None,
        help="an option of the pooling of the extension, e.g. the field of the metadata the ratios are pooled by; its default without",
    )
    s.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="the change of a metric which is no regression, default 0.05",
    )
    return p


def _positive_int(text: str) -> int:
    """Parse a positive integer of the command line."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"a positive number, not {text}")
    return value


def verify(args: Sequence[str] | None = None) -> int:
    """Run a subcommand.

    Args:
        args: the command line arguments, `sys.argv[1:]` by default.

    Returns:
        The exit code: 0, 1 if the subcommand found a difference, 2 for a
        command which cannot be run as it was asked (`UsageError`, or the
        arguments, which argparse rejects) and 4 for a project which cannot be
        loaded (`ProjectNotLoaded`, as the kinds of the run layer).
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        options = parser().parse_args(args)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else EXIT_USAGE
    try:
        return options.function(options)
    except ProjectNotLoaded as err:
        logger.error("%s", err)
        return EXIT_CONFIGURATION
    except UsageError as err:
        logger.error("%s", err)
        return EXIT_USAGE
