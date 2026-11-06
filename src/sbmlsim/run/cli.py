"""The command line of the run layer, `python -m sbmlsim.run`.

    python -m sbmlsim.run <kind> <spec> --out DIR [--workers N] [--resume]
        [--provenance FILE] [--summary FILE] [--input PORT=PATH ...]
    python -m sbmlsim.run plan <kind> <spec> [-o FILE]
    python -m sbmlsim.run unit <kind> <spec> <index> --out DIR [--input PORT=PATH ...]
    python -m sbmlsim.run merge <kind> <spec> --out DIR [--summary FILE]
        [--input PORT=PATH ...]
    python -m sbmlsim.run list <project> [--json] [--experiments]

`--profile` anywhere after the program runs a command which writes into
`--out` (`run`, `unit`, `merge`) under the sampling profiler of python 3.15 and
writes the profile into `profiling/` of the output, see
`sbmlsim.run.profiling`; with an older python it is a bad command line.

A kind is the name of a registered kind or `module:attribute` of a kind, a
spec a YAML or JSON file, an input `PORT=PATH` split at the first `=`, also
as `--input=PORT=PATH`. `run` is the command when the first argument names no
other command. `plan` writes the planned spec as JSON, `list` the analyses of
a project as a table or as JSON, with `--experiments` also its groups of
experiments and every experiment, which the kind `simulate` runs by these
names. `--summary FILE` writes the summary of a
successful run or merge also into a file, e.g. where simrun reads the summary
of a task.

The exit codes: 0 done, 2 a bad command line, 3 the analysis failed (its
traceback is logged), also by an invalid object the analysis builds, 4 a
configuration error (`ConfigurationError`: an invalid spec or input, an
unknown kind or project, a planned spec which does not match what it resolves
to, an output which is not empty without `--resume` or a resume of another
spec), its message on stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from sbmlsim.run.kinds import RESERVED_NAMES, ConfigurationError
from sbmlsim.run.profiling import (
    PROFILE_OPTION,
    ProfilerUnavailable,
    profile_dir,
    run_profiled,
)

__all__ = [
    "EXIT_ANALYSIS",
    "EXIT_CONFIGURATION",
    "EXIT_OK",
    "EXIT_USAGE",
    "input_pair",
    "main",
]

logger = logging.getLogger(__name__)

#: the run is done
EXIT_OK = 0
#: the command line is bad (argparse)
EXIT_USAGE = 2
#: the analysis failed
EXIT_ANALYSIS = 3
#: the run is configured wrongly
EXIT_CONFIGURATION = 4

#: the commands; `run` is the command of a first argument which is none
COMMANDS: frozenset[str] = RESERVED_NAMES

#: the commands which write into `--out`, which `--profile` profiles
PROFILED_COMMANDS: frozenset[str] = frozenset({"run", "unit", "merge"})


def input_pair(value: str) -> tuple[str, Path]:
    """Parse an input `PORT=PATH`, split at the first `=`.

    Args:
        value: the argument.

    Returns:
        The port and the path.

    Raises:
        argparse.ArgumentTypeError: without `=` or with an empty port or path.
    """
    port, equals, path = value.partition("=")
    if not equals or not port or not path:
        raise argparse.ArgumentTypeError(f"an input is PORT=PATH, not '{value}'")
    return port, Path(path)


def _positive(value: str) -> int:
    """Parse a positive number."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is no number") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"'{value}' is not positive")
    return number


def _parser() -> argparse.ArgumentParser:
    """Build the parser of the commands."""
    parser = argparse.ArgumentParser(
        prog="python -m sbmlsim.run",
        description=(
            "Run an analysis of sbmlsim from a spec into an output directory, "
            "whole or in units which merge. `run` is the command when the first "
            "argument is a kind."
        ),
    )
    # taken out of the arguments before they are parsed, so it may be given
    # anywhere; here for the help
    parser.add_argument(
        PROFILE_OPTION,
        action="store_true",
        help="run a command which writes into --out under the sampling profiler "
        "of python 3.15, the profile into profiling/ of the output",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def kind_and_spec(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "kind", help="the kind, e.g. fit, or module:attribute of a kind"
        )
        command.add_argument("spec", type=Path, help="the spec, a YAML or JSON file")

    def out_and_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--out", type=Path, required=True, help="the output directory"
        )
        command.add_argument(
            "--input",
            type=input_pair,
            action="append",
            default=[],
            metavar="PORT=PATH",
            help="the path of an input field of the spec, repeated per input",
        )

    def summary_file(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--summary",
            type=Path,
            default=None,
            metavar="FILE",
            help="write the summary of a successful run also into this file",
        )

    run = commands.add_parser("run", help="run an analysis whole")
    kind_and_spec(run)
    out_and_inputs(run)
    summary_file(run)
    run.add_argument(
        "--workers",
        type=_positive,
        default=None,
        help="the number of processes, the kind decides without",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="continue the run of the same spec in the output",
    )
    run.add_argument(
        "--provenance",
        type=Path,
        default=None,
        help="the provenance of the sources, $SIMRUN_PROVENANCE without",
    )

    plan = commands.add_parser("plan", help="write the planned spec as JSON")
    kind_and_spec(plan)
    plan.add_argument(
        "-o", "--output", type=Path, default=None, help="the file, stdout without"
    )

    unit = commands.add_parser("unit", help="compute one unit of a planned spec")
    kind_and_spec(unit)
    unit.add_argument("index", type=int, help="the index of the unit")
    out_and_inputs(unit)

    merge = commands.add_parser("merge", help="merge the units of a planned spec")
    kind_and_spec(merge)
    out_and_inputs(merge)
    summary_file(merge)

    analyses = commands.add_parser("list", help="list the analyses of a project")
    analyses.add_argument("project", help="the project, module:attribute")
    analyses.add_argument("--json", action="store_true", help="write JSON")
    analyses.add_argument(
        "--experiments",
        action="store_true",
        help="also list the groups of experiments and every experiment",
    )
    return parser


def _table(entries: list[dict[str, str]]) -> str:
    """Format the analyses of a project as a table of kind, name and description."""
    columns = ("kind", "name", "description")
    rows = [columns, *[tuple(entry[c] for c in columns) for entry in entries]]
    widths = [max(len(row[i]) for row in rows) for i in range(2)]
    return "".join(
        f"{row[0]:<{widths[0]}}  {row[1]:<{widths[1]}}  {row[2]}\n" for row in rows
    )


def _command(args: argparse.Namespace) -> None:
    """Run a parsed command."""
    import sbmlsim.run as api

    inputs = dict(getattr(args, "input", None) or [])
    if args.command == "run":
        api.execute(
            args.kind,
            args.spec,
            args.out,
            workers=args.workers,
            resume=args.resume,
            provenance=args.provenance,
            inputs=inputs,
            summary=args.summary,
        )
    elif args.command == "plan":
        text = api.plan(args.kind, args.spec).model_dump_json(indent=2) + "\n"
        if args.output is None:
            sys.stdout.write(text)
        else:
            args.output.write_text(text, encoding="utf-8")
    elif args.command == "unit":
        api.compute_unit(args.kind, args.spec, args.index, args.out, inputs=inputs)
    elif args.command == "merge":
        api.merge(args.kind, args.spec, args.out, inputs=inputs, summary=args.summary)
    else:
        entries = api.list_analyses(args.project, experiments=args.experiments)
        if args.json:
            sys.stdout.write(json.dumps(entries, indent=2) + "\n")
        else:
            sys.stdout.write(_table(entries))


def _profiled(args: argparse.Namespace, arguments: list[str]) -> int:
    """Run a parsed command under the profiler, see `sbmlsim.run.profiling`.

    Args:
        args: the parsed command.
        arguments: the arguments of the command, without `--profile`.

    Returns:
        The exit code the command recorded (1 if none, see
        `sbmlsim.run.profiling`), `EXIT_USAGE` for a command which writes no
        output and a python without the profiler.
    """
    if args.command not in PROFILED_COMMANDS:
        sys.stderr.write(
            f"{PROFILE_OPTION} profiles the commands which write into --out, "
            f"{sorted(PROFILED_COMMANDS)}, not '{args.command}'.\n"
        )
        return EXIT_USAGE
    try:
        folder = profile_dir(
            args.out, args.command, args.kind, getattr(args, "index", None)
        )
        return run_profiled(arguments, folder)
    except ProfilerUnavailable as err:
        sys.stderr.write(f"{err}\n")
        return EXIT_USAGE


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: the arguments without the program, `sys.argv[1:]` without.

    Returns:
        The exit code, see the module.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    profile = PROFILE_OPTION in arguments
    arguments = [argument for argument in arguments if argument != PROFILE_OPTION]
    if arguments and arguments[0] not in COMMANDS and not arguments[0].startswith("-"):
        arguments.insert(0, "run")
    try:
        args = _parser().parse_args(arguments)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else EXIT_USAGE
    if profile:
        return _profiled(args, arguments)

    try:
        _command(args)
    except ConfigurationError as err:
        sys.stderr.write(f"{err}\n")
        return EXIT_CONFIGURATION
    except Exception:
        logger.exception("The command '%s' failed.", " ".join(arguments))
        return EXIT_ANALYSIS
    return EXIT_OK
