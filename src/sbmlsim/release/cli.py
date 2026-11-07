"""The command line of the release of a model repository, `python -m sbmlsim.release`.

    python -m sbmlsim.release build <project> --version X.Y.Z --out DIR
        [--repository PATH] [--framework-path PATH] [--workers N] [--resume]
    python -m sbmlsim.release publish <project> --version X.Y.Z --out DIR

A project is `module:attribute` of a `sbmlsim.project.Project` with its
release settings. `build` runs the kind `release` of the run layer into the
output (`sbmlsim.run.execute`, the same as `python -m sbmlsim.run release
<spec> --out DIR`): it clones the release repository into `DIR/repository`
and builds the release there without committing it, so it can be reviewed;
see `sbmlsim.run.release_kinds`. `publish` publishes the build of the output,
a local step which never runs as a kind: it checks that the build is still
the one of the source, commits, tags and pushes it, deploys the site, pushes
the Docker image and creates the GitHub release (`publish_release`); a
publishing which failed half way is finished by running it again.

The exit codes of `build` are the ones of `python -m sbmlsim.run`: 0 done, 2 a
bad command line, 3 the build failed (its traceback is logged) and 4 a
configuration error, e.g. a refused release, its message on stderr. The exit
codes of `publish`: 0 published, 1 refused or failed, its message on stderr, 2
a bad command line and 4 a project which is not loaded.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from sbmlsim.release.source import Runner, run
from sbmlsim.run.cli import EXIT_ANALYSIS, EXIT_CONFIGURATION, EXIT_OK, EXIT_USAGE
from sbmlsim.run.kinds import ConfigurationError

if TYPE_CHECKING:
    from sbmlsim.project import Project

__all__ = ["EXIT_REFUSED", "main"]

logger = logging.getLogger(__name__)

#: the exit code of a publishing which is refused or fails
EXIT_REFUSED = 1


def _positive(value: str) -> int:
    """Parse a positive number of workers."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is no number") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"'{value}' is not positive")
    return number


def _parser() -> argparse.ArgumentParser:
    """Get the parser of the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m sbmlsim.release",
        description="Build the release of a model repository into an output and "
        "publish it.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build",
        help="build the release into DIR/repository, see the kind release",
    )
    publish = commands.add_parser(
        "publish",
        help="commit, tag and push the build of DIR, deploy the site, push the "
        "Docker image and create the GitHub release",
    )
    for command in (build, publish):
        command.add_argument("project", help="the project, module:attribute")
        command.add_argument(
            "--version", required=True, help="the version of the release, X.Y.Z"
        )
        command.add_argument(
            "--out", required=True, type=Path, help="the output of the build"
        )
    build.add_argument(
        "--repository",
        type=Path,
        metavar="PATH",
        help="a checkout of the release repository which is cloned instead of the "
        "repository on GitHub; only read",
    )
    build.add_argument(
        "--framework-path",
        type=Path,
        metavar="PATH",
        help="a local checkout of the framework instead of the released one; "
        "such a build is not published",
    )
    build.add_argument(
        "--workers",
        type=_positive,
        help="the number of workers of the simulations, the CPUs of the quota without",
    )
    build.add_argument(
        "--resume",
        action="store_true",
        help="build again into the output of an earlier build of the same spec",
    )
    return parser


def _build(args: argparse.Namespace, runner: Runner) -> int:
    """Run the kind `release` into the output."""
    from sbmlsim.release.workflow import execute_release

    try:
        summary = execute_release(
            args.project,
            args.version,
            args.out,
            repository=args.repository,
            framework_path=args.framework_path,
            workers=args.workers,
            resume=args.resume,
            runner=runner,
        )
    except ConfigurationError as err:
        sys.stderr.write(f"{err}\n")
        return EXIT_CONFIGURATION
    except Exception:
        logger.exception("The build of the release %s failed.", args.version)
        return EXIT_ANALYSIS
    repository = args.out / summary["repository"]
    sys.stdout.write(f"{repository}\nreview, then publish it\n")
    return EXIT_OK


def _load(reference: str) -> Project:
    """Load a project.

    Raises:
        ConfigurationError: for a project which is not loaded.
    """
    from sbmlsim.project import load_project

    try:
        return load_project(reference)
    except (ImportError, ValueError) as err:
        raise ConfigurationError(
            f"The project '{reference}' is not loaded: {err}"
        ) from err


def _publish(args: argparse.Namespace, runner: Runner) -> int:
    """Publish the build of the output."""
    from sbmlsim.release.source import ReleaseError
    from sbmlsim.release.workflow import publish_release

    try:
        project = _load(args.project)
    except ConfigurationError as err:
        sys.stderr.write(f"{err}\n")
        return EXIT_CONFIGURATION
    try:
        publish_release(project, args.version, args.out, runner=runner)
    except ReleaseError as err:
        sys.stderr.write(f"python -m sbmlsim.release publish: error: {err}\n")
        return EXIT_REFUSED
    sys.stdout.write(f"Released {args.version} from {args.out}.\n")
    return EXIT_OK


def main(argv: Sequence[str] | None = None, *, runner: Runner = run) -> int:
    """Run the command line.

    Args:
        argv: the arguments without the program, `sys.argv[1:]` without.
        runner: runs git, uv, the analyses of the build, docker and gh.

    Returns:
        The exit code, see the module.
    """
    try:
        args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else EXIT_USAGE
    if args.command == "build":
        return _build(args, runner)
    return _publish(args, runner)
