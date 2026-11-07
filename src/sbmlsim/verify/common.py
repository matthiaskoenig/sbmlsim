"""What the subcommands of `verify` share: the output and the project."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from sbmlsim.console import console

if TYPE_CHECKING:
    from sbmlsim.project import Project

logger = logging.getLogger(__name__)

ROOT_HELP = (
    "directory put first on the python path, e.g. a checkout of the repository "
    "of the project; the project must be imported from below it"
)
PROJECT_HELP = "the project, module:attribute, e.g. pkdb_models.models.hctz:PROJECT"


class UsageError(Exception):
    """A subcommand cannot run as it was asked to, exit code 2."""


def report(text: object = "") -> None:
    """Write a line of the result of a subcommand to stdout.

    Unlike a log record it is the output of the command, unwrapped and
    unformatted, so that it can be redirected and compared.
    """
    console.print(str(text), soft_wrap=True, markup=False, highlight=False)


def add_project_arguments(
    parser: argparse.ArgumentParser, required: bool = True
) -> None:
    """Add `--project` and `--root` to the parser of a subcommand."""
    parser.add_argument("--project", required=required, help=PROJECT_HELP)
    parser.add_argument("--root", default=None, help=ROOT_HELP)


def outside_root(project: str, root: Path) -> str | None:
    """Check that the module of a project was imported from below `root`.

    `--root` is put first on the python path, but a package which is already
    imported (a second call in the same process) or installed in an environment
    in which the root does not take effect would silently be dumped from
    somewhere else.

    Args:
        project: the reference `module:attribute` of the project.
        root: the directory the module must be imported from.

    Returns:
        A message if the module comes from elsewhere, else `None`.
    """
    name = project.partition(":")[0]
    module = importlib.import_module(name)
    file = getattr(module, "__file__", None)
    if file is not None and Path(file).resolve().is_relative_to(root):
        return None
    return f"{name} is imported from {file}, not from below {root}"


def load_project_of(args: argparse.Namespace) -> Project:
    """Load the project of the arguments, `--project` below `--root`.

    Args:
        args: the parsed arguments with `project` and `root`.

    Returns:
        The project.

    Raises:
        UsageError: if the project cannot be loaded or is imported from
            outside of `--root`.
    """
    from sbmlsim.project import load_project

    root = None if args.root is None else Path(args.root).resolve()
    if root is not None:
        sys.path.insert(0, str(root))
    try:
        if root is not None and (message := outside_root(args.project, root)):
            raise UsageError(message)
        return load_project(args.project)
    except (ValueError, ImportError) as err:
        raise UsageError(f"{args.project}: {err}") from err
