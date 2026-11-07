"""The source repository and the release repository of a model.

A release exports the tracked files of a model package from its source
repository (e.g. a monorepo of models) into a checkout of the release
repository. The functions here check both repositories and write the package;
every refusal is a `ReleaseError`. Git is called through a `Runner`, so the
tests can replace it.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol

from sbmlsim.release.settings import ReleaseSettings

if TYPE_CHECKING:
    from sbmlsim.project import Project

__all__ = [
    "ReleaseError",
    "ReleaseRepo",
    "Runner",
    "SourceRepo",
    "check_version",
    "clone_url",
    "export_package",
    "locked_version",
    "package_dirs",
    "parse_version",
    "release_repo",
    "run",
    "source_framework",
    "source_repo",
    "stage",
]

logger = logging.getLogger(__name__)

_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_GITHUB_URL = re.compile(
    r"^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
    r"(?P<repository>[^/]+/[^/]+?)(?:\.git)?/?$"
)
_STDERR_TAIL = 20


class ReleaseError(Exception):
    """A refusal of the release command, the message is for the user."""


class Runner(Protocol):
    """Runs a command in a folder, see `run`."""

    def __call__(self, args: Sequence[str], cwd: Path, *, stream: bool = False) -> str:
        """Run the command `args` in `cwd`; `stream` shows its output."""
        ...


def stage(text: str) -> None:
    """Log the start of a stage of the release."""
    logger.info("%s", text)


def run(args: Sequence[str], cwd: Path, *, stream: bool = False) -> str:
    """Run a command and return its stdout.

    A command whose output is parsed (git, gh, docker queries) is captured. A long
    step of the build or the publishing (`uv`, the analyses of the model, `docker
    build`) is streamed: its output goes to the terminal as it runs. The command
    does not inherit `VIRTUAL_ENV`: the environment the release runs in is the
    one of the source repository, uv in the release repository uses its own.

    Args:
        args: the command and its arguments.
        cwd: the working directory.
        stream: show the output of the command instead of capturing it.

    Returns:
        The standard output of the command, empty if it is streamed.

    Raises:
        ReleaseError: if the command fails or does not exist; the message has the
            command and the tail of its standard error (a streamed command has
            shown its standard error already).
    """
    command = " ".join(str(a) for a in args)
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    if stream:
        try:
            subprocess.run([str(a) for a in args], cwd=cwd, env=env, check=True)
        except subprocess.CalledProcessError as err:
            raise ReleaseError(
                f"'{command}' failed in {cwd} with exit code {err.returncode}, "
                "see its output above"
            ) from err
        except OSError as err:
            raise ReleaseError(f"'{command}' could not run in {cwd}: {err}") from err
        return ""
    try:
        result = subprocess.run(
            [str(a) for a in args],
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as err:
        tail = "\n".join((err.stderr or "").strip().splitlines()[-_STDERR_TAIL:])
        raise ReleaseError(f"'{command}' failed in {cwd}:\n{tail}") from err
    except OSError as err:
        raise ReleaseError(f"'{command}' could not run in {cwd}: {err}") from err
    return result.stdout


@dataclass(frozen=True)
class SourceRepo:
    """The clean source repository of a model.

    Attributes:
        root: the top level folder of the git repository.
        commit: the commit the release is built from.
    """

    root: Path
    commit: str


@dataclass(frozen=True)
class ReleaseRepo:
    """The checkout of the release repository.

    Attributes:
        path: the folder of the checkout.
        last_tag: the highest released version, `None` without a release.
        ahead: the number of local commits which `origin/main` lacks.
    """

    path: Path
    last_tag: str | None
    ahead: int = 0


def source_repo(project: Project, runner: Runner = run) -> SourceRepo:
    """Find the source repository of a project and check it is clean.

    Args:
        project: the project, its `root` lies in the repository.
        runner: runs the git commands.

    Returns:
        The root and the current commit of the repository.

    Raises:
        ReleaseError: if there are uncommitted or untracked files.
    """
    root = Path(runner(["git", "rev-parse", "--show-toplevel"], project.root).strip())
    if runner(["git", "status", "--porcelain"], root).strip():
        raise ReleaseError(f"commit the changes of {root} first")
    commit = runner(["git", "rev-parse", "HEAD"], root).strip()
    return SourceRepo(root=root, commit=commit)


def clone_url(repository: str) -> str:
    """The URL the release repository is cloned from.

    Args:
        repository: the GitHub repository as `owner/name`.

    Returns:
        The https URL of the repository.
    """
    return f"https://github.com/{repository}.git"


def _is_repository(origin: str, repository: str) -> bool:
    if origin == clone_url(repository):
        return True
    match = _GITHUB_URL.match(origin)
    return bool(match) and match["repository"].lower() == repository.lower()


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse a version of a release.

    Args:
        version: the version `X.Y.Z`.

    Returns:
        The three numbers of the version.

    Raises:
        ReleaseError: if the version is not `X.Y.Z`.
    """
    match = _VERSION.match(version)
    if match is None:
        raise ReleaseError(f"'{version}' is not a version 'X.Y.Z'")
    return int(match[1]), int(match[2]), int(match[3])


def locked_version(package: Mapping[str, Any]) -> str:
    """Get the version of a package of `uv.lock`.

    uv locks a local package (a path, directory or editable source) whose version
    is dynamic without a version, e.g. a checkout of the framework given as
    `framework_path`; its version is then the kind and the location of its
    source, e.g. `editable /home/user/sbmlsim`.

    Args:
        package: a `[[package]]` table of `uv.lock`.

    Returns:
        The locked version or the description of the local source.
    """
    if "version" in package:
        return str(package["version"])
    source = package.get("source", {})
    return " ".join(f"{kind} {location}" for kind, location in source.items()) or "?"


def source_framework(source: SourceRepo, framework: str) -> str | None:
    """Get the version of the framework which the source repository locks.

    The release repository is locked to the library versions of the source (see
    `sbmlsim.release.project`), so this is the framework of the release,
    whichever version runs the command.

    Args:
        source: the source repository.
        framework: the name of the framework, see
            `sbmlsim.release.settings.ReleaseSettings.framework`.

    Returns:
        The locked version (see `locked_version`), `None` without `uv.lock` or
        without the framework in it.
    """
    lock_path = source.root / "uv.lock"
    if not lock_path.is_file():
        return None
    lock = tomllib.loads(lock_path.read_text("utf-8"))
    for package in lock.get("package", []):
        if package.get("name") == framework:
            return locked_version(package)
    return None


def _last_tag(tags: Sequence[str]) -> str | None:
    versions = [t for t in tags if _VERSION.match(t)]
    return max(versions, key=parse_version, default=None)


def _clone(
    settings: ReleaseSettings, path: Path, checkout: Path | None, runner: Runner
) -> None:
    """Clone the release repository into `path`.

    A checkout of the release repository is cloned with its history, tags and
    branches, and the origin of the clone is set to the origin of the checkout,
    so the clone fetches from and pushes to the release repository; the
    checkout itself is only read.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if checkout is None:
        runner(["git", "clone", clone_url(settings.repository), str(path)], path.parent)
        return
    if not checkout.is_dir():
        raise ReleaseError(
            f"The checkout of the release repository {checkout} does not exist."
        )
    origin = runner(["git", "remote", "get-url", "origin"], checkout).strip()
    runner(["git", "clone", "--no-hardlinks", str(checkout), str(path)], path.parent)
    runner(["git", "remote", "set-url", "origin", origin], path)


def release_repo(
    settings: ReleaseSettings,
    path: Path,
    runner: Runner = run,
    *,
    checkout: Path | None = None,
) -> ReleaseRepo:
    """Find or clone the release repository and check it is ready.

    Args:
        settings: the release settings of the model.
        path: the checkout the release is built in, cloned when it does not
            exist.
        runner: runs the git commands.
        checkout: a checkout of the release repository which a missing `path`
            is cloned from instead of the repository on GitHub, see `_clone`.

    Returns:
        The checkout, its highest released version and the number of its
        commits which are not pushed (only the release commit of a release
        which failed while publishing may be, see `sbmlsim.release.workflow`).

    An empty repository (a new repository on GitHub without a commit) has no
    release and no commit ahead.

    Raises:
        ReleaseError: if `origin` is another repository, the branch is not `main`
            or the branch is behind `origin/main`.
    """
    if not path.exists():
        _clone(settings, path, checkout, runner)
    origin = runner(["git", "remote", "get-url", "origin"], path).strip()
    if not _is_repository(origin, settings.repository):
        raise ReleaseError(
            f"{path} has origin {origin}, not the release repository "
            f"{settings.repository}"
        )
    # empty on a detached HEAD; the branch of an empty repository has no commit
    branch = runner(["git", "branch", "--show-current"], path).strip()
    if branch != "main":
        raise ReleaseError(f"{path} is on branch '{branch or 'HEAD'}', not 'main'")
    runner(["git", "fetch", "--tags", "origin"], path)
    local = _has_ref(path, "refs/heads/main", runner)
    remote = _has_ref(path, "refs/remotes/origin/main", runner)
    if remote and (not local or _count(path, "HEAD..origin/main", runner) > 0):
        raise ReleaseError(f"{path} is behind origin/main, pull it first")
    if not local:
        ahead = 0
    elif remote:
        ahead = _count(path, "origin/main..HEAD", runner)
    else:
        ahead = _count(path, "HEAD", runner)
    tags = runner(["git", "tag", "--list"], path).split()
    return ReleaseRepo(path=path, last_tag=_last_tag(tags), ahead=ahead)


def _has_ref(path: Path, ref: str, runner: Runner) -> bool:
    return bool(
        runner(["git", "for-each-ref", "--format=%(refname)", ref], path).strip()
    )


def _count(path: Path, revisions: str, runner: Runner) -> int:
    return int(runner(["git", "rev-list", "--count", revisions], path).strip() or "0")


def check_version(version: str, last_tag: str | None) -> None:
    """Check that a version is newer than the last release.

    Args:
        version: the new version `X.Y.Z`.
        last_tag: the last released version, `None` without a release.

    Raises:
        ReleaseError: if a version is malformed or `version` is not greater.
    """
    new = parse_version(version)
    if last_tag is not None and new <= parse_version(last_tag):
        raise ReleaseError(
            f"version {version} must be greater than the last release {last_tag}"
        )


def _excluded(relative: PurePosixPath, exclude: Sequence[str]) -> bool:
    for entry in exclude:
        entry_path = PurePosixPath(entry.strip("/"))
        if relative == entry_path or entry_path in relative.parents:
            return True
    return False


def package_dirs(project: Project) -> Path:
    """Get the folders of the package of a project, `pkg/models/hctz` of `pkg.models.hctz`.

    Raises:
        ReleaseError: for a project without a package, which is not released.
    """
    if not project.package:
        raise ReleaseError(
            f"The project '{project.name}' has no package, a release exports it."
        )
    return Path(*project.package.split("."))


def export_package(
    project: Project,
    settings: ReleaseSettings,
    source: SourceRepo,
    release_path: Path,
    runner: Runner = run,
) -> list[Path]:
    """Write the tracked files of the model package into the release repository.

    The files below `src/<package as folders>` are replaced. The parent folders
    of the package (and `src`) are namespace folders, so an `__init__.py` there is removed
    and none is created.

    Args:
        project: the project, `root` is the folder and `package` the dotted
            name of its package.
        settings: the release settings, `exclude` lists paths of the model folder
            which are not released (a folder excludes everything below it).
        source: the source repository.
        release_path: the checkout of the release repository.
        runner: runs the git commands.

    Returns:
        The written files, relative to `release_path`.

    Raises:
        ReleaseError: for a project without a package.
    """
    folders = package_dirs(project)
    listing = runner(["git", "ls-files", "-z", "--", "."], project.root)
    files = [
        PurePosixPath(name)
        for name in listing.split("\0")
        if name and not _excluded(PurePosixPath(name), settings.exclude)
    ]
    src = release_path / "src"
    target = src / folders
    if target.exists():
        shutil.rmtree(target)
    for parent in folders.parents:
        stale = src / parent / "__init__.py"
        if stale.exists():
            stale.unlink()
    written: list[Path] = []
    for relative in files:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project.root / relative, destination)
        written.append(destination.relative_to(release_path))
    return written
