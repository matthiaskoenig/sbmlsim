"""Provenance of a fit: what it ran with, so that it can be reproduced.

A fit stores the arguments it was started with, the seeds of its runs, the
versions of the simulation stack, the revision of the repository of its model
and the checksums of its models and data (sbmlsim #334). The provenance is
written with every run and with the result and read back by the report.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any

#: the packages whose versions a fit records
PACKAGES: tuple[str, ...] = ("sbmlsim", "libroadrunner", "scipy", "numpy")

#: directories below a data path which hold no data: caches of python, of
#: notebooks and of version control
IGNORED_DIRECTORIES: frozenset[str] = frozenset(
    {"__pycache__", ".ipynb_checkpoints", ".git", ".svn", ".hg"}
)

#: files below a data path which hold no data: the code of a data package,
#: e.g. the script which collects the data, and the files of the system and of
#: version control
IGNORED_SUFFIXES: frozenset[str] = frozenset({".py", ".pyc", ".pyo"})
IGNORED_NAMES: frozenset[str] = frozenset(
    {".DS_Store", "Thumbs.db", ".gitignore", ".gitattributes", ".gitkeep"}
)


@dataclass(frozen=True)
class GitInfo:
    """Revision of the git repository which contains the model of a fit.

    Attributes:
        root: top level directory of the repository.
        sha: commit of the working tree.
        dirty: the working tree has uncommitted changes.
    """

    root: str
    sha: str
    dirty: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary."""
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> GitInfo:
        """Create from a dictionary of `to_dict`."""
        return GitInfo(root=str(d["root"]), sha=str(d["sha"]), dirty=bool(d["dirty"]))


@dataclass(frozen=True)
class FitProvenance:
    """Provenance of a fit.

    Attributes:
        created: date and time the fit started, ISO 8601 in UTC.
        arguments: the arguments the fit was started with.
        seed: seed of the fit, `None` for a fit which is not reproducible.
        run_seeds: seed of every run, `None` for a run which needs none.
        versions: versions of python and of the simulation stack.
        git: revision of the repository of the model, `None` outside of one.
        files: SHA-256 of the models and the data by path.
        start_values: start value of every fit parameter.
        fixed_values: values of the model which the caller keeps fixed;
            recorded, not applied.
    """

    created: str
    arguments: dict[str, Any]
    seed: int | None
    run_seeds: list[int | None]
    versions: dict[str, str]
    git: GitInfo | None
    files: dict[str, str] = field(default_factory=dict)
    start_values: dict[str, float | None] = field(default_factory=dict)
    fixed_values: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON types."""
        d = asdict(self)
        d["git"] = self.git.to_dict() if self.git else None
        return d

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> FitProvenance:
        """Create from a dictionary of `to_dict`."""
        git = d.get("git")
        return FitProvenance(
            created=str(d["created"]),
            arguments=dict(d.get("arguments", {})),
            seed=d.get("seed"),
            run_seeds=list(d.get("run_seeds", [])),
            versions=dict(d.get("versions", {})),
            git=GitInfo.from_dict(git) if git else None,
            files=dict(d.get("files", {})),
            start_values=dict(d.get("start_values", {})),
            fixed_values=dict(d.get("fixed_values", {})),
        )


def collect_versions() -> dict[str, str]:
    """Get the versions of python and of the simulation stack.

    Returns:
        The version by package, "not installed" for a package which is missing.
    """
    versions = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def git_info(path: Path) -> GitInfo | None:
    """Get the revision of the git repository which contains a path.

    Args:
        path: a file or directory.

    Returns:
        The revision, `None` if the path is in no repository or git is not
        installed.
    """
    directory = path if path.is_dir() else path.parent

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        root = run("rev-parse", "--show-toplevel")
        sha = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return None
    return GitInfo(root=root, sha=sha, dirty=dirty)


def data_files(roots: Iterable[Path]) -> list[Path]:
    """Get the data files below data paths.

    The data of a study are hidden files in many repositories (the
    `.<study>_<figure>.tsv` of PK-DB), so a hidden file counts. Left out are
    the files of `IGNORED_DIRECTORIES`, python code (`IGNORED_SUFFIXES`) and
    the files of the system and of version control (`IGNORED_NAMES`).

    Args:
        roots: the data paths, directories or files; a path which does not
            exist is left out.

    Returns:
        The sorted resolved paths of the data files.
    """
    paths: set[Path] = set()
    for root in roots:
        if root.is_file():
            paths.add(root.resolve())
            continue
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if IGNORED_DIRECTORIES.intersection(relative.parts[:-1]):
                continue
            if path.suffix in IGNORED_SUFFIXES or path.name in IGNORED_NAMES:
                continue
            paths.add(path.resolve())
    return sorted(paths)


def checksum_key(path: PurePath, root: PurePath | None) -> str:
    """Get the key of the checksum of a file, the same on every platform.

    Args:
        path: the resolved path of the file.
        root: the resolved root the key is relative to; `None` for no root.

    Returns:
        The path relative to `root` where possible, the absolute path
        otherwise, with `/` as separator.
    """
    if root is not None and path.is_relative_to(root):
        return path.relative_to(root).as_posix()
    return path.as_posix()


def file_checksums(paths: Iterable[Path], root: Path | None = None) -> dict[str, str]:
    """Get the SHA-256 of files.

    Args:
        paths: the files; a path which is no file is left out.
        root: the keys are relative to it where possible, absolute otherwise.

    Returns:
        The hex digest by path (`checksum_key`, `/` as separator on every
        platform), sorted by path.
    """
    sums: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        key = checksum_key(path.resolve(), root.resolve() if root else None)
        sums[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(sums.items()))


def create_provenance(
    *,
    arguments: Mapping[str, Any],
    seed: int | None,
    run_seeds: Sequence[int | None],
    base_path: Path,
    files: Iterable[Path],
    start_values: Mapping[str, float | None],
    fixed_values: Mapping[str, float] | None = None,
) -> FitProvenance:
    """Create the provenance of a fit before it starts.

    Args:
        arguments: the arguments of the fit, JSON types.
        seed: seed of the fit.
        run_seeds: seed of every run.
        base_path: directory of the experiments, it locates the repository.
        files: the models and the data of the problem.
        start_values: start value of every fit parameter.
        fixed_values: values of the model which the caller keeps fixed;
            recorded, not applied.

    Returns:
        The provenance.
    """
    git = git_info(base_path)
    root = Path(git.root) if git else base_path
    return FitProvenance(
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        arguments=dict(arguments),
        seed=seed,
        run_seeds=list(run_seeds),
        versions=collect_versions(),
        git=git,
        files=file_checksums(files, root=root),
        start_values=dict(start_values),
        fixed_values=dict(fixed_values or {}),
    )
