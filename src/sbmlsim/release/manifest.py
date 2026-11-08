"""The manifest of a release and the removal of what the release did not build.

The build writes `release.json` (the version, the commit of the source, the
library versions and a hash of every generated file). The publishing finds a
build which is not committed yet and publishes it as it is when the manifest
still matches, so a reviewed build needs no new simulations. `prune`
makes the release repository hold only what the command generates.
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from sbmlsim.hashing import file_sha256
from sbmlsim.release.source import (
    ReleaseError,
    Runner,
    SourceRepo,
    locked_version,
    run,
)

__all__ = [
    "BUILD_MARKER",
    "LEGACY_FRAMEWORK",
    "MANIFEST_FILE",
    "MAX_FILE_SIZE",
    "Manifest",
    "changed_by_build",
    "check_file_sizes",
    "check_not_ignored",
    "finish_build",
    "matches",
    "prune",
    "start_build",
    "stop_build",
    "write_manifest",
]

MANIFEST_FILE = "release.json"
BUILD_MARKER = "sbmlsim_release_build"
"""The file in the git folder which marks a build which has not finished."""
LEGACY_FRAMEWORK = "pkpdsim"
"""The framework of a manifest of pkpdsim 0.19, which has the field `pkpdsim`."""
_LEGACY_LIBRARIES = ("pkpdsim", "sbmlsim", "sbmlutils")
_RELEASE_NOTES = "release-notes"
_SHOWN = 20
MAX_FILE_SIZE = 100 * 1024 * 1024
"""The largest file a release may hold: GitHub rejects a push of a larger one."""


@dataclass(frozen=True)
class Manifest:
    """What a build of the release repository produced.

    Attributes:
        version: the released version.
        source_commit: the commit of the source repository.
        framework: the name of the framework of the release, see
            `sbmlsim.release.settings.ReleaseSettings.framework`.
        framework_version: the version of the framework in the lock of the
            release.
        libraries: the versions of the framework and the core libraries in the
            lock of the release by their name.
        files: the sha256 of every generated file by its POSIX path relative to
            the release repository.
    """

    version: str
    source_commit: str
    framework: str
    framework_version: str
    libraries: Mapping[str, str] = field(default_factory=dict)
    files: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Get the manifest as JSON data."""
        return {
            "version": self.version,
            "source_commit": self.source_commit,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "libraries": dict(self.libraries),
            "files": dict(self.files),
        }

    def to_json(self) -> str:
        """Serialise the manifest.

        Returns:
            The JSON text with sorted keys and a final newline.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> Manifest:
        """Get a manifest from its JSON data, also the one of pkpdsim 0.19.

        A manifest of pkpdsim 0.19 has the versions of pkpdsim, sbmlsim and
        sbmlutils as fields of their own; its framework is pkpdsim
        (`LEGACY_FRAMEWORK`).

        Args:
            data: the JSON data of `release.json`.

        Returns:
            The manifest.

        Raises:
            TypeError: for data which is no manifest.
            KeyError: for a manifest of pkpdsim 0.19 without a version of a
                library.
        """
        if "framework" in data:
            return Manifest(**data)
        legacy = dict(data)
        libraries = {name: str(legacy.pop(name)) for name in _LEGACY_LIBRARIES}
        return Manifest(
            **legacy,
            framework=LEGACY_FRAMEWORK,
            framework_version=libraries[LEGACY_FRAMEWORK],
            libraries=libraries,
        )

    @staticmethod
    def read(path: Path) -> Manifest | None:
        """Read a manifest, also the one of pkpdsim 0.19 (see `from_dict`).

        Args:
            path: the `release.json` file.

        Returns:
            The manifest, `None` if the file is missing.

        Raises:
            ReleaseError: if the file is not a manifest.
        """
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise TypeError("it is no JSON object")
            return Manifest.from_dict(data)
        except (ValueError, TypeError, KeyError) as err:
            raise ReleaseError(f"'{path}' is not a release manifest: {err}") from err


def _relative(release_path: Path, path: Path) -> str:
    """Normalise a path of a stage to a POSIX path relative to the release."""
    if path.is_absolute():
        try:
            path = path.relative_to(release_path)
        except ValueError as err:
            raise ReleaseError(
                f"'{path}' is not inside the release repository {release_path}."
            ) from err
    return PurePosixPath(*path.parts).as_posix()


def _generated(release_path: Path, written: Iterable[Path]) -> set[str]:
    """The relative POSIX paths of all files written; folders are expanded."""
    files: set[str] = set()
    for path in written:
        relative = _relative(release_path, path)
        absolute = release_path / relative
        if not absolute.exists():
            raise ReleaseError(f"The stage wrote '{relative}', but it does not exist.")
        if absolute.is_dir():
            for root, _, names in os.walk(absolute):
                for name in names:
                    files.add(_relative(release_path, Path(root) / name))
        else:
            files.add(relative)
    return files


def _library_versions(release_path: Path) -> dict[str, str]:
    lock_path = release_path / "uv.lock"
    if not lock_path.is_file():
        raise ReleaseError(f"The release has no 'uv.lock' at {lock_path}.")
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    return {p["name"]: locked_version(p) for p in lock.get("package", [])}


def write_manifest(
    release_path: Path,
    version: str,
    source: SourceRepo,
    written: Iterable[Path],
    *,
    framework: str,
    core_libraries: Sequence[str] = (),
) -> Manifest:
    """Write `release.json` with a hash of every generated file.

    Args:
        release_path: the release repository.
        version: the released version.
        source: the source repository the release is built from.
        written: the files (or folders) the stages wrote, absolute or relative to
            the release repository.
        framework: the framework of the release.
        core_libraries: the core libraries of the release.

    Returns:
        The manifest which was written.

    Raises:
        ReleaseError: if `uv.lock` lacks the framework or a core library.
    """
    names = list(dict.fromkeys((framework, *core_libraries)))
    locked = _library_versions(release_path)
    missing = [n for n in names if n not in locked]
    if missing:
        raise ReleaseError(f"'uv.lock' of the release lacks {', '.join(missing)}.")
    files = {
        relative: file_sha256(release_path / relative)
        for relative in sorted(_generated(release_path, written))
        if relative != MANIFEST_FILE
    }
    manifest = Manifest(
        version=version,
        source_commit=source.commit,
        framework=framework,
        framework_version=locked[framework],
        libraries={name: locked[name] for name in names},
        files=files,
    )
    (release_path / MANIFEST_FILE).write_text(
        manifest.to_json(), encoding="utf-8", newline="\n"
    )
    return manifest


def _status_paths(release_path: Path, runner: Runner) -> set[str]:
    out = runner(
        ["git", "status", "--porcelain", "-z", "--untracked-files=all"], release_path
    )
    paths: set[str] = set()
    entries = iter(out.split("\0"))
    for entry in entries:
        if len(entry) < 4:
            continue
        if "D" not in entry[:2]:  # a deleted file is not an addition
            paths.add(entry[3:])
        if entry[0] in "RC":  # the source of a rename or copy follows
            paths.add(next(entries, ""))
    return paths


def matches(
    release_path: Path,
    version: str,
    source_commit: str,
    framework: str,
    framework_version: str | None,
    runner: Runner = run,
) -> bool:
    """Tell whether the release repository holds the build of this release.

    Args:
        release_path: the release repository.
        version: the version to release.
        source_commit: the commit of the source repository.
        framework: the name of the framework of the release.
        framework_version: the version of the framework which the source locks
            (see `sbmlsim.release.source.source_framework`), `None` matches no
            build.
        runner: runs git.

    Returns:
        `True` if the manifest exists, version, commit, framework and its
        version are equal, every listed file hashes the same and git status
        lists no file outside the manifest (besides `release.json`).
    """
    manifest = Manifest.read(release_path / MANIFEST_FILE)
    if manifest is None:
        return False
    if (
        manifest.version,
        manifest.source_commit,
        manifest.framework,
        manifest.framework_version,
    ) != (version, source_commit, framework, framework_version):
        return False
    for relative, digest in manifest.files.items():
        path = release_path / relative
        if not path.is_file() or file_sha256(path) != digest:
            return False
    allowed = {*manifest.files, MANIFEST_FILE}
    return _status_paths(release_path, runner) <= allowed


def _marker(release_path: Path, runner: Runner) -> Path:
    # the git folder of the checkout, `.git` is a file in a worktree or submodule
    git_dir = runner(["git", "rev-parse", "--absolute-git-dir"], release_path)
    return Path(git_dir.strip()) / BUILD_MARKER


def _read_marker(marker: Path) -> dict[str, object] | None:
    if not marker.is_file():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except ValueError:
        return {}  # a marker of an unknown format, the build left no paths
    return data if isinstance(data, dict) else {}


def start_build(release_path: Path, version: str, runner: Runner = run) -> None:
    """Mark the start of a build, before the first file is written.

    The marker lies in the git folder, so it is never committed or pruned.

    Args:
        release_path: the release repository.
        version: the version which is built.
        runner: runs git.
    """
    _marker(release_path, runner).write_text(
        json.dumps({"version": version}) + "\n", encoding="utf-8", newline="\n"
    )


def stop_build(release_path: Path, runner: Runner = run) -> None:
    """Record the paths a build changed before it failed or was interrupted.

    The marker keeps the paths which git status lists, so the next run tells
    the changes of the build from a file added by hand (`changed_by_build`). A
    build which could not record them (it was killed) leaves a marker without
    paths.

    Args:
        release_path: the release repository.
        runner: runs git.
    """
    marker = _marker(release_path, runner)
    data = _read_marker(marker)
    if data is None:
        return
    data["changed"] = sorted(_status_paths(release_path, runner))
    marker.write_text(json.dumps(data) + "\n", encoding="utf-8", newline="\n")


def finish_build(release_path: Path, runner: Runner = run) -> None:
    """Remove the marker of a build, after the manifest is written.

    Args:
        release_path: the release repository.
        runner: runs git.
    """
    _marker(release_path, runner).unlink(missing_ok=True)


def changed_by_build(release_path: Path, version: str, runner: Runner = run) -> bool:
    """Tell whether the release repository is clean or changed only by a build.

    A build of the same version may be rebuilt, the files may differ from their
    hashes (e.g. after a hand edit during the review). Deleted files are no
    change, a build deletes what it did not generate. A build which failed or
    was interrupted left its marker with the paths it had changed
    (`stop_build`), these are rebuilt as well; a build which was killed left
    no paths, then every change is taken as its change.

    Args:
        release_path: the release repository.
        version: the version to release.
        runner: runs git.

    Returns:
        `True` if git status lists no file, or lists only files of the manifest
        of `version`, `release.json` and the paths of an unfinished build.
    """
    changed = _status_paths(release_path, runner)
    if not changed:
        return True
    allowed = {MANIFEST_FILE}
    manifest = Manifest.read(release_path / MANIFEST_FILE)
    if manifest is not None and manifest.version == version:
        allowed |= set(manifest.files)
    marker = _read_marker(_marker(release_path, runner))
    if marker is not None:
        paths = marker.get("changed")
        if not isinstance(paths, list):
            return True
        allowed |= {str(p) for p in paths}
    elif manifest is None or manifest.version != version:
        return False
    return changed <= allowed


def _ignored(release_path: Path, runner: Runner) -> tuple[set[str], set[str]]:
    """The untracked ignored files and folders (the latter with a trailing /)."""
    out = runner(
        [
            "git",
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--directory",
            "-z",
        ],
        release_path,
    )
    entries = {e for e in out.split("\0") if e}
    files = {e for e in entries if not e.endswith("/")}
    return files, entries - files


def check_not_ignored(
    release_path: Path, written: Iterable[Path], runner: Runner = run
) -> None:
    """Refuse a build which wrote files that git ignores.

    The commit of the release (`git add -A`) leaves an ignored file out, e.g. a
    tracked `results/` folder of a model package below an ignore pattern of the
    release repository.

    Args:
        release_path: the release repository.
        written: the files (or folders) the stages wrote.
        runner: runs git.

    Raises:
        ReleaseError: if git ignores a written file; the message names them.
    """
    ignored_files, ignored_dirs = _ignored(release_path, runner)
    ignored = sorted(
        relative
        for relative in _generated(release_path, written)
        if relative in ignored_files
        or any(relative.startswith(d) for d in ignored_dirs)
    )
    if ignored:
        shown = ignored[:_SHOWN]
        if len(ignored) > _SHOWN:
            shown.append(f"... and {len(ignored) - _SHOWN} more")
        raise ReleaseError(
            "git ignores files of the release, its commit would lack them; "
            "change the ignore pattern which matches them (git check-ignore -v):\n- "
            + "\n- ".join(shown)
        )


def check_file_sizes(
    release_path: Path, written: Iterable[Path], max_size: int = MAX_FILE_SIZE
) -> None:
    """Refuse a build which wrote a file above the size GitHub accepts.

    GitHub rejects a push with a file above 100 MB, after the commit and the
    tag of the release; e.g. the netCDF results of the simulations of a model
    which keeps its results, gigabytes for the steps of the integrator.

    Args:
        release_path: the release repository.
        written: the files (or folders) the stages wrote.
        max_size: the largest size of a file in bytes.

    Raises:
        ReleaseError: if a written file is larger; the message names them with
            their size.
    """
    sizes = {
        relative: (release_path / relative).stat().st_size
        for relative in _generated(release_path, written)
    }
    large = sorted(
        (relative for relative, size in sizes.items() if size > max_size),
        key=lambda relative: -sizes[relative],
    )
    if large:
        shown = [f"{relative} ({sizes[relative] / 1e6:.0f} MB)" for relative in large]
        if len(shown) > _SHOWN:
            shown = [*shown[:_SHOWN], f"... and {len(large) - _SHOWN} more"]
        raise ReleaseError(
            f"files of the release are larger than {max_size / 1e6:.0f} MB, "
            "which GitHub rejects; leave them out of the outputs of the model "
            "(e.g. the results of the simulations):\n- " + "\n- ".join(shown)
        )


def prune(
    release_path: Path, written: Iterable[Path], runner: Runner = run
) -> list[Path]:
    """Remove every file of the release repository the command did not generate.

    Kept are `.git`, ignored files (`.gitignore`), `release-notes/*.md`,
    `release.json` and `written`. Folders left empty are removed.

    Args:
        release_path: the release repository.
        written: the files (or folders) the stages wrote.
        runner: runs git.

    Returns:
        The removed files and folders (absolute).
    """
    keep = _generated(release_path, written) | {MANIFEST_FILE}
    ignored_files, ignored_dirs = _ignored(release_path, runner)
    removed: list[Path] = []
    for root, dirs, names in os.walk(release_path):
        here = Path(root)
        relative_root = here.relative_to(release_path)
        dirs[:] = [
            d
            for d in dirs
            if not (relative_root == Path() and d == ".git")
            and (relative_root / d).as_posix() + "/" not in ignored_dirs
        ]
        for name in names:
            relative = (relative_root / name).as_posix()
            if (
                relative in keep
                or relative == ".git"  # the git file of a worktree or submodule
                or relative in ignored_files
                or (relative_root.as_posix() == _RELEASE_NOTES and name.endswith(".md"))
            ):
                continue
            (here / name).unlink()
            removed.append(here / name)
    for root, _, _ in os.walk(release_path, topdown=False):
        here = Path(root)
        relative = here.relative_to(release_path)
        if relative == Path() or relative.parts[0] == ".git":
            continue
        prefix = relative.as_posix() + "/"
        if any(prefix.startswith(d) for d in ignored_dirs) or any(
            i.startswith(prefix) for i in ignored_files
        ):
            continue
        if not any(here.iterdir()):
            here.rmdir()
            removed.append(here)
    return removed
