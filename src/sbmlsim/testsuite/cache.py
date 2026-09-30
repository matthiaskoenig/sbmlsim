"""The download and the cache of a test suite.

A test suite is an archive which is downloaded once and unpacked into the
user cache, i.e. `XDG_CACHE_HOME` or `~/.cache`, under `sbmlsim/`. An
environment variable points at the cases when they live elsewhere, e.g. on a
machine without a network. The archive is unpacked next to its target and
moved into place, so an interrupted download does not leave a directory which
looks like a cached suite.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

#: suffix of the staging directory of a fetch
STAGING_SUFFIX = ".incomplete"

#: age in seconds after which the staging directory of a fetch is stale, i.e.
#: left behind by a fetch which was killed. A fetch takes minutes
STALE_AFTER = 6 * 3600.0


def cache_root() -> Path:
    """Get the directory the test suites are cached in.

    Returns:
        `sbmlsim` in the user cache, i.e. in `XDG_CACHE_HOME` or `~/.cache`.
    """
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "sbmlsim"


def cache_path(variable: str, *parts: str) -> Path:
    """Get the directory a test suite is unpacked into.

    Args:
        variable: the environment variable which overrides the directory.
        *parts: the directories below the cache of `sbmlsim`, e.g. the name
            of the suite and its version.

    Returns:
        The value of the environment variable when it is set, the directory
        in the user cache otherwise.
    """
    override = os.environ.get(variable)
    if override:
        return Path(override)
    return cache_root().joinpath(*parts)


def fetch(url: str, path: Path, select: Callable[[Path], Path]) -> Path:
    """Download an archive and move a directory of it into place.

    Every fetch unpacks into a staging directory of its own next to `path`,
    so two processes which fetch one target do not share one. A target which
    is in place when the cases are moved there is the result of another
    fetch, and is used. The members of the archive are unpacked below the
    staging directory, a member with `..` or an absolute path does not leave
    it. The suite which calls `fetch` logs what it downloads, `fetch` logs the
    URL at the level `DEBUG`.

    Args:
        url: the zip archive.
        path: the directory which holds the cases afterwards.
        select: gets the directory the archive was unpacked into and returns
            the directory of it which becomes `path`.

    Returns:
        `path`.

    Raises:
        OSError: if the archive cannot be downloaded or unpacked, if `select`
            does not find the cases, or if they cannot be moved into place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    remove_stale(path)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{path.name}.", suffix=STAGING_SUFFIX, dir=path.parent
        )
    )
    archive = staging / "archive.zip"
    try:
        logger.debug("Downloading '%s' into '%s'", url, staging)
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
        archive.unlink()
        selected = select(staging)
        try:
            selected.replace(path)
        except OSError:
            if not path.is_dir():
                raise
            logger.debug("'%s' was fetched by another process", path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return path


def remove_stale(path: Path, stale_after: float = STALE_AFTER) -> list[Path]:
    """Remove the staging directories a killed fetch of a target left behind.

    A fetch which is running has a staging directory as well, which is why
    only a directory older than `stale_after` is removed.

    Args:
        path: the target of the fetch.
        stale_after: age in seconds after which a staging directory is stale.

    Returns:
        The directories which were removed.
    """
    removed: list[Path] = []
    now = time.time()
    for staging in path.parent.glob(f".{path.name}.*{STAGING_SUFFIX}"):
        if not staging.is_dir():
            continue
        try:
            age = now - staging.stat().st_mtime
        except OSError:
            continue
        if age < stale_after:
            continue
        logger.info("Removing the stale staging directory '%s'", staging)
        shutil.rmtree(staging, ignore_errors=True)
        removed.append(staging)
    return removed
