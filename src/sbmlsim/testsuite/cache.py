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
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


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

    Args:
        url: the zip archive.
        path: the directory which holds the cases afterwards. It must not
            exist.
        select: gets the directory the archive was unpacked into and returns
            the directory of it which becomes `path`.

    Returns:
        `path`.

    Raises:
        OSError: if the archive cannot be downloaded or unpacked, or if
            `select` does not find the cases.
    """
    staging = path.parent / f".{path.name}.incomplete"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    archive = staging / "archive.zip"
    try:
        logger.info("Downloading '%s'", url)
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
        archive.unlink()
        selected = select(staging)
        path.parent.mkdir(parents=True, exist_ok=True)
        selected.replace(path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return path
