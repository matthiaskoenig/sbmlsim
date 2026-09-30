"""The download and the cache of a test suite.

A test suite is an archive which is downloaded once and unpacked into the
user cache, i.e. `XDG_CACHE_HOME` or `~/.cache`, under `sbmlsim/`. An
environment variable points at the cases when they live elsewhere, e.g. on a
machine without a network. The archive is unpacked next to its target and
moved into place, so an interrupted download does not leave a directory which
looks like a cached suite.

A fetch holds a lock on a file in its staging directory while it runs, which
the operating system releases when the process ends, also when it is killed.
A staging directory whose lock nobody holds is what a killed fetch left
behind, the next fetch or load of the target removes it. Without `fcntl`,
i.e. on Windows, the age of a staging directory decides.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

if sys.platform != "win32":
    import fcntl

logger = logging.getLogger(__name__)

#: suffix of the staging directory of a fetch
STAGING_SUFFIX = ".incomplete"

#: the file in the staging directory a running fetch holds a lock on
LOCK_NAME = ".fetch.lock"

#: age in seconds after which a staging directory without a lock file is
#: stale, i.e. left behind by a fetch which was killed. A fetch takes minutes.
#: A directory with a lock file is stale as soon as nobody holds its lock
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


def is_overridden(variable: str) -> bool:
    """Check whether an environment variable points at the cases.

    The directory of an override is not in the cache: nothing next to it
    was left by a fetch, see `remove_stale`.

    Args:
        variable: the environment variable which overrides the directory.

    Returns:
        Whether the variable is set.
    """
    return bool(os.environ.get(variable))


def fetch(url: str, path: Path, select: Callable[[Path], Path]) -> Path:
    """Download an archive and move a directory of it into place.

    Every fetch unpacks into a staging directory of its own next to `path`,
    so two processes which fetch one target do not share one. The fetch
    holds the lock of the staging directory until it is removed, and removes
    the staging directories of the target nobody holds, see `remove_stale`.
    A target which is in place when the cases are moved there is the result
    of another fetch, and is used. The members of the archive are unpacked below the
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
    # the archive is unpacked below the lock file, which does not move with
    # the directory `select` chooses
    unpacked = staging / "unpacked"
    lock: int | None = None
    try:
        # a lock which fails leaves no staging directory without a lock file
        lock = _lock(staging)
        logger.debug("Downloading '%s' into '%s'", url, staging)
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(unpacked)
        archive.unlink()
        selected = select(unpacked)
        try:
            selected.replace(path)
        except OSError:
            if not path.is_dir():
                raise
            logger.debug("'%s' was fetched by another process", path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if lock is not None:
            os.close(lock)
    return path


def _lock(staging: Path) -> int | None:
    """Create the lock file of a staging directory and hold its lock.

    The lock file is locked under another name and renamed, so a fetch which
    looks for stale directories never finds the lock file of a running fetch
    unlocked.

    Args:
        staging: the staging directory of a fetch.

    Returns:
        The file descriptor which holds the lock, `None` without `fcntl`.
    """
    if sys.platform == "win32":
        return None
    pending = staging / f"{LOCK_NAME}.pending"
    fd = os.open(pending, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        pending.replace(staging / LOCK_NAME)
    except OSError:
        os.close(fd)
        raise
    return fd


def _try_lock(lock_file: Path) -> int | None:
    """Take the lock of a staging directory if nobody holds it.

    Args:
        lock_file: the lock file of the staging directory.

    Returns:
        The file descriptor which holds the lock, `None` if a running fetch
        holds it or the file is gone.
    """
    if sys.platform == "win32":
        return None
    try:
        fd = os.open(lock_file, os.O_RDWR)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def remove_stale(path: Path, stale_after: float = STALE_AFTER) -> list[Path]:
    """Remove the staging directories a killed fetch of a target left behind.

    A fetch which is running has a staging directory as well and holds the
    lock of its lock file. A staging directory with a lock file is removed
    when its lock can be taken, i.e. the fetch which created it has ended. A
    staging directory without a lock file, or any on a platform without
    `fcntl`, is removed when it is older than `stale_after`.

    Args:
        path: the target of the fetch.
        stale_after: age in seconds after which a staging directory without a
            lock file is stale.

    Returns:
        The directories which were removed.
    """
    if not path.parent.is_dir():
        return []
    pattern = re.compile(rf"\.{re.escape(path.name)}\.[^.]+{re.escape(STAGING_SUFFIX)}")
    removed: list[Path] = []
    now = time.time()
    for staging in sorted(path.parent.iterdir()):
        if not pattern.fullmatch(staging.name) or not staging.is_dir():
            continue
        lock_file = staging / LOCK_NAME
        if sys.platform != "win32" and lock_file.is_file():
            lock = _try_lock(lock_file)
            if lock is None:
                continue
            try:
                _remove(staging)
            finally:
                os.close(lock)
        else:
            try:
                age = now - staging.stat().st_mtime
            except OSError:
                continue
            if age < stale_after:
                continue
            _remove(staging)
        removed.append(staging)
    return removed


def _remove(staging: Path) -> None:
    """Remove a stale staging directory.

    Args:
        staging: the staging directory.
    """
    logger.info("Removing the stale staging directory '%s'", staging)
    shutil.rmtree(staging, ignore_errors=True)
