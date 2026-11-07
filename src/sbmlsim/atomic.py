"""Atomic writes of files and directories.

A file is written into a temporary file of its directory, which replaces it
once it is complete, so a reader sees the old or the new file and never a
partial one, and a file written twice, e.g. by a unit of a run which runs
again, keeps one complete content. The temporary file is created with the
mode `0o666` and the umask of the process, the mode of a file written with
`open` or `Path.write_text`, so a file written atomically is readable as any
other file of the directory. A directory is written the same way,
`atomic_directory`: a temporary directory next to it replaces it once it is
complete.

The module imports only the standard library.
"""

import logging
import os
import secrets
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

__all__ = ["atomic_directory", "atomic_path", "write_text_atomic"]

logger = logging.getLogger(__name__)

#: attempts to find a name for the temporary file which does not exist
_ATTEMPTS = 100


def _create_temporary(path: Path) -> Path:
    """Create an empty temporary file next to the path.

    The file is created exclusively with the mode `0o666`, which the umask of
    the process reduces, as for a file created with `open`.

    Args:
        path: the file the temporary file replaces later.

    Returns:
        The path of the temporary file, `.<name>.<random>.tmp`.

    Raises:
        FileExistsError: if no free name was found.
    """
    for _ in range(_ATTEMPTS):
        temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
        try:
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError:
            continue
        os.close(fd)
        return temporary
    raise FileExistsError(f"No free name for a temporary file of '{path}'.")


@contextmanager
def atomic_path(path: Path) -> Generator[Path]:
    """Write a file atomically through a temporary path.

    The caller writes the complete file to the yielded path, e.g. with
    `Path.write_text` or `xarray.Dataset.to_netcdf`, which may replace the
    empty file created there. On success the temporary file gets the mode of
    a new file of the process, `0o666` without the bits of the umask, and
    replaces `path`; on an exception it is removed and `path` is left as it
    was.

    Args:
        path: the file to write; its directory must exist.

    Yields:
        The temporary path in the directory of `path`.
    """
    path = Path(path)
    temporary = _create_temporary(path)
    try:
        # the mode `0o666` without the umask, read from the file created with
        # it rather than by setting the umask, which threads share
        mode = temporary.stat().st_mode & 0o777
        yield temporary
        # a writer may have replaced the file with one of another mode
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, text: str) -> None:
    """Write a text file atomically in utf-8, see `atomic_path`.

    Args:
        path: the file to write; its directory must exist.
        text: the content of the file.
    """
    with atomic_path(path) as temporary:
        temporary.write_text(text, encoding="utf-8")


@contextmanager
def atomic_directory(path: Path) -> Generator[Path]:
    """Write a directory atomically through a temporary directory.

    The caller writes the complete directory into the yielded one, which
    replaces `path` on success: an existing `path` is moved aside first and
    removed after the new one took its place (best effort: a directory which
    is not removed is logged and left next to it). On an exception the
    temporary directory is removed and `path` is left as it was, also when
    the new directory fails to take its place after the old one was moved.
    The directory is created with the mode `0o777` and the umask of the
    process, as a directory created with `mkdir`.

    Args:
        path: the directory to write; its parent must exist.

    Yields:
        The temporary directory next to `path`.

    Raises:
        FileExistsError: if no free name was found.
    """
    path = Path(path)
    for _ in range(_ATTEMPTS):
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        try:
            temporary.mkdir(mode=0o777)
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(f"No free name for a temporary directory of '{path}'.")
    try:
        yield temporary
        if path.exists():
            previous = temporary.with_name(f"{temporary.name}.old")
            path.rename(previous)
            try:
                temporary.rename(path)
            except BaseException:
                previous.rename(path)
                raise
            try:
                shutil.rmtree(previous)
            except OSError as err:
                logger.warning(
                    "The replaced directory '%s' is not removed: %s", previous, err
                )
        else:
            temporary.rename(path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
