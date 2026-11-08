"""Utility functions."""

import functools
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)


def timeit(function):
    """Time function via timing decorator."""

    @functools.wraps(function)
    def timed(*args, **kw):
        ts = time.time()
        result = function(*args, **kw)
        te = time.time()

        if "log_time" in kw:
            name = kw.get("log_name", function.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            logger.info(
                "%-20s  %8.4f [s]", f"{function.__name__} <{os.getpid()}>", te - ts
            )
        return result

    return timed


def paths_text(paths: str | Path | Iterable[str | Path] | None) -> str:
    """Get the text of a path or of several paths, one path per line.

    The data path of simulation experiments is a path or several paths; `str`
    of a list of paths shows the representation of every path, e.g.
    `[PosixPath('/data')]`, instead of the path.

    Args:
        paths: a path, several paths or `None`.

    Returns:
        The paths, one per line, `"none"` for `None`.
    """
    if paths is None:
        return "none"
    if isinstance(paths, (str, Path)):
        return str(paths)
    return "\n".join(str(path) for path in paths)
