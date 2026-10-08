"""Utility functions."""

import functools
import hashlib
import logging
import multiprocessing
import os
import time
from collections.abc import Iterable
from multiprocessing.context import BaseContext
from pathlib import Path

logger = logging.getLogger(__name__)


def md5_for_path(path):
    """Calculate MD5 of file content."""
    # Open,close, read file and calculate MD5 on its contents
    with open(path, "rb") as f_check:
        # read contents of the file
        data = f_check.read()
        # pipe contents of the file through
        return hashlib.md5(data).hexdigest()


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


def process_context() -> BaseContext:
    """Get the multiprocessing context a pool of sbmlsim is created from.

    A start method set with `multiprocessing.set_start_method` is used, else
    the default of the platform, except `fork`: the process of a simulation
    runs the threads of roadrunner and of the linear algebra, and a fork of a
    process with threads may deadlock in the child. Python 3.14 made
    `forkserver` the default on linux for this reason, a pool takes it on
    python 3.13 as well, where it warns about the fork of a process with
    threads with a `DeprecationWarning`.

    Every pool of sbmlsim is created from this context, i.e.
    `process_context().Pool(...)` or
    `ProcessPoolExecutor(mp_context=process_context())`, and the start method
    of the process is not fixed by the call (`get_context()` without a method
    would), because a start method which is set is read as one the user chose.

    Python 3.13 fixes the start method of the process to the default of the
    platform when a process is started by `spawn` or `forkserver`, python 3.14
    does not. On linux the first pool of this context, or of any other code
    which does not fork, would make every later call read `fork` as the choice
    of the user. A start method which equals the default of the platform is
    therefore no choice, so on python 3.13 on linux `fork` cannot be asked for,
    while it can wherever `fork` is not the default (python 3.14, macOS).

    Returns:
        The context.
    """
    method = multiprocessing.get_start_method(allow_none=True)
    # the first of the supported methods is the default of the platform
    methods = multiprocessing.get_all_start_methods()
    if method is None or method == methods[0]:
        method = methods[0]
        if method == "fork" and "forkserver" in methods:
            method = "forkserver"
    return multiprocessing.get_context(method)
