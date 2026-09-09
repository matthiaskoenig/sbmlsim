"""Utility functions."""

import functools
import hashlib
import inspect
import logging
import os
import time

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


def function_name() -> str:
    """Get current function name."""
    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("No current frame available.")
    return inspect.getframeinfo(frame).function
