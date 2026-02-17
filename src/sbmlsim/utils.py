"""Utility functions."""
import functools
import hashlib
import inspect
import os
import time
import warnings

from pymetadata import log


logger = log.get_logger(__name__)


def md5_for_path(path):
    """Calculate MD5 of file content."""

    # Open,close, read file and calculate MD5 on its contents
    with open(path, "rb") as f_check:
        # read contents of the file
        data = f_check.read()
        # pipe contents of the file through
        return hashlib.md5(data).hexdigest()


def deprecated(function):
    """Get decorator for deprecation.

    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    @functools.wraps(function)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(function.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return function(*args, **kwargs)

    return new_func


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
                "{:20}  {:8.4f} [s]".format(
                    f"{function.__name__} <{os.getpid()}>", (te - ts)
                )
            )
        return result

    return timed


def function_name() -> str:
    """Get current function name."""
    frame = inspect.currentframe()
    return inspect.getframeinfo(frame).function
