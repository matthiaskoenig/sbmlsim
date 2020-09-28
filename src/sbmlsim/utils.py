import functools
import hashlib
import inspect
import time
import warnings


def md5_for_path(path):
    """Calculates MD5 of file content."""

    # Open,close, read file and calculate MD5 on its contents
    with open(path, "rb") as f_check:
        # read contents of the file
        data = f_check.read()
        # pipe contents of the file through
        return hashlib.md5(data).hexdigest()


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def timeit(method):
    """Timing decorator"""

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("{:20}  {:8.4f} [s]".format(method.__name__, (te - ts)))
        return result

    return timed


def function_name():
    """Returns current function name"""
    frame = inspect.currentframe()
    return inspect.getframeinfo(frame).function
