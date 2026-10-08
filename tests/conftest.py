"""Pytest configuration."""

import logging
import multiprocessing
import re
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import roadrunner

from sbmlsim.log import PACKAGE_LOGGER
from sbmlsim.resources import DEMO_SBML, REPRESSILATOR_SBML

data_dir = Path(__file__).parent / "data"

# the message of the warning python 3.12 and later give when a process which runs
# threads forks
FORK_OF_THREADS = r"This process .* is multi-threaded, use of fork\(\)"


@pytest.fixture(autouse=True)
def _working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test in a temporary working directory.

    The examples write their figures and results into the working directory,
    so a test which runs an example must not write into the repository.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _package_logging() -> Iterator[None]:
    """Restore the logging configuration of the package after every test.

    A test which runs a command line tool or an example calls
    `log.enable_rich_logging()`, which adds a handler writing to stdout to the
    `sbmlsim` logger. Without the restore every later test in the same worker
    finds the log records in its captured output.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.fixture(autouse=True)
def _no_fork_of_threads() -> Iterator[None]:
    """Fail a test in which a process which runs threads is forked.

    No pool of sbmlsim forks, see `sbmlsim.utils.process_context`: a fork of a
    process with threads may deadlock in the child, and the process of a test
    runs threads (every worker of pytest-xdist does). Python reports the fork
    as a `DeprecationWarning` which it clears right after raising it, so the
    filter `error` cannot turn it into a failure and the warning is recorded
    instead. Every other warning is passed on, as it would be without the
    fixture.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("always", message=FORK_OF_THREADS)
        yield
    forks = [w for w in caught if re.match(FORK_OF_THREADS, str(w.message))]
    for w in caught:
        if w not in forks:
            warnings.warn_explicit(
                w.message, w.category, w.filename, w.lineno, source=w.source
            )
    assert not forks, (
        f"a process which runs threads was forked at "
        f"{forks[0].filename}:{forks[0].lineno}: {forks[0].message}"
    )


@pytest.fixture
def start_methods(monkeypatch: pytest.MonkeyPatch) -> Callable[[str | None, str], None]:
    """Get the function which pretends the start methods of a platform.

    `start_methods(explicit, default)` makes `multiprocessing` report the start
    method set with `set_start_method` (`None` if none is set) and the default
    of the platform, the first of the supported start methods.
    """

    def pretend(explicit: str | None, default: str) -> None:
        monkeypatch.setattr(
            multiprocessing,
            "get_start_method",
            lambda allow_none=False: explicit if allow_none else explicit or default,
        )
        monkeypatch.setattr(
            multiprocessing,
            "get_all_start_methods",
            lambda: [default, *({"fork", "spawn", "forkserver"} - {default})],
        )

    return pretend


@pytest.fixture
def repressilator_model_state() -> str:
    """Get repressilator roadrunner state."""
    rr: roadrunner.RoadRunner = roadrunner.RoadRunner(str(REPRESSILATOR_SBML))
    return rr.saveStateS()


@pytest.fixture
def repressilator_path() -> Path:
    """Get repressilator SBML path."""
    return REPRESSILATOR_SBML


@pytest.fixture
def demo_path() -> Path:
    """Get demo SBML path."""
    return DEMO_SBML
