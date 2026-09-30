"""Pytest configuration."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import roadrunner

from sbmlsim.log import PACKAGE_LOGGER
from sbmlsim.resources import DEMO_SBML, REPRESSILATOR_SBML

data_dir = Path(__file__).parent / "data"


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
