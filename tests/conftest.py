"""Pytest configuration."""

from pathlib import Path

import pytest
import roadrunner

from sbmlsim.resources import DEMO_SBML, REPRESSILATOR_SBML

data_dir = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test in a temporary working directory.

    The examples write their figures and results into the working directory,
    so a test which runs an example must not write into the repository.
    """
    monkeypatch.chdir(tmp_path)


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
