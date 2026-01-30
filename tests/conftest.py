"""Pytest configuration."""

from pathlib import Path

import pytest
import roadrunner

from sbmlsim.resources import DEMO_SBML, REPRESSILATOR_SBML


data_dir = Path(__file__).parent / "data"


@pytest.fixture
def repressilator_model_state() -> str:
    """Get repressilator roadrunner state."""
    rr: roadrunner.RoadRunner = roadrunner.RoadRunner(str(REPRESSILATOR_SBML))
    return rr.saveStateS()


@pytest.fixture
def repressilator_path() -> str:
    """Get repressilator SBML path."""
    return REPRESSILATOR_SBML


@pytest.fixture
def demo_path() -> str:
    """Get demo SBML path."""
    return DEMO_SBML
