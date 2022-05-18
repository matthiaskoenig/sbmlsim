"""Pytest configuration."""

from pathlib import Path

import pytest

from sbmlsim.model.rr_model import roadrunner
from sbmlsim.resources import (
    MODEL_DEMO_SBML,
    MODEL_MIDAZOLAM_SBML,
    MODEL_REPRESSILATOR_SBML,
)


data_dir = Path(__file__).parent / "data"


@pytest.fixture
def repressilator_model_state() -> str:
    """Get repressilator roadrunner state."""
    rr: roadrunner.RoadRunner = roadrunner.RoadRunner(str(MODEL_REPRESSILATOR_SBML))
    return rr.saveStateS()


@pytest.fixture
def repressilator_path() -> str:
    """Get repressilator SBML path."""
    return MODEL_REPRESSILATOR_SBML


@pytest.fixture
def demo_path() -> str:
    """Get demo SBML path."""
    return MODEL_DEMO_SBML
