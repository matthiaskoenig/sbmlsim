from pathlib import Path

import pytest

from sbmlsim.model.rr_model import roadrunner


data_dir = Path(__file__).parent / "data"


@pytest.fixture
def repressilator_model_state() -> str:
    """Get repressilator roadrunner state."""
    model_path: Path = data_dir / "models" / "repressilator.xml"
    rr: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
    return rr.saveStateS()


@pytest.fixture
def repressilator_path() -> str:
    """Get repressilator SBML path."""
    return data_dir / "models" / "repressilator.xml"


@pytest.fixture
def demo_path() -> str:
    """Get demo SBML path."""
    return data_dir / "models" / "Koenig_demo_14.xml"
