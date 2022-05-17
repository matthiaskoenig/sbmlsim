from pathlib import Path

from sbmlsim.simulator.rr_model import roadrunner
import pytest

data_dir = Path(__file__).parent / "data"


@pytest.fixture
def repressilator_model_state() -> str:
    model_path: Path = data_dir / "models" / "repressilator.xml"
    rr: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
    return rr.saveStateS()


@pytest.fixture
def repressilator_path() -> str:
    model_path: Path = data_dir / "models" / "repressilator.xml"
    return model_path
