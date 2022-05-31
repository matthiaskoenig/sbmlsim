"""sbmlsim package."""
from pathlib import Path

__author__ = "Matthias Koenig"
__version__ = "0.2.2"


BASE_PATH = Path(__file__).parent
RESOURCES_DIR = BASE_PATH / "resources"

MODEL_REPRESSILATOR = RESOURCES_DIR / "models" / "repressilator.xml"
