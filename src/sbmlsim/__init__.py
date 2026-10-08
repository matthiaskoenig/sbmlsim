"""sbmlsim package."""

from pathlib import Path

__author__ = "Matthias Koenig"
__version__ = "0.8.4"


BASE_PATH = Path(__file__).parent
RESOURCES_DIR = BASE_PATH / "resources"

# the quantities of the unit registry of the package, imported after the
# version, which hatchling reads from this file
from sbmlsim.units import Q  # noqa: E402

__all__ = ["BASE_PATH", "RESOURCES_DIR", "Q"]
