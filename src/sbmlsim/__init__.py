"""sbmlsim package."""

from pathlib import Path

__author__ = "Matthias Koenig"
__version__ = "0.24.0"


BASE_PATH = Path(__file__).parent
RESOURCES_DIR = BASE_PATH / "resources"

# the check of the installed libroadrunner, which must be a build of the
# `develop` branch (`sbmlsim.engine` imports the standard library and
# packaging, no roadrunner): an environment with the release of PyPI simulates
# and gives wrong results, so the import reports it with the install. Imported
# after the version, which hatchling reads from this file
from sbmlsim.engine import check_roadrunner  # noqa: E402

check_roadrunner()

# the quantities of the unit registry of the package
from sbmlsim.units import Q  # noqa: E402

__all__ = ["BASE_PATH", "RESOURCES_DIR", "Q", "check_roadrunner"]
