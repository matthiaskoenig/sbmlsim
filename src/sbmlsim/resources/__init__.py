"""Resources."""
from pathlib import Path

RESOURCES_DIR = Path(__file__).parent

# models
REPRESSILATOR_SBML = RESOURCES_DIR / "models" / "repressilator.xml"
MIDAZOLAM_SBML = RESOURCES_DIR / "models" / "midazolam_body_flat.xml"
DEMO_SBML = RESOURCES_DIR / "models" / "Koenig_demo_14.xml"
