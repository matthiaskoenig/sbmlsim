"""Resources."""
from pathlib import Path

RESOURCES_DIR = Path(__file__).parent

# models
MODEL_REPRESSILATOR_SBML = RESOURCES_DIR / "models" / "repressilator.xml"
MODEL_MIDAZOLAM_SBML = RESOURCES_DIR / "models" / "midazolam_body_flat.xml"
MODEL_DEMO_SBML = RESOURCES_DIR / "models" / "Koenig_demo_14.xml"
