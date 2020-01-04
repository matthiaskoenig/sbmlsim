"""
Definition of data and files for tests.
The files are located in the data directory.
"""
from pathlib import Path

TEST_PATH = Path(__file__).parents[0]  # directory of test files
DATA_PATH = TEST_PATH / 'data'  # directory of data for tests

MODEL_REPRESSILATOR = DATA_PATH / "models" / "repressilator.xml"
MODEL_GLCWB = DATA_PATH / "models" / "body21_livertoy_flat.xml"
MODEL_DEMO = DATA_PATH / "models" / "Koenig_demo_14.xml"

# FIXME: make unitreg work with legacy paths
MODEL_ACETAMINOPHEN_LIVER = Path("/home/brandhorst/Coding/Work/Matthias/sbmlutils/sbmlutils/examples/models/acetaminophen/paracetamol_model_liver.xml")
MODEL_ACETAMINOPHEN_KIDNEY = Path("/home/brandhorst/Coding/Work/Matthias/sbmlutils/sbmlutils/examples/models/acetaminophen/paracetamol_model_kidney.xml")
MODEL_MIDAZOLAM = Path("/home/mkoenig/git/sbmlutils/sbmlutils/examples/models/midazolam/models/midazolam_liver.xml")
MODEL_MIDAZOLAM_BODY = Path("/home/mkoenig/git/sbmlutils/sbmlutils/examples/models/midazolam/models/midazolam_body.xml")
MODEL_ACETAMINOPHEN = Path("/home/mkoenig/git/sbmlutils/sbmlutils/examples/models/acetaminophen/paracetamol_model.xml")
