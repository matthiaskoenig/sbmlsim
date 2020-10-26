"""
Definition of data and files for tests.
The files are located in the data directory.
"""


from pathlib import Path

TEST_PATH = Path(__file__).parents[0]  # directory of test files
DATA_DIR = TEST_PATH / "data"  # directory of data for tests

MODEL_PATH = DATA_DIR / "models"
MODEL_REPRESSILATOR = MODEL_PATH / "repressilator.xml"
MODEL_GLCWB = MODEL_PATH / "body21_livertoy_flat.xml"
MODEL_DEMO = MODEL_PATH / "Koenig_demo_14.xml"
MODEL_MIDAZOLAM = MODEL_PATH / "midazolam_body_flat.xml"
