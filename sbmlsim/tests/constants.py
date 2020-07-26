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
