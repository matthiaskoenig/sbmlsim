"""
Definition of data and files for the tests.
The files are located in the data directory.
"""
import os
from os.path import join as pjoin

TEST_PATH = os.path.dirname(os.path.abspath(__file__))  # directory of test files
DATA_PATH = pjoin(TEST_PATH, 'data')  # directory of data for tests

MODEL_REPRESSILATOR = pjoin(DATA_PATH, "models", "repressilator.xml")
MODEL_GLCWB = pjoin(DATA_PATH, "models", "body19_livertoy_flat.xml")