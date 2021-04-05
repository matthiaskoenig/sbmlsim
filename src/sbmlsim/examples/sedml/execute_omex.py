"""
Execute a COMBINE archive.
"""
from pathlib import Path

from sbmlsim.test import DATA_DIR

from sbmlsim.examples.sedml.execute_sedml import execute_omex

if __name__ == "__main__":
    repressilator_omex = DATA_DIR / "omex" / "tellurium" / "repressilator.omex"
    working_dir = Path(__file__).parent / "results" / "repressilator_omex"
    working_dir.mkdir(exist_ok=True)
    execute_omex(omex_path=repressilator_omex, working_dir=working_dir)
