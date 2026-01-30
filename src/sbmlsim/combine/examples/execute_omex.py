"""
Execute a COMBINE archive.
"""
from pathlib import Path

from sbmlsim.combine.examples import execute_sedml
from tests import DATA_DIR


def run_repressilator():
    repressilator_omex = DATA_DIR / "omex" / "tellurium" / "repressilator.omex"
    working_dir = Path(__file__).parent / "results" / "repressilator_omex"
    working_dir.mkdir(exist_ok=True)
    execute_sedml(path=repressilator_omex, working_dir=working_dir)


def run_omex(omex_path: Path):
    # print(omex_path)
    working_dir = Path(__file__).parent / "results" / omex_path.name
    # print(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    execute_sedml(path=omex_path, working_dir=working_dir, output_path=working_dir)


if __name__ == "__main__":

    biomodels_omex_base_path = DATA_DIR / "combine" / "omex" / "biomodels" / "omex"
    biomodels_omex_paths = []
    for path in Path(biomodels_omex_base_path).rglob("*.omex"):
        biomodels_omex_paths.append(path)
    biomodels_omex_paths = sorted(biomodels_omex_paths)

    for omex_path in [biomodels_omex_base_path / "BIOMD0000000111_fi4_sedml.omex"]:
        run_omex(omex_path)

    # for omex_path in sorted(biomodels_omex_paths):
    #     run_biomodel_omex(omex_path)

    "/home/mkoenig/git/sbmlsim/src/sbmlsim/tests/data/combine/omex/jws/omex/fraser2002_fig1a_1b_2a_2b.sedx"
    "/home/mkoenig/git/sbmlsim/src/sbmlsim/tests/data/combine/omex/jws/omex/levering2012_fig2-user.sedx",
    "/home/mkoenig/git/sbmlsim/src/sbmlsim/tests/data/combine/omex/jws/omex/levering2012_fig5-user.sedx",
    "/home/mkoenig/git/sbmlsim/src/sbmlsim/tests/data/combine/omex/jws/omex/martins2016_fig4b.sedx"
