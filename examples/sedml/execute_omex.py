"""Execute a COMBINE archive.

The results are written into `results/` in the working directory.
"""

from pathlib import Path

from sbmlsim.combine.sedml.runner import execute_sedml

#: test data of the repository, the archives are not part of the package
DATA_DIR = Path(__file__).parents[2] / "tests" / "data"


def run_omex(omex_path: Path, output_path: Path) -> None:
    """Execute the COMBINE archive and write the results to the output path."""
    working_dir = output_path / omex_path.stem
    working_dir.mkdir(parents=True, exist_ok=True)
    execute_sedml(path=omex_path, working_dir=working_dir, output_path=working_dir)


def run_repressilator(output_path: Path) -> None:
    """Execute the repressilator archive of tellurium."""
    run_omex(
        omex_path=DATA_DIR / "combine" / "omex" / "tellurium" / "repressilator.omex",
        output_path=output_path,
    )


if __name__ == "__main__":
    run_repressilator(output_path=Path.cwd() / "results")
