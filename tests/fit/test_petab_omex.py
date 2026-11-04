"""Testing OMEX generation for PEtab problems."""

from pathlib import Path

import pytest
from pymetadata.omex import Omex

from sbmlsim.fit.petab_omex import create_petab_omex

PETAB_DIR = Path(__file__).parent.parent.parent / "examples" / "petab"
BOEHM_YAML = PETAB_DIR / "Boehm_JProteomeRes2014" / "problem.yaml"


def test_create_petab_omex_boehm(tmp_path: Path) -> None:
    """Create a COMBINE archive for the Boehm PEtab problem."""
    omex_file = tmp_path / "boehm.omex"
    create_petab_omex(omex_file=omex_file, yaml_file=BOEHM_YAML)
    assert omex_file.exists()

    omex = Omex.from_omex(omex_file)
    locations = [e.location for e in omex.manifest.entries]
    # the archive itself and its manifest are entries, too
    assert sorted(set(locations) - {".", "./manifest.xml"}) == [
        "./measurements.tsv",
        "./model.xml",
        "./observables.tsv",
        "./parameters.tsv",
        "./problem.yaml",
    ]


def test_create_petab_omex_single_master(tmp_path: Path) -> None:
    """Only the PEtab YAML file is the master entry of the archive."""
    omex_file = tmp_path / "boehm.omex"
    create_petab_omex(omex_file=omex_file, yaml_file=BOEHM_YAML)

    omex = Omex.from_omex(omex_file)
    masters = [e.location for e in omex.manifest.entries if e.master]
    assert masters == ["./problem.yaml"]


def test_create_petab_omex_missing_yaml(tmp_path: Path) -> None:
    """A missing PEtab YAML file raises a FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        create_petab_omex(
            omex_file=tmp_path / "missing.omex",
            yaml_file=tmp_path / "missing.yaml",
        )


def test_a_problem_of_petab_v1_is_refused(tmp_path: Path) -> None:
    """A problem of PEtab v1 is converted with `petab1to2` first."""
    yaml_file = tmp_path / "problem.yaml"
    yaml_file.write_text(
        "format_version: 1\nparameter_file: parameters.tsv\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="petab1to2"):
        create_petab_omex(omex_file=tmp_path / "v1.omex", yaml_file=yaml_file)
