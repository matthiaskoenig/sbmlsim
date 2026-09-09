"""Testing OMEX generation for PEtab problems."""

from pathlib import Path

import pytest
from pymetadata.omex import Omex

from sbmlsim.fit.petab_omex import create_petab_omex

PETAB_DIR = Path(__file__).parent.parent.parent / "examples" / "petab"
BOEHM_YAML = PETAB_DIR / "Boehm_JProteomeRes2014" / "Boehm_JProteomeRes2014.yaml"


def test_create_petab_omex_boehm(tmp_path: Path) -> None:
    """Create a COMBINE archive for the Boehm PEtab problem."""
    omex_file = tmp_path / "boehm.omex"
    create_petab_omex(omex_file=omex_file, yaml_file=BOEHM_YAML)
    assert omex_file.exists()

    omex = Omex.from_omex(omex_file)
    locations = [e.location for e in omex.manifest.entries]
    assert "./Boehm_JProteomeRes2014.yaml" in locations
    assert "./parameters_Boehm_JProteomeRes2014.tsv" in locations
    assert "./model_Boehm_JProteomeRes2014.xml" in locations
    assert "./measurementData_Boehm_JProteomeRes2014.tsv" in locations


def test_create_petab_omex_single_master(tmp_path: Path) -> None:
    """Only the PEtab YAML file is the master entry of the archive."""
    omex_file = tmp_path / "boehm.omex"
    create_petab_omex(omex_file=omex_file, yaml_file=BOEHM_YAML)

    omex = Omex.from_omex(omex_file)
    masters = [e.location for e in omex.manifest.entries if e.master]
    assert masters == ["./Boehm_JProteomeRes2014.yaml"]


def test_create_petab_omex_missing_yaml(tmp_path: Path) -> None:
    """A missing PEtab YAML file raises a FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        create_petab_omex(
            omex_file=tmp_path / "missing.omex",
            yaml_file=tmp_path / "missing.yaml",
        )
