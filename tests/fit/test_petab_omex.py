"""Testing OMEX generation for PETab problems."""
import pytest
from pymetadata.omex import Omex

from sbmlsim import RESOURCES_DIR
from sbmlsim.fit.petab_omex import create_petab_omex


data_dir = RESOURCES_DIR / "testdata" / "petab" / "icg_example1"


@pytest.mark.skip(reason="no fit support")
def test_create_petab_omex_icg(tmp_path) -> None:
    """Create PEtab for ICG problem."""
    omex_file = tmp_path / "icg.omex"
    create_petab_omex(
        omex_file=omex_file,
        yaml_file=data_dir / "icg.yaml",
    )
    assert omex_file

    # read omex
    omex = Omex.from_omex(omex_file)
    locations = [e.location for e in omex.manifest.entries]
    assert "./icg.yaml" in locations
