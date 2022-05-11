"""Testing reading of numl files."""
import pytest

from sbmlsim.combine.sedml.numl import NumlParser
from tests import DATA_DIR


BASE_DIR = DATA_DIR / "sedml" / "data"

SOURCE_NUML = BASE_DIR / "./oscli.xml"
SOURCE_NUML_1D = BASE_DIR / "./numlData1D.xml"
SOURCE_NUML_2D = BASE_DIR / "./numlData2D.xml"
SOURCE_NUML_2DRC = BASE_DIR / "./numlData2DRC.xml"


@pytest.mark.skip(reason="no SED-ML support")
def test_load_numl() -> None:
    """Load NuML."""
    data = NumlParser.load_numl_data(SOURCE_NUML)
    assert data is not None


@pytest.mark.skip(reason="no SED-ML support")
def test_load_numl_1D() -> None:
    """Load NuML."""
    data = NumlParser.load_numl_data(SOURCE_NUML_1D)
    assert data is not None


@pytest.mark.skip(reason="no SED-ML support")
def test_load_numl_2D() -> None:
    """Load NuML."""
    data = NumlParser.load_numl_data(SOURCE_NUML_2D)
    assert data is not None


@pytest.mark.skip(reason="no SED-ML support")
def test_load_numl_2DRC() -> None:
    """Load NuML."""
    data = NumlParser.load_numl_data(SOURCE_NUML_2D)
    assert data is not None
