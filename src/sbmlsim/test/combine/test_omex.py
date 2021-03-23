"""
Testing the omex module.
"""

from pathlib import Path

import pytest

from sbmlsim.combine.omex import Creator, Entry, Omex
from sbmlsim.test import DATA_DIR


OMEX_SHOWCASE = DATA_DIR / "omex" / "CombineArchiveShowCase.omex"


@pytest.mark.parametrize("method", ["zip", "omex"])
def test_extract(tmp_path: Path, method: str) -> None:
    """Test extraction of COMBINE archive."""
    omex = Omex(omex_path=OMEX_SHOWCASE, working_dir=tmp_path)
    omex.extract(output_dir=tmp_path, method=method)


@pytest.mark.parametrize(
    "format_key, count",
    [
        ("sed-ml", 2),
        ("sbml", 1),
        ("cellml", 1),
    ],
)
def test_locations_by_format_omex(format_key: str, count: int, tmp_path: Path) -> None:
    """Test getting locations by format"""
    omex = Omex(omex_path=OMEX_SHOWCASE, working_dir=tmp_path)
    locations = omex.locations_by_format(format_key=format_key, method="omex")
    assert len(locations) == count


@pytest.mark.parametrize(
    "format_key, count",
    [
        ("sed-ml", 2),
        ("sbml", 1),
        ("cellml", 1),
    ],
)
def test_locations_by_format_zip(format_key: str, count: int, tmp_path: Path) -> None:
    """Test getting locations by format."""
    omex = Omex(omex_path=OMEX_SHOWCASE, working_dir=tmp_path)
    locations = omex.locations_by_format(format_key=format_key, method="zip")
    assert len(locations) == count


def test_locations_by_format_omex2(tmp_path: Path) -> None:
    """Test getting locations."""
    omex = Omex(omex_path=OMEX_SHOWCASE, working_dir=tmp_path)
    locations = omex.locations_by_format(format_key="sed-ml", method="omex")

    # master=True file first
    assert locations[0].endswith("Calzone2007-simulation-figure-1B.xml")
    # master=False afterwards
    assert locations[1].endswith("Calzone2007-default-simulation.xml")


def test_locations_by_format_zip2(tmp_path: Path) -> None:
    """Test getting locations."""
    omex = Omex(omex_path=OMEX_SHOWCASE, working_dir=tmp_path)
    locations = omex.locations_by_format(format_key="sed-ml", method="zip")
    # master=True file first
    assert "experiment/Calzone2007-simulation-figure-1B.xml" in locations
    # master=False afterwards
    assert "experiment/Calzone2007-default-simulation.xml" in locations


def test_list_contents(tmp_path: Path) -> None:
    """Test the listing of the contents."""
    omex = Omex(omex_path=OMEX_SHOWCASE, working_dir=tmp_path)
    contents = omex.list_contents()
    assert len(contents) == 20


def test_str(tmp_path: Path) -> None:
    """Print contents of archive."""
    omex = Omex(omex_path=OMEX_SHOWCASE, working_dir=tmp_path)
    omex_str = omex.__str__()
    assert isinstance(omex_str, str)
