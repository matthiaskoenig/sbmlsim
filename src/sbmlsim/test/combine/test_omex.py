"""
Testing the omex module.
"""

from pathlib import Path

import pytest

from sbmlsim.test import DATA_DIR
from sbmlsim.combine import omex

OMEX_SHOWCASE = DATA_DIR / "omex" / "CombineArchiveShowCase.omex"


@pytest.mark.parametrize('method', ["zip", "omex"])
def test_omex_extract_combine_archive(tmp_path: Path, method: str) -> None:
    omex.extract_combine_archive(
        omex_path=OMEX_SHOWCASE, output_dir=tmp_path, method=method
    )


# testing the omex based methods
def test_getLocationsByFormat1():
    locations = omex.get_locations_by_format(omex_path=OMEX_SHOWCASE, format_key="sed-ml")
    assert len(locations) == 2


def test_getLocationsByFormat2():
    locations = omex.get_locations_by_format(omex_path=OMEX_SHOWCASE, format_key="sbml")
    assert len(locations) == 1


def test_getLocationsByFormat3():
    locations = omex.get_locations_by_format(omex_path=OMEX_SHOWCASE, format_key="cellml")
    assert len(locations) == 1



def test_getLocationsByFormat4():
    locations = omex.get_locations_by_format(omex_path=OMEX_SHOWCASE, format_key="sed-ml")
    assert len(locations) == 2
    # master=True file first
    assert locations[0].endswith("Calzone2007-simulation-figure-1B.xml")
    # master=False afterwards
    assert locations[1].endswith("Calzone2007-default-simulation.xml")


# test the zip based methods
def test_getLocationsByFormat1_zip():
    locations = omex.get_locations_by_format(omex_path=OMEX_SHOWCASE, format_key="sed-ml", method="zip")
    assert len(locations) == 2


def test_getLocationsByFormat2_zip():
    locations = omex.get_locations_by_format(omex_path=OMEX_SHOWCASE, format_key="sbml", method="zip")
    assert len(locations) == 1


def test_getLocationsByFormat3_zip():
    locations = omex.get_locations_by_format(omex_path=OMEX_SHOWCASE, format_key="cellml", method="zip")
    assert len(locations) == 1


def test_getLocationsByFormat4_zip():
    locations = omex.get_locations_by_format(omex_path=OMEX_SHOWCASE, format_key="sed-ml", method="zip")
    assert len(locations) == 2
    # master=True file first
    assert "experiment/Calzone2007-simulation-figure-1B.xml" in locations
    # master=False afterwards
    assert "experiment/Calzone2007-default-simulation.xml" in locations


def test_listContents():
    contents = omex.listContents(omexPath=OMEX_SHOWCASE, method="omex")
    assert len(contents) == 20


def test_printContents():
    omex.printContents(omexPath=OMEX_SHOWCASE)
