"""Testing biomodel model resources."""

import libsbml

from sbmlsim.model import model_resources


def _check_sbml_str(sbml_str: str) -> None:
    assert sbml_str
    doc = libsbml.readSBMLFromString(sbml_str)  # type: libsbml.SBMLDocument
    assert doc
    model = doc.getModel()  # type: libsbml.Model
    assert model


def test_from_biomodels_url():
    mid = "BIOMD0000000012"
    url = f"https://www.ebi.ac.uk/biomodels/model/download/{mid}?filename={mid}_url.xml"
    sbml_str = model_resources.model_from_url(url)
    _check_sbml_str(sbml_str=sbml_str)


def test_from_biomodels_url_deprecated():
    mid = "BIOMD0000000012"
    url = f"https://www.ebi.ac.uk/biomodels-main/download?mid={mid}"
    sbml_str = model_resources.model_from_url(url)
    _check_sbml_str(sbml_str=sbml_str)


def test_from_biomodels_url():
    mid = "BIOMD0000000012"
    url = f"https://www.ebi.ac.uk/biomodels/model/download/{mid}?filename={mid}_url.xml"

    sbml_str = model_resources.model_from_url(url)
    _check_sbml_str(sbml_str=sbml_str)


def test_from_biomodels_urn():
    """Check that string is returned.

    :return:
    """
    urn = "urn:miriam:biomodels.db:BIOMD0000000139"
    sbml_str = model_resources.model_from_urn(urn)
    assert sbml_str
    doc = libsbml.readSBMLFromString(sbml_str)  # type: libsbml.SBMLDocument
    assert doc
