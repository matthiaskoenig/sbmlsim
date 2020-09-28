"""
Testing biomodel model resources
"""
import libsbml

from sbmlsim.model import model_resources


def test_from_url():
    mid = "BIOMD0000000012"
    url = f"https://www.ebi.ac.uk/biomodels/model/download/{mid}?filename={mid}_url.xml"

    sbml_str = model_resources.model_from_url(url)
    assert sbml_str
    doc = libsbml.readSBMLFromString(sbml_str)  # type: libsbml.SBMLDocument
    assert doc


def test_from_urn1():
    """Check that string is returned.

    :return:
    """
    urn = "urn:miriam:biomodels.db:BIOMD0000000139"
    sbml_str = model_resources.model_from_urn(urn)
    assert sbml_str
    doc = libsbml.readSBMLFromString(sbml_str)  # type: libsbml.SBMLDocument
    assert doc
