"""Test biomodel model resources."""

import libsbml

from sbmlsim.model import model_resources


def _check_sbml_str(sbml_str: str) -> None:
    """Check SBML string."""
    assert sbml_str
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml_str)
    assert doc
    model: libsbml.Model = doc.getModel()
    assert model


def test_from_biomodels_url() -> None:
    """Test from BioModels URL."""
    mid = "BIOMD0000000012"
    url = f"https://www.ebi.ac.uk/biomodels/model/download/{mid}?filename={mid}_url.xml"
    sbml_str = model_resources.model_from_url(url)
    _check_sbml_str(sbml_str=sbml_str)


def test_from_biomodels_url_deprecated() -> None:
    """Test from deprecated BioModels URL."""
    mid = "BIOMD0000000012"
    url = f"https://www.ebi.ac.uk/biomodels-main/download?mid={mid}"
    sbml_str = model_resources.model_from_url(url)
    _check_sbml_str(sbml_str=sbml_str)


def test_from_biomodels_urn() -> None:
    """Check that string is returned."""
    urn = "urn:miriam:biomodels.db:BIOMD0000000139"
    sbml_str = model_resources.model_from_urn(urn)
    assert sbml_str
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml_str)
    assert doc
