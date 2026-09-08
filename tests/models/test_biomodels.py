"""Test biomodel model resources.

The tests download models from BioModels. The service can be unreachable, and
it refuses the requests of some hosts (the GitHub runners get `403 Forbidden`),
so the tests are skipped when the service does not answer.
"""

import libsbml
import pytest
import requests

from sbmlsim.model import model_resources

#: a model which exists, used by the tests and by the probe
BIOMODEL_ID = "BIOMD0000000012"


@pytest.fixture(scope="module")
def biomodels_available() -> None:
    """Skip the test when the BioModels download service does not answer.

    Raises:
        Skipped: if the service is unreachable or refuses the request.
    """
    url = f"https://www.ebi.ac.uk/biomodels/model/download/{BIOMODEL_ID}"
    try:
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
    except requests.RequestException as err:
        pytest.skip(f"BioModels does not answer '{url}': {err}")


def _check_sbml_str(sbml_str: str) -> None:
    """Check SBML string."""
    assert sbml_str
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml_str)
    assert doc
    model: libsbml.Model = doc.getModel()
    assert model


@pytest.mark.usefixtures("biomodels_available")
def test_from_biomodels_url() -> None:
    """Test from BioModels URL."""
    mid = BIOMODEL_ID
    url = f"https://www.ebi.ac.uk/biomodels/model/download/{mid}?filename={mid}_url.xml"
    sbml_str = model_resources.model_from_url(url)
    _check_sbml_str(sbml_str=sbml_str)


@pytest.mark.usefixtures("biomodels_available")
def test_from_biomodels_url_deprecated() -> None:
    """Test from deprecated BioModels URL."""
    mid = BIOMODEL_ID
    url = f"https://www.ebi.ac.uk/biomodels-main/download?mid={mid}"
    sbml_str = model_resources.model_from_url(url)
    _check_sbml_str(sbml_str=sbml_str)


@pytest.mark.usefixtures("biomodels_available")
def test_from_biomodels_urn() -> None:
    """Check that string is returned."""
    urn = "urn:miriam:biomodels.db:BIOMD0000000139"
    sbml_str = model_resources.model_from_urn(urn)
    _check_sbml_str(sbml_str=sbml_str)
