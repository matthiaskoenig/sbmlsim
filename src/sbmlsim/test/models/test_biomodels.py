"""
Testing temiriam module
"""
from sbmlsim.model import model_resources
import roadrunner


def test_from_url():
    mid = "BIOMD0000000012"
    url = f"https://www.ebi.ac.uk/biomodels-main/download?mid={mid}"
    sbml_str = model_resources.model_from_url(url)


def test_from_urn1():
    """ Check that string is returned.

    :return:
    """
    urn = 'urn:miriam:biomodels.db:BIOMD0000000139'
    sbml = model_resources.model_from_urn(urn)
    assert sbml is not None
    # check that string
    assert isinstance(sbml, str)


def test_from_urn2():
    """ Check that model can be loaded in roadrunner.

    :return:
    """
    urn = 'urn:miriam:biomodels.db:BIOMD0000000139'
    sbml = model_resources.model_from_urn(urn)

    print("*" * 80)
    print(type(sbml))
    print("*" * 80)
    print(sbml)
    print("*" * 80)

    r = roadrunner.RoadRunner(sbml)
    assert r is not None
