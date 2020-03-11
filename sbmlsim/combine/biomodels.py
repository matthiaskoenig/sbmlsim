"""
Interacting with biomodels to resolve models.
"""
import re
import requests


def mid_from_str(text) -> str:
    """Resolve biomodel id from string."""

    pattern = "((BIOMD|MODEL)\d{10})|(BMID\d{12})"
    match = re.search(pattern, text)
    mid = match.group(0)
    return mid


def from_urn(urn) -> str:
    """ Get SBML string from given BioModels URN.

    Searches for a BioModels identifier in the given urn and retrieves the SBML from biomodels.
    For example:
        urn:miriam:biomodels.db:BIOMD0000000003.xml

    Handles redirects of the download page.

    :param urn:
    :return: SBML string for given model urn
    """
    mid = mid_from_str(urn)
    url = f"https://www.ebi.ac.uk/biomodels-main/download?mid={mid}"
    return from_url(url)


def from_url(url) -> str:
    """ Get SBML string from given

    :param url:
    :return:
    """
    response = requests.get(url, allow_redirects=True)
    response.raise_for_status()

    sbml = response.content

    # bytes array in py3
    return str(sbml.decode("utf-8"))
