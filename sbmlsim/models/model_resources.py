"""
Interacting with model resources to retrieve models.
This currently includes BioModels, but can easily be extended to other models.
"""
import Path
import re
import requests


def is_path(source, base_path: Path = None):
    """Check if source is a path.

    Tries to resolve source relative to base_path.
    """
    if is_urn(source):
        return False
    elif is_http(source):
        return False
    elif isinstance(source, Path):
        return True
    elif isinstance(source, str):
        # try to resolve path
        if base_path:
            path = Path(base_path) / source
            return path.exists()
        else:
            path = Path(source)
            return path.exists()

    raise ValueError(f"Unclear if source is path: {source}")


def is_urn(source):
    return source.lower().startswith('urn')

def is_http(source):
    return source.lower().startswith('http')


def sbml_from_url(url: str) -> str:
    """ Get SBML string from given URL

    Handles redirects of the download page.

    :param url:
    :return:
    """
    response = requests.get(url, allow_redirects=True)
    response.raise_for_status()
    sbml = response.content

    # bytes array in py3
    return str(sbml.decode("utf-8"))


# --- BioModels ---
def biomodels_mid_from_str(text) -> str:
    """Resolve biomodel id from string."""

    pattern = "((BIOMD|MODEL)\d{10})|(BMID\d{12})"
    match = re.search(pattern, text)
    mid = match.group(0)
    return mid


def sbml_from_biomodels_urn(urn) -> str:
    """ Get SBML string from given BioModels URN.

    Searches for a BioModels identifier in the given urn and retrieves the SBML from biomodels.
    For example:
        urn:miriam:biomodels.db:BIOMD0000000003.xml

    :param urn:
    :return: SBML string for given model urn
    """
    mid = biomodels_mid_from_str(urn)
    return sbml_from_biomodels_id(mid)


def sbml_from_biomodels_id(mid: str) -> str:
    """ Get SBML string from given BioModels identifier.

    :param mid: biomodels id
    :return: SBML string
    """
    url = f"https://www.ebi.ac.uk/biomodels-main/download?mid={mid}"
    return sbml_from_url(url)

