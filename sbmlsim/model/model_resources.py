"""
Interacting with model resources to retrieve models.
This currently includes BioModels, but can easily be extended to other models.
"""
from pathlib import Path
import re
import requests
from dataclasses import dataclass


@dataclass
class Source:
    """Class for keeping track of the resolved sources."""
    source: str
    path: Path = None  # if source is a path
    content: Path = None  # if source is something which has to be resolved

    def is_path(self):
        return self.path is not None

    def is_content(self):
        return self.content is not None

    def to_dict(self):
        # add keys individually for order!
        d = dict()
        d["source"] = str(self.source)
        d["path"] = str(self.path) if self.path else None
        d["content"] = self.content
        return d

    @classmethod
    def from_source(cls, source: str, base_dir: Path = None) -> 'Source':
        """Resolves the source string.

        # FIXME: handle the case of models given as strings.
        """
        path = None
        content = None

        if isinstance(source, str):
            if is_urn(source):
                content = model_from_urn(source)
            elif is_http(source):
                content = model_from_url()

        # is path
        if content is None:
            if base_dir:
                path = Path(base_dir) / Path(source)
            else:
                # uses current working dir as base_dir
                path = Path(source).resolve()
            path = path.resolve()
            if not path.exists():
                raise IOError(f"Path '{path}' for model source '{source}' "
                              f"does not exist.")

        return Source(source, path, content)


def is_urn(source):
    return source.lower().startswith('urn')


def is_http(source):
    return source.lower().startswith('http')


def model_from_urn(urn: str) -> str:
    """ Get model string from given URN"""
    if "biomodel" in urn:
        mid = parse_biomodels_mid(urn)
        content = model_from_biomodels(mid)
    else:
        raise ValueError(f"Unkown URN for model: {urn}")

    return content


def model_from_url(url: str) -> str:
    """ Get model string from given URL

    Handles redirects of the download page.

    :param url:
    :return:
    """
    response = requests.get(url, allow_redirects=True)
    response.raise_for_status()
    model_str = response.content

    # bytes array in py3
    return str(model_str.decode("utf-8"))


# --- BioModels ---
def parse_biomodels_mid(text) -> str:
    """Resolve biomodel id from string."""

    pattern = "((BIOMD|MODEL)\d{10})|(BMID\d{12})"
    match = re.search(pattern, text)
    mid = match.group(0)
    return mid

def model_from_biomodels(mid: str) -> str:
    """ Get SBML string from given BioModels identifier.

    :param mid: biomodels id
    :return: SBML string
    """
    url = f"https://www.ebi.ac.uk/biomodels-main/download?mid={mid}"
    return model_from_url(url)

