"""Model resources.

Interacting with model resources to retrieve models.
This currently includes BioModels, but can easily be extended to other models.
"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import requests


logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Class for keeping track of the resolved sources."""

    source: str
    path: Optional[Path] = None  # if source is a path
    content: Optional[Path] = None  # if source is something which has to be resolved

    def is_path(self) -> bool:
        """Check if the source is a Path."""
        return self.path is not None

    def is_content(self) -> bool:
        """Check if the source is Content."""
        return self.content is not None

    def to_dict(self) -> Dict[str, str]:
        """Convert to dict.

        Used for serialization.
        """
        return {
            "source": str(self.source),
            "path": str(self.path) if self.path else None,
            "content": str(self.content),
        }

    @classmethod
    def from_source(cls, source: str, base_dir: Path = None) -> "Source":
        """Resolve the source string.

        # FIXME: handle the case of models given as strings.
        """
        if isinstance(source, Source):
            return source

        path = None
        content = None

        if isinstance(source, str):
            if is_urn(source):
                content = model_from_urn(source)
            elif is_http(source):
                content = model_from_url(url=source)

        # is path
        if content is None:
            if base_dir:
                path = Path(base_dir) / Path(source)
            else:
                # uses current working dir as base_dir
                path = Path(source).resolve()
            path = path.resolve()
            if not path.exists():
                raise IOError(
                    f"Path '{path}' for model source '{source}' " f"does not exist."
                )

        return Source(source, path, content)


def is_urn(source: str) -> bool:
    """Check if urn source."""
    return source.lower().startswith("urn")


def is_http(source: str) -> bool:
    """Check if http source."""
    return source.lower().startswith("http")


def model_from_urn(urn: str) -> str:
    """Get model string from given URN."""
    if "biomodel" in urn:
        mid = parse_biomodels_mid(urn)
        content = model_from_biomodels(mid)
    else:
        raise ValueError(f"Unkown URN for model: {urn}")

    return content


def model_from_url(url: str) -> str:
    """Get model string from given URL.

    Handles redirects of the download page.

    :param url:
    :return:
    """
    # check for special case of old biomodel urls
    if url.startswith("https://www.ebi.ac.uk/biomodels-main/download?mid="):
        mid = parse_biomodels_mid(url)
        logger.error(
            f"Use of deprecated biomodels URL '{url}' ,"
            f"use updated url instead: "
            f"'https://www.ebi.ac.uk/biomodels/model/download/{mid}?filename={mid}_url.xml'"
        )
        return model_from_biomodels(mid)

    response = requests.get(url, allow_redirects=True)
    response.raise_for_status()
    model_str = response.content

    # bytes array in py3
    return str(model_str.decode("utf-8"))


# --- BioModels ---
def parse_biomodels_mid(text: str) -> str:
    """Parse biomodel id from string."""
    pattern = r"((BIOMD|MODEL)\d{10})|(BMID\d{12})"
    match = re.search(pattern, text)
    if match:
        mid = match.group(0)
    else:
        raise ValueError(f"Biomodel id pattern '{pattern}' not found in string: 'text'")
    return mid


def model_from_biomodels(mid: str) -> str:
    """Get SBML string from given BioModels identifier.

    :param mid: biomodels id
    :return: SBML string
    """
    # query file information
    url = f"https://www.ebi.ac.uk/biomodels/{mid}?format=json"
    r = requests.get(url)
    r.raise_for_status()

    # query main file
    json = r.json()
    try:
        filename = json["files"]["main"][0]["name"]
        url = (
            f"https://www.ebi.ac.uk/biomodels/model/download/{mid}?filename={filename}"
        )
    except (TypeError, KeyError) as err:
        logger.error(
            f"Filename of 'main' file could not be resolved from response: " f"'{json}'"
        )
        raise err

    return model_from_url(url)
