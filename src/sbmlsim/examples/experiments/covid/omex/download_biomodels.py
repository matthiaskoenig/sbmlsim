import json
import shutil
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import requests

from sbmlsim.combine.omex import Omex


def query_covid19_biomodels() -> List[str]:
    """Query the COVID-19 biomodels.

    :return List of biomodel identifiers
    """
    url = "https://www.ebi.ac.uk/biomodels/search?query=submitter_keywords%3A%22COVID-19%22%20AND%20curationstatus%3A%22Manually%20curated%22&numResults=100&format=json"
    response = requests.get(url)
    response.raise_for_status()
    json = response.json()
    biomodel_ids = [model["id"] for model in json["models"]]
    return sorted(biomodel_ids)


def download_file(url: str, path: Path):
    """Download file from url to path."""

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)
    return path


def download_biomodel_omex(biomodel_id: str, output_dir: Path) -> Path:
    """Downloads omex for given biomodel id."""
    import tempfile

    tmp_path = tempfile.mkdtemp()

    url = f"https://www.ebi.ac.uk/biomodels/model/download/{biomodel_id}"
    omex_path = Path(tmp_path) / f"{biomodel_id}.omex"

    print(f"Download: {omex_path}")
    download_file(url, omex_path)

    # within the omex archive is another omex archive which can be executed
    omex = Omex(omex_path=omex_path, working_dir=Path(tmp_path))
    omex.extract()
    contents = omex.list_contents()
    pprint(contents)

    local_omex_path = Path(output_dir) / f"{biomodel_id}.omex"
    for content in contents:
        if content.format == "http://identifiers.org/combine.specifications/omex":
            shutil.copyfile(src=Path(tmp_path) / content.location, dst=local_omex_path)

    shutil.rmtree(tmp_path)

    return local_omex_path


def get_covid19_model(output_dir: Path) -> Dict[str, Path]:
    """Get all manually curated COVID-19 models.

    :return dictionary of model ids to Paths.
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    biomodel_ids = query_covid19_biomodels()
    pprint(biomodel_ids)
    omex_paths = {}
    for biomodel_id in biomodel_ids:
        path = download_biomodel_omex(biomodel_id, output_dir=output_dir)
        omex_paths[biomodel_id] = str(path)
    return omex_paths


if __name__ == "__main__":
    omex_paths = get_covid19_model(output_dir=Path(__file__).parent / "results")
    # store json
    with open(Path(__file__).parent / "models.json", "w") as f_json:
        json.dump(omex_paths, f_json, indent=2)
