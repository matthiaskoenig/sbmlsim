"""Helper module for downloading COVID-19 biomodels."""

import json
from pathlib import Path
from pprint import pprint

import requests
from sbmlutils.biomodels import download_biomodel_omex


def query_covid19_biomodels() -> list[str]:
    """Query the COVID-19 biomodels.

    :return list of biomodel identifiers
    """
    url = "https://www.ebi.ac.uk/biomodels/search?query=submitter_keywords%3A%22COVID-19%22%20AND%20curationstatus%3A%22Manually%20curated%22&numResults=100&format=json"
    response = requests.get(url)
    response.raise_for_status()
    json = response.json()
    biomodel_ids = [model["id"] for model in json["models"]]
    return sorted(biomodel_ids)


def get_covid19_model(output_dir: Path) -> dict[str, Path]:
    """Get all manually curated COVID-19 models.

    :return dictionary of model ids to Paths.
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    biomodel_ids = query_covid19_biomodels()
    pprint(biomodel_ids)
    omex_paths = {}
    for biomodel_id in biomodel_ids:
        omex_path = output_dir / f"{biomodel_id}.omex"
        path = download_biomodel_omex(biomodel_id, omex_path=omex_path)
        omex_paths[biomodel_id] = str(path)

    return omex_paths


if __name__ == "__main__":
    omex_paths = get_covid19_model(output_dir=Path(__file__).parent / "results")
    # store json
    with open(Path(__file__).parent / "models.json", "w") as f_json:
        json.dump(omex_paths, f_json, indent=2)
