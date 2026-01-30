"""Run COMBINE archive examples.

"""
import json
from pathlib import Path

from sbmlsim.combine.sedml.runner import execute_sedml


with open(Path(__file__).parent / "omex" / "models.json", "r") as f_json:
    omex_paths = json.load(f_json)

    for biomodel_id, omex_path_str in omex_paths.items():
        if biomodel_id.endswith("55"):
            continue

        print(f"Execute OMEX: {biomodel_id}: {omex_path_str}")

        # execute OMEX archive
        execute_sedml(
            path=Path(omex_path_str),
            working_dir=Path(__file__).parent / "omex" / "results" / biomodel_id,
            output_path=Path(__file__).parent / "omex" / "results" / biomodel_id,
        )
