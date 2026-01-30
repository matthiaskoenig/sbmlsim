"""COMBINE archive for PEtab problems."""

from pathlib import Path

import numpy as np
import petab
from petab import (
    CONDITION_FILES,
    MEASUREMENT_FILES,
    OBSERVABLE_FILES,
    PARAMETER_FILE,
    PROBLEMS,
    SBML_FILES,
    VISUALIZATION_FILES,
)
from pymetadata.omex import EntryFormat, ManifestEntry, Omex


def create_petab_omex(
    omex_file: Path,
    yaml_file: Path,
) -> None:
    """Create COMBINE archive for PETab."""
    omex = Omex()

    # get relative path from yaml file
    base_dir: Path = yaml_file.parent
    yaml_config = petab.yaml.load_yaml(yaml_file)

    # "PEtab YAML file"
    omex.add_entry(
        entry_path=yaml_file,
        entry=ManifestEntry(
            location=f"./{yaml_file.relative_to(base_dir)}",
            format=EntryFormat.YAML,
            master=True,
        ),
    )

    # Add parameter file(s) that describe a single parameter table.
    # Works for a single file name, or a list of file names.
    for parameter_subset_file in list(np.array(yaml_config[PARAMETER_FILE]).flat):
        omex.add_entry(
            entry_path=base_dir / parameter_subset_file,
            entry=ManifestEntry(
                location=parameter_subset_file,
                format=EntryFormat.TSV,
                master=True,
            ),
        )

    for problem in yaml_config[PROBLEMS]:
        for sbml_file in problem[SBML_FILES]:
            omex.add_entry(
                entry_path=base_dir / sbml_file,
                entry=ManifestEntry(
                    location=sbml_file,
                    format=EntryFormat.SBML,
                    master=True,
                ),
            )

        for field in [
            MEASUREMENT_FILES,
            OBSERVABLE_FILES,
            VISUALIZATION_FILES,
            CONDITION_FILES,
        ]:
            if field not in problem:
                continue

            for file in problem[field]:
                omex.add_entry(
                    entry_path=base_dir / file,
                    entry=ManifestEntry(
                        location=file,
                        format=EntryFormat.TSV,
                        master=True,
                    ),
                )

    omex.to_omex(omex_file)


if __name__ == "__main__":
    from sbmlsim import RESOURCES_DIR

    data_dir = RESOURCES_DIR / "testdata" / "petab" / "icg_example1"
    print(data_dir, data_dir.exists())

    create_petab_omex(
        omex_file=data_dir / "icg.omex",
        yaml_file=data_dir / "icg.yaml",
    )
