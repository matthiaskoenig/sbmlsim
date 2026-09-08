"""COMBINE archive for PEtab problems."""

import logging
from pathlib import Path

import numpy as np
from petab.v1 import (
    CONDITION_FILES,
    MEASUREMENT_FILES,
    OBSERVABLE_FILES,
    PARAMETER_FILE,
    PROBLEMS,
    SBML_FILES,
    VISUALIZATION_FILES,
)
from petab.v1.yaml import load_yaml
from pymetadata.omex import EntryFormat, ManifestEntry, Omex

logger = logging.getLogger(__name__)

TABLE_FIELDS = (
    MEASUREMENT_FILES,
    OBSERVABLE_FILES,
    VISUALIZATION_FILES,
    CONDITION_FILES,
)


def create_petab_omex(omex_file: Path, yaml_file: Path) -> None:
    """Create a COMBINE archive for a PEtab problem.

    The YAML file of the PEtab problem is the master entry of the archive, all
    files it references are added relative to the directory of the YAML file.

    Args:
        omex_file: path of the COMBINE archive to write.
        yaml_file: path of the PEtab YAML file, all other files are relative to it.

    Raises:
        FileNotFoundError: if the YAML file or a referenced file does not exist.
    """
    yaml_file = Path(yaml_file)
    if not yaml_file.exists():
        raise FileNotFoundError(f"PEtab YAML file does not exist: '{yaml_file}'")

    omex = Omex()

    # all locations are relative to the directory of the yaml file
    base_dir: Path = yaml_file.parent
    yaml_config = load_yaml(yaml_file)

    _add_entry(
        omex,
        base_dir=base_dir,
        location=yaml_file.name,
        entry_format=EntryFormat.YAML,
        master=True,
    )

    # parameter table(s); a single file name or a list of file names
    for parameter_file in np.array(yaml_config[PARAMETER_FILE]).flat:
        _add_entry(
            omex,
            base_dir=base_dir,
            location=str(parameter_file),
            entry_format=EntryFormat.TSV,
        )

    for problem in yaml_config[PROBLEMS]:
        for sbml_file in problem.get(SBML_FILES, []):
            _add_entry(
                omex,
                base_dir=base_dir,
                location=sbml_file,
                entry_format=EntryFormat.SBML,
            )

        for field in TABLE_FIELDS:
            for table_file in problem.get(field, []):
                _add_entry(
                    omex,
                    base_dir=base_dir,
                    location=table_file,
                    entry_format=EntryFormat.TSV,
                )

    omex.to_omex(Path(omex_file))


def _add_entry(
    omex: Omex,
    base_dir: Path,
    location: str,
    entry_format: EntryFormat,
    master: bool = False,
) -> None:
    """Add a single file of the PEtab problem to the archive.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    entry_path = base_dir / location
    if not entry_path.exists():
        raise FileNotFoundError(
            f"File of the PEtab problem does not exist: '{entry_path}'"
        )

    logger.debug("Adding '%s' to the COMBINE archive", location)
    omex.add_entry(
        entry_path=entry_path,
        entry=ManifestEntry(
            location=f"./{Path(location).as_posix()}",
            format=entry_format,
            master=master,
        ),
    )
