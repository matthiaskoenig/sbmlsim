"""COMBINE archives of PEtab v2 problems."""

import logging
from pathlib import Path
from typing import Any

import yaml
from pymetadata.omex import EntryFormat, ManifestEntry, Omex

logger = logging.getLogger(__name__)

#: the entries of the YAML file of a problem which list tables
TABLE_FIELDS: tuple[str, ...] = (
    "parameter_files",
    "condition_files",
    "experiment_files",
    "measurement_files",
    "observable_files",
    "mapping_files",
)

#: the format of a model in the archive by its language
MODEL_FORMATS: dict[str, EntryFormat] = {"sbml": EntryFormat.SBML}


def create_petab_omex(omex_file: Path, yaml_file: Path) -> None:
    """Create a COMBINE archive for a PEtab v2 problem.

    The YAML file of the PEtab problem is the master entry of the archive, the
    models and the tables it references are added relative to the directory of
    the YAML file. A problem of PEtab v1 is converted with
    `petab.v2.petab1to2` first, see `sbmlsim.fit.petab_v2.conversion`.

    Args:
        omex_file: path of the COMBINE archive to write.
        yaml_file: path of the PEtab YAML file, all other files are relative to it.

    Raises:
        FileNotFoundError: if the YAML file or a referenced file does not exist.
        ValueError: if the problem is no problem of PEtab v2.
    """
    yaml_file = Path(yaml_file)
    if not yaml_file.exists():
        raise FileNotFoundError(f"PEtab YAML file does not exist: '{yaml_file}'")
    config: dict[str, Any] = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
    version = str(config.get("format_version", ""))
    if not version.startswith("2"):
        raise ValueError(
            f"'{yaml_file}' is a problem of PEtab '{version}', a COMBINE archive "
            f"is written of a problem of PEtab v2; convert it with "
            f"`petab.v2.petab1to2` first."
        )

    omex = Omex()
    # all locations are relative to the directory of the yaml file
    base_dir: Path = yaml_file.parent
    _add_entry(
        omex,
        base_dir=base_dir,
        location=yaml_file.name,
        entry_format=EntryFormat.YAML,
        master=True,
    )
    for model in (config.get("model_files") or {}).values():
        language = str(model.get("language", "")).lower()
        _add_entry(
            omex,
            base_dir=base_dir,
            location=str(model["location"]),
            entry_format=MODEL_FORMATS.get(language, EntryFormat.XML),
        )
    for field in TABLE_FIELDS:
        for table_file in config.get(field) or []:
            _add_entry(
                omex,
                base_dir=base_dir,
                location=str(table_file),
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
