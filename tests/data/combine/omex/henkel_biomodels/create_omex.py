"""Create omex files from SED-ML files."""

from pathlib import Path

from pymetadata import omex as pyomex

from pymetadata import log

logger = log.get_logger(__name__)


EXAMPLES_DIR: Path = Path(__file__).parent
SEDML_DIR: Path = EXAMPLES_DIR / "sedml"
OMEX_DIR: Path = EXAMPLES_DIR / "omex"


def create_omex_from_sedml(sedml_path: Path, omex_path: Path) -> None:
    """Create a combine archive from SED-ML path."""

    omex = pyomex.Omex()
    entry = pyomex.ManifestEntry(
        master=True, format=pyomex.EntryFormat.SEDML, location=f"./{sedml_path.name}"
    )
    omex.add_entry(entry_path=sedml_path, entry=entry)
    omex.to_omex(omex_path=omex_path)
    logger.info(omex)


def create_all_omex() -> None:
    """Create all omex from the SED-ML file."""

    sedml_paths: list[Path] = []
    for p in SEDML_DIR.rglob("*"):
        sedml_suffixes = {".xml", ".sedml"}
        if p.is_file() and p.suffix in sedml_suffixes:
            sedml_paths.append(SEDML_DIR / p)

    for p in sorted(sedml_paths):
        logger.info(f"{p.name}")
        create_omex_from_sedml(sedml_path=p, omex_path=OMEX_DIR / f"{p.stem}.omex")


if __name__ == "__main__":
    create_all_omex()
