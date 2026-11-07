"""The COMBINE archive of the models of a release: the SBML, its ODEs and antimony.

`write_archive` writes, for every given SBML model, its flat model (a model
with submodels of the SBML package `comp` is flattened, `<stem>_flat.xml`),
the ordinary differential equations of the flat model as markdown
(`sbmlode.OdeSystem`, `<flat stem>.md`) and its antimony (`<flat stem>.ant`),
the license texts of the models (CC BY-SA 4.0) and of the code (MIT, with the
copyright) and the COMBINE archive (`pymetadata.omex`) with the README, the
licenses, the models, the markdown, the antimony and further models and files.
The flat model of the first model is the master entry of the archive.

A framework which builds its models itself (e.g. the PBPK models of pkpdsim)
writes them and calls `write_archive` with them; the factory of any other
model package calls it with the SBML files it created.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pymetadata.omex import EntryFormat, ManifestEntry

__all__ = ["EXTRA_FORMATS", "ArchiveModel", "is_hierarchical", "write_archive"]

#: the formats of the extra files of an archive by their suffix, the names of
#: the members of `pymetadata.omex.EntryFormat`
EXTRA_FORMATS: dict[str, str] = {
    ".md": "MARKDOWN",
    ".ant": "ANTIMONY",
    ".txt": "PLAIN",
    ".png": "PNG",
    ".svg": "SVG",
    ".csv": "CSV",
    ".tsv": "TSV",
    ".pdf": "PDF",
}


@dataclass(frozen=True)
class ArchiveModel:
    """A model of an archive.

    Attributes:
        path: the SBML file of the model, in L3V2, e.g. a model with submodels.
        flat: the flat model of a model with submodels, if it exists already,
            e.g. written by the builder of the model; `write_archive` flattens
            a model with submodels without it into `<stem>_flat.xml`, a model
            without submodels is its own flat model.
    """

    path: Path
    flat: Path | None = None


def is_hierarchical(path: Path) -> bool:
    """Tell whether an SBML model has submodels of the SBML package `comp`.

    Args:
        path: the SBML file.

    Returns:
        Whether its model has a submodel.
    """
    import libsbml

    document: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(path))
    model: libsbml.Model | None = document.getModel()
    if model is None:
        return False
    plugin = model.getPlugin("comp")
    return plugin is not None and plugin.getNumSubmodels() > 0


def write_archive(
    models: Sequence[Path | ArchiveModel],
    out: Path,
    *,
    name: str,
    copyright: str,
    readme: Path | None = None,
    extra_models: Sequence[Path] = (),
    extra_files: Sequence[Path] = (),
) -> dict[str, dict[str, Any]]:
    """Write the flat models, their ODEs and antimony, the licenses and the archive.

    For every model the flat model (see `ArchiveModel`), the ODEs of the flat
    model as markdown and its antimony are written next to the flat model, the
    texts of CC BY-SA 4.0 and MIT (with the copyright) into `out`
    (`sbmlsim.licenses`) and the archive `<name>.omex` there, with the README,
    the licenses, every model, its flat model (the one of the first model the
    master entry), the markdown, the antimony, the extra models and the extra
    files.

    Every file of the archive has one location rule: it is below `out`, and
    its location in the archive is its path relative to it, e.g.
    `./models/<name>` for the directory `models` of a model package, the
    licenses `./mit.txt`, an extra model or file where it lies
    (`./models/<sid>_fluid.xml`, `./figures/<x>.png`). The README alone is
    placed, as `./README.md` of the archive, wherever the file lies.

    Args:
        models: the SBML files of the models, or `ArchiveModel`s with their
            flat models; the first is the master of the archive.
        out: the directory of the archive and the licenses, every other file
            of the archive lies below it.
        name: the name of the archive, `<name>.omex`.
        copyright: copyright of the MIT license, e.g. `2024-2026 Matthias König`.
        readme: the README of the model repository, left out of the archive if
            `None`.
        extra_models: further SBML files (in L3V2) which are entries of the
            archive, e.g. a submodel which is simulated on its own.
        extra_files: further files of other formats which are entries of the
            archive, e.g. the ODEs of an extra model (`.md`) or a figure
            (`.png`), with the format of their suffix (`EXTRA_FORMATS`, the
            antimony of `.ant` as the antimony of the flat model).

    Returns:
        The files with their entries in the COMBINE archive, by the keys of a
        factory, each `{"path": Path, "entry": ManifestEntry}`: `README`,
        `cc-by`, `mit`, for a model with submodels its stem (the model) and
        `<stem>_flat` (its flat model), for a model without its stem,
        `<flat stem>_md` and `<flat stem>_ant` of the markdown and the
        antimony, the stem of every extra model and `<stem>_<suffix>` of every
        extra file (`fluid_md`).

    Raises:
        ValueError: before anything is written, for no model, if the key of a
            model, an extra model or an extra file is the key of another
            entry, if a model, an extra model or file is not below `out` or
            the suffix of an extra file has no format.
    """
    from pymetadata.omex import EntryFormat, ManifestEntry, Omex
    from sbmlode import OdeSystem
    from sbmlutils.comp import flatten_sbml
    from sbmlutils.io import sbml_to_antimony

    from sbmlsim.licenses import License, write_license

    def entry(
        location: str, fmt: EntryFormat, *, master: bool = False
    ) -> ManifestEntry:
        return ManifestEntry(location=location, format=fmt, master=master)

    def location(path: Path, what: str) -> str:
        """The location of a file in the archive, its path below `out`."""
        try:
            relative = path.resolve().relative_to(out.resolve())
        except ValueError:
            raise ValueError(
                f"The {what} {str(path)!r} is not below {str(out)!r}, the "
                "directory of the archive."
            ) from None
        return f"./{relative.as_posix()}"

    if not models:
        raise ValueError(f"The archive '{name}' has no model.")
    given = [
        m if isinstance(m, ArchiveModel) else ArchiveModel(Path(m)) for m in models
    ]
    # the flat model of every model, before anything is written
    planned: list[tuple[ArchiveModel, Path, bool]] = []
    keys: set[str] = set()
    for model in given:
        location(model.path, "model")
        hierarchical = model.flat is not None or is_hierarchical(model.path)
        flat = model.flat or (
            model.path.with_name(f"{model.path.stem}_flat.xml")
            if hierarchical
            else model.path
        )
        names = [model.path.stem, *([flat.stem] if hierarchical else [])]
        names += [f"{flat.stem}_md", f"{flat.stem}_ant"]
        repeated = sorted(keys & set(names))
        if repeated:
            raise ValueError(f"The models of the archive '{name}' repeat {repeated}.")
        keys.update(names)
        planned.append((model, flat, hierarchical))
    extra_keys = [extra.stem for extra in extra_models]
    extra_keys += [f"{e.stem}_{e.suffix.removeprefix('.')}" for e in extra_files]
    for extra in extra_files:
        if EXTRA_FORMATS.get(extra.suffix.lower()) is None:
            raise ValueError(
                f"The extra file {extra.name!r} has the suffix {extra.suffix!r} of "
                f"no format of the archive, one of {sorted(EXTRA_FORMATS)}."
            )
    for key, extra in zip(extra_keys, [*extra_models, *extra_files], strict=True):
        if key in keys or key in ("README", "cc-by", "mit"):
            kind = "model" if extra in extra_models else "file"
            raise ValueError(
                f"The extra {kind} {extra.name!r} has the key {key!r} of an entry "
                "of the archive."
            )
        location(extra, f"extra {'model' if extra in extra_models else 'file'}")
        keys.add(key)

    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    if readme is not None:
        results["README"] = {
            "path": readme,
            "entry": entry("./README.md", EntryFormat.MARKDOWN),
        }
    cc_by = write_license(out, License.CC_BY_SA_4_0)
    mit = write_license(out, License.MIT, copyright=copyright)
    results["cc-by"] = {
        "path": cc_by,
        "entry": entry(location(cc_by, "license"), EntryFormat.TXT),
    }
    results["mit"] = {
        "path": mit,
        "entry": entry(location(mit, "license"), EntryFormat.TXT),
    }
    for k, (model, flat, hierarchical) in enumerate(planned):
        if hierarchical and model.flat is None:
            flatten_sbml(model.path, flat)
        md_path = flat.with_suffix(".md")
        OdeSystem.from_sbml(flat).write(md_path)
        ant_path = flat.with_suffix(".ant")
        ant_path.write_text(sbml_to_antimony(flat), encoding="utf-8")
        master = k == 0
        if hierarchical:
            results[model.path.stem] = {
                "path": model.path,
                "entry": entry(location(model.path, "model"), EntryFormat.SBML_L3V2),
            }
        results[flat.stem] = {
            "path": flat,
            "entry": entry(
                location(flat, "model"), EntryFormat.SBML_L3V2, master=master
            ),
        }
        results[f"{flat.stem}_md"] = {
            "path": md_path,
            "entry": entry(location(md_path, "markdown"), EntryFormat.MARKDOWN),
        }
        results[f"{flat.stem}_ant"] = {
            "path": ant_path,
            "entry": entry(location(ant_path, "antimony"), EntryFormat.ANTIMONY),
        }
    for extra in extra_models:
        results[extra.stem] = {
            "path": extra,
            "entry": entry(location(extra, "extra model"), EntryFormat.SBML_L3V2),
        }
    for extra in extra_files:
        fmt = EXTRA_FORMATS[extra.suffix.lower()]
        results[f"{extra.stem}_{extra.suffix.removeprefix('.')}"] = {
            "path": extra,
            "entry": entry(location(extra, "extra file"), getattr(EntryFormat, fmt)),
        }

    omex = Omex()
    for info in results.values():
        omex.add_entry(entry_path=info["path"], entry=info["entry"])
    omex.to_omex(omex_path=out / f"{name}.omex")
    return results
