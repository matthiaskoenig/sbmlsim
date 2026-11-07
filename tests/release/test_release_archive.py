"""Tests of `write_archive`, the COMBINE archive of the models of a release."""

import zipfile
from pathlib import Path
from typing import Any

import pytest
from pymetadata.omex import EntryFormat

from sbmlsim.licenses import License, license_text
from sbmlsim.release.archive import ArchiveModel, is_hierarchical, write_archive
from tests.model.test_model_roadrunner_init import HIERARCHICAL_SBML
from tests.project_toy import TOY_PATH

COPYRIGHT = "2024-2026 Matthias König"
MINIMAL_SBML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
    '<model id="fluid"/></sbml>\n'
)


def _models(tmp_path: Path, directory: str = "models") -> list[Path]:
    """Write a model with submodels and the toy model, without submodels."""
    models = tmp_path / directory
    models.mkdir(exist_ok=True)
    comp = models / "simple.xml"
    comp.write_text(HIERARCHICAL_SBML, encoding="utf-8")
    toy = models / "toy.xml"
    toy.write_bytes((TOY_PATH / "models" / "toy.xml").read_bytes())
    return [comp, toy]


def _create(
    tmp_path: Path,
    *,
    extra: tuple[Path, ...] = (),
    files: tuple[Path, ...] = (),
    readme: bool = True,
) -> dict[str, dict[str, Any]]:
    """Write the archive of the two models below `tmp_path`."""
    readme_path = tmp_path / "README.md"
    readme_path.write_text("# simple\n", encoding="utf-8")
    return write_archive(
        _models(tmp_path),
        tmp_path,
        name="simple_model",
        copyright=COPYRIGHT,
        readme=readme_path if readme else None,
        extra_models=extra,
        extra_files=files,
    )


def test_a_model_with_submodels_is_hierarchical(tmp_path: Path) -> None:
    comp, toy = _models(tmp_path)
    assert is_hierarchical(comp)
    assert not is_hierarchical(toy)


def test_the_archive_has_the_flat_model_as_master(tmp_path: Path) -> None:
    """The flat model of the first model is the master entry."""
    results = _create(tmp_path)
    assert list(results) == [
        "README",
        "cc-by",
        "mit",
        "simple",
        "simple_flat",
        "simple_flat_md",
        "simple_flat_ant",
        "toy",
        "toy_md",
        "toy_ant",
    ]
    masters = [k for k, v in results.items() if v["entry"].master]
    assert masters == ["simple_flat"]
    assert results["simple_flat"]["entry"].location == "./models/simple_flat.xml"
    assert results["simple"]["entry"].format == EntryFormat.SBML_L3V2
    assert results["simple_flat_md"]["entry"].format == EntryFormat.MARKDOWN
    assert results["simple_flat_ant"]["entry"].format == EntryFormat.ANTIMONY
    assert results["toy"]["entry"].location == "./models/toy.xml"
    assert results["README"]["entry"].location == "./README.md"
    for info in results.values():
        assert info["path"].exists()
    assert not is_hierarchical(tmp_path / "models" / "simple_flat.xml")
    assert "model" in (tmp_path / "models" / "toy.ant").read_text(encoding="utf-8")
    assert "$$" in (tmp_path / "models" / "toy.md").read_text(encoding="utf-8")
    with zipfile.ZipFile(tmp_path / "simple_model.omex") as z:
        names = set(z.namelist())
    assert {
        "README.md",
        "cc-by-sa-4.0.txt",
        "mit.txt",
        "models/simple.xml",
        "models/simple_flat.xml",
        "models/simple_flat.md",
        "models/simple_flat.ant",
        "models/toy.xml",
        "models/toy.md",
        "models/toy.ant",
        "manifest.xml",
    } <= names


def test_a_given_flat_model_is_not_flattened_again(tmp_path: Path) -> None:
    """A builder which writes the flat model passes it, e.g. a PBPK build."""
    comp, toy = _models(tmp_path)
    flat = tmp_path / "models" / "built_flat.xml"
    flat.write_bytes(toy.read_bytes())
    results = write_archive(
        [ArchiveModel(comp, flat=flat)], tmp_path, name="built", copyright=COPYRIGHT
    )
    assert results["simple"]["path"] == comp
    assert results["built_flat"]["path"] == flat
    assert results["built_flat"]["entry"].master
    assert flat.read_bytes() == toy.read_bytes()
    assert not (tmp_path / "models" / "simple_flat.xml").exists()
    assert (tmp_path / "models" / "built_flat.md").is_file()


def test_the_archive_carries_the_extra_models(tmp_path: Path) -> None:
    """An extra SBML file is an entry by its stem, after the others."""
    (tmp_path / "models").mkdir()
    extra = tmp_path / "models" / "simple_fluid.xml"
    extra.write_text(MINIMAL_SBML, encoding="utf-8")
    results = _create(tmp_path, extra=(extra,))
    assert list(results)[-1] == "simple_fluid"
    entry = results["simple_fluid"]["entry"]
    assert entry.location == "./models/simple_fluid.xml"
    assert entry.format == EntryFormat.SBML_L3V2
    assert not entry.master
    with zipfile.ZipFile(tmp_path / "simple_model.omex") as z:
        assert "models/simple_fluid.xml" in z.namelist()


def test_the_licenses_are_the_texts(tmp_path: Path) -> None:
    """The license files are the texts, the MIT one with the copyright."""
    results = _create(tmp_path)
    assert results["cc-by"]["path"] == tmp_path / "cc-by-sa-4.0.txt"
    assert results["cc-by"]["path"].read_bytes() == license_text(
        License.CC_BY_SA_4_0
    ).encode("utf-8")
    assert results["mit"]["path"].read_bytes() == license_text(
        License.MIT, copyright=COPYRIGHT
    ).encode("utf-8")


def test_the_archive_returns_the_entries_of_a_factory(tmp_path: Path) -> None:
    """Every value is a path and an entry, without a README the key is left out."""
    results = _create(tmp_path, readme=False)
    assert "README" not in results
    for info in results.values():
        assert set(info) == {"path", "entry"}
        assert isinstance(info["path"], Path)


def test_an_extra_model_may_not_take_a_key_of_the_archive(tmp_path: Path) -> None:
    """An extra model named like a written file is an error, before anything is written."""
    (tmp_path / "models").mkdir()
    extra = tmp_path / "models" / "simple_flat.xml"
    extra.write_text(MINIMAL_SBML, encoding="utf-8")
    with pytest.raises(ValueError, match="simple_flat"):
        _create(tmp_path, extra=(extra,))
    assert not (tmp_path / "mit.txt").exists()


def test_the_models_may_not_repeat_a_key(tmp_path: Path) -> None:
    toy = _models(tmp_path)[1]
    with pytest.raises(ValueError, match="repeat"):
        write_archive([toy, toy], tmp_path, name="x", copyright=COPYRIGHT)
    with pytest.raises(ValueError, match="no model"):
        write_archive([], tmp_path, name="x", copyright=COPYRIGHT)


def test_the_archive_carries_extra_files(tmp_path: Path) -> None:
    """A further file (the ODEs of an extra model, a figure) is an entry by its name.

    The key is the stem with the suffix (`simple_fluid_md`), the location the
    path below the directory of the archive, the format the one of the suffix.
    """
    (tmp_path / "models").mkdir()
    (tmp_path / "figures").mkdir()
    extra = tmp_path / "models" / "simple_fluid.xml"
    extra.write_text(MINIMAL_SBML, encoding="utf-8")
    md = tmp_path / "models" / "simple_fluid.md"
    md.write_text("# fluid\n", encoding="utf-8")
    png = tmp_path / "figures" / "simple.png"
    png.write_bytes(b"\x89PNG")
    svg = tmp_path / "models" / "simple.svg"
    svg.write_text("<svg/>", encoding="utf-8")
    results = _create(tmp_path, extra=(extra,), files=(md, png, svg))
    assert list(results)[-4:] == [
        "simple_fluid",
        "simple_fluid_md",
        "simple_png",
        "simple_svg",
    ]
    assert results["simple_fluid_md"]["entry"].location == "./models/simple_fluid.md"
    assert results["simple_fluid_md"]["entry"].format == EntryFormat.MARKDOWN
    assert results["simple_png"]["entry"].location == "./figures/simple.png"
    assert results["simple_png"]["entry"].format == EntryFormat.PNG
    assert results["simple_svg"]["entry"].format == EntryFormat.SVG
    assert not any(
        results[k]["entry"].master for k in ("simple_fluid_md", "simple_png")
    )
    with zipfile.ZipFile(tmp_path / "simple_model.omex") as z:
        names = set(z.namelist())
    assert {"models/simple_fluid.md", "figures/simple.png"} <= names


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("simple_flat.md", "simple_flat_md"),
        ("simple.unknown", r"suffix '\.unknown'"),
    ],
    ids=["key of the archive", "unknown suffix"],
)
def test_an_extra_file_is_checked(tmp_path: Path, name: str, match: str) -> None:
    """An extra file with the key of an entry or a suffix of no format is an error."""
    (tmp_path / "models").mkdir()
    extra = tmp_path / "models" / name
    extra.write_text("x\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _create(tmp_path, files=(extra,))


def test_an_extra_file_outside_the_archive_is_an_error(tmp_path: Path) -> None:
    """A file which is not below the directory of the archive has no location."""
    outside = tmp_path.parent / f"{tmp_path.name}_outside.md"
    outside.write_text("x\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="not below"):
            _create(tmp_path, files=(outside,))
    finally:
        outside.unlink()


def test_an_extra_model_has_the_location_rule_of_an_extra_file(tmp_path: Path) -> None:
    """An extra model below the directory is located by its path, outside it is an error."""
    (tmp_path / "standalone").mkdir()
    extra = tmp_path / "standalone" / "simple_liver.xml"
    extra.write_text(MINIMAL_SBML, encoding="utf-8")
    results = _create(tmp_path, extra=(extra,))
    assert results["simple_liver"]["entry"].location == "./standalone/simple_liver.xml"
    with zipfile.ZipFile(tmp_path / "simple_model.omex") as z:
        assert "standalone/simple_liver.xml" in z.namelist()
    outside = tmp_path.parent / f"{tmp_path.name}_outside.xml"
    outside.write_text(MINIMAL_SBML, encoding="utf-8")
    try:
        with pytest.raises(ValueError, match=r"extra model .* not below"):
            _create(tmp_path, extra=(outside,))
    finally:
        outside.unlink()


def test_a_model_outside_the_archive_is_an_error(tmp_path: Path) -> None:
    (model,) = _models(tmp_path)[1:]
    with pytest.raises(ValueError, match=r"model .* not below"):
        write_archive([model], tmp_path / "other", name="x", copyright=COPYRIGHT)


def test_an_extra_antimony_file_is_antimony(tmp_path: Path) -> None:
    """The antimony of an extra model has the format of the antimony of the flat model."""
    (tmp_path / "models").mkdir()
    ant = tmp_path / "models" / "simple_fluid.ant"
    ant.write_text("model fluid()\nend\n", encoding="utf-8")
    results = _create(tmp_path, files=(ant,))
    assert results["simple_fluid_ant"]["entry"].format == EntryFormat.ANTIMONY


def test_the_written_files_have_the_location_rule_of_the_extra_files(
    tmp_path: Path,
) -> None:
    """Every file of the archive is located by its path below its directory.

    The models, their flat models, markdown and antimony and an extra file in
    the directory of the models share one rule, also when the directory is
    not named `models`; the README is the `README.md` of the archive.
    """
    readme = tmp_path / "README.md"
    readme.write_text("# simple\n", encoding="utf-8")
    models = _models(tmp_path, "sbml")
    extra = tmp_path / "sbml" / "notes.md"
    extra.write_text("notes\n", encoding="utf-8")
    results = write_archive(
        models[:1],
        tmp_path,
        name="simple_model",
        copyright=COPYRIGHT,
        readme=readme,
        extra_files=(extra,),
    )
    locations = {k: v["entry"].location for k, v in results.items()}
    assert locations == {
        "README": "./README.md",
        "cc-by": "./cc-by-sa-4.0.txt",
        "mit": "./mit.txt",
        "simple": "./sbml/simple.xml",
        "simple_flat": "./sbml/simple_flat.xml",
        "simple_flat_md": "./sbml/simple_flat.md",
        "simple_flat_ant": "./sbml/simple_flat.ant",
        "notes_md": "./sbml/notes.md",
    }
