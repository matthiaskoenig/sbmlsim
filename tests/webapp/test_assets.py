"""The web assets a project writes with its factory, and the app of one SBML file."""

import json
import shutil
from pathlib import Path

import libsbml
import pytest

from sbmlsim.panel import MAX_CURVES
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.webapp.__main__ import main
from sbmlsim.webapp.assets import DEFAULT_PANEL, app_project, default_panel
from sbmlsim.webapp.page import build_app

pytest.importorskip("sbmlsim_core")

from tests.project_toy import PROJECT as TOY


def _bundle_panels(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    start = text.index('id="sbmlsim-bundle">') + len('id="sbmlsim-bundle">')
    bundle = json.loads(text[start : text.index("</script>", start)])
    return list(bundle["answers"][""]["panels"])


def test_the_default_panel_shows_the_species_of_a_model() -> None:
    panel = default_panel(REPRESSILATOR_SBML)
    assert panel.id == DEFAULT_PANEL
    assert 0 < len(panel.curves) <= MAX_CURVES
    doc = libsbml.readSBMLFromFile(str(REPRESSILATOR_SBML))
    expected = []
    for species in doc.getModel().getListOfSpecies():
        sid = species.getId()
        expected.append(sid if species.getHasOnlySubstanceUnits() else f"[{sid}]")
    assert [curve.id for curve in panel.curves] == expected[:MAX_CURVES]


def test_a_model_without_panels_gets_an_app_of_its_species(tmp_path: Path) -> None:
    project = app_project(REPRESSILATOR_SBML)
    assert list(project.panels) == [DEFAULT_PANEL]
    assert list(build_app(project).sbml) == ["model"]


def test_the_command_line_writes_the_app_of_an_sbml_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sbml = tmp_path / "model.xml"
    shutil.copy(REPRESSILATOR_SBML, sbml)
    assert main([str(sbml), "-o", str(tmp_path / "out")]) == 0
    app = tmp_path / "out" / "model_app.html"
    assert capsys.readouterr().out.strip() == str(app)
    assert _bundle_panels(app) == [DEFAULT_PANEL]
    assert (
        '<sbmlsim-panel endpoint="local:model" model="model" panel="species" auto>'
        in (app.read_text(encoding="utf-8"))
    )


def test_the_command_line_refuses_a_file_which_is_no_model(tmp_path: Path) -> None:
    empty = tmp_path / "empty.xml"
    empty.write_text("<sbml/>", encoding="utf-8")
    assert main([str(empty)]) == 4


@pytest.fixture
def toy(tmp_path: Path):
    """The toy project in a copy of its package, so its factory writes into the copy."""
    assert TOY.factory_dir is not None
    root = tmp_path / "project_toy"
    shutil.copytree(TOY.root, root)
    variants = {
        name: root / path.relative_to(TOY.root) for name, path in TOY.variants.items()
    }
    return TOY.replace(
        root=root,
        variants=variants,
        factory_dir=root / TOY.factory_dir.relative_to(TOY.root),
    )


def test_the_factory_writes_the_compiled_variants_and_the_app(toy) -> None:
    directory = toy.create_models()
    for path in toy.variants.values():
        compiled = directory / "wasm" / path.stem
        assert (compiled / "model.wasm").read_bytes()[:4] == b"\0asm"
        assert (
            json.loads((compiled / "model.json").read_text(encoding="utf-8"))[
                "contract"
            ]
            == "model/2"
        )
    assert (directory / f"{toy.name}_app.html").exists()


def test_a_rebuild_writes_the_assets_of_the_models_it_built(
    toy, tmp_path: Path
) -> None:
    first = toy.create_models(tmp_path / "first")
    second = toy.create_models(tmp_path / "second")
    names = sorted(
        p.relative_to(first).as_posix() for p in first.rglob("*") if p.is_file()
    )
    assert names == sorted(
        p.relative_to(second).as_posix() for p in second.rglob("*") if p.is_file()
    )
    assert f"{toy.name}_app.html" in names
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    # the package is not written by a rebuild
    assert not (toy.factory_dir / f"{toy.name}_app.html").exists()


def test_a_project_without_the_web_app_writes_no_assets(toy) -> None:
    directory = toy.replace(web_app=False).create_models()
    assert not (directory / "wasm").exists()
    assert not (directory / f"{toy.name}_app.html").exists()
