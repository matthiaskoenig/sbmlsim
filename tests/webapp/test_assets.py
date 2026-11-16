"""The web assets a project writes with its factory, and the app of one SBML file."""

import json
import shutil
from collections.abc import Mapping
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


RULES = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="rules" name="rules only">
    <listOfParameters>
      <parameter id="a" name="the assigned" value="0" constant="false"/>
      <parameter id="x" name="the integrated" value="1" constant="false"/>
      <parameter id="k" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="a">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><cn>2</cn><ci>x</ci></apply>
        </math>
      </assignmentRule>
      <rateRule variable="x">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><apply><minus/><ci>k</ci></apply><ci>x</ci></apply>
        </math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>
"""


def test_a_model_without_species_shows_the_variables_of_its_rules(
    tmp_path: Path,
) -> None:
    sbml = tmp_path / "rules.xml"
    sbml.write_text(RULES, encoding="utf-8")
    panel = default_panel(sbml)
    assert [(curve.id, curve.label) for curve in panel.curves] == [
        ("x", "the integrated"),
        ("a", "the assigned"),
    ]
    assert panel.title == "rules only"


def test_a_model_without_species_and_rules_has_no_default_panel(
    tmp_path: Path,
) -> None:
    sbml = tmp_path / "constant.xml"
    sbml.write_text(
        RULES.split("<listOfRules>")[0] + "  </model>\n</sbml>\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="neither species nor rules"):
        default_panel(sbml)


def test_a_framework_gives_the_default_panels_of_its_models(tmp_path: Path) -> None:
    from sbmlsim.panel import Panel, PanelCurve, PanelSimulation
    from sbmlsim.project import Project
    from sbmlsim.webapp.assets import _with_default_panels

    class Framework(Project):
        def default_panels(self, sbml: Path) -> dict[str, Panel]:
            return {
                "framework": Panel(
                    id="framework",
                    title=sbml.stem,
                    curves=(PanelCurve(id="PX", label="X"),),
                    simulation=PanelSimulation(end=10),
                )
            }

    sbml = tmp_path / "model.xml"
    shutil.copy(REPRESSILATOR_SBML, sbml)
    project = Framework(name="model", root=tmp_path, variants={"model": sbml})
    shown = _with_default_panels(project, project)
    assert list(shown.panels) == ["framework"]
    assert shown.panels["framework"].title == "model"


def test_a_framework_gives_the_default_app_of_its_models(tmp_path: Path) -> None:
    from sbmlsim.panel import App, AppPage, Panel, PanelCurve, PanelSimulation
    from sbmlsim.project import Project
    from sbmlsim.webapp.assets import _with_default_panels

    class Framework(Project):
        def default_app(self, sbml: Path) -> tuple[dict[str, Panel], App | None]:
            panel = Panel(
                id="readout",
                title="readout",
                curves=(PanelCurve(id="PX", label="X"),),
                simulation=PanelSimulation(end=10),
            )
            page = AppPage(id="main", title="Main", panels=("readout",), size="small")
            return {"readout": panel}, App(title="Framework", pages=(page,))

    sbml = tmp_path / "model.xml"
    shutil.copy(REPRESSILATOR_SBML, sbml)
    project = Framework(name="model", root=tmp_path, variants={"model": sbml})
    shown = _with_default_panels(project, project)
    assert shown.app is not None
    assert shown.app.pages[0].size == "small"
    assert list(shown.panels) == ["readout"]


def test_a_framework_gives_the_panels_of_a_variant(tmp_path: Path) -> None:
    from sbmlsim.panel import App, AppPage, Panel, PanelCurve, PanelSimulation
    from sbmlsim.project import Project
    from sbmlsim.webapp.assets import _with_default_panels

    def panel(title: str) -> Panel:
        return Panel(
            id="readout",
            title=title,
            curves=(PanelCurve(id="PX", label="X"),),
            simulation=PanelSimulation(end=10),
        )

    class Framework(Project):
        def default_app(self, sbml: Path) -> tuple[dict[str, Panel], App | None]:
            page = AppPage(id="main", title="Main", panels=("readout",))
            return {"readout": panel("male")}, App(title="Framework", pages=(page,))

        def default_variant_panels(
            self, panels: Mapping[str, Panel], variants: Mapping[str, Path]
        ) -> dict[str, dict[str, Panel]]:
            assert list(variants) == ["male", "female"]
            return {"female": {"readout": panel("female")}}

    male, female = tmp_path / "male.xml", tmp_path / "female.xml"
    shutil.copy(REPRESSILATOR_SBML, male)
    shutil.copy(REPRESSILATOR_SBML, female)
    project = Framework(
        name="model", root=tmp_path, variants={"male": male, "female": female}
    )
    shown = _with_default_panels(project, project)
    assert shown.panels["readout"].title == "male"
    assert shown.variant_panels["female"]["readout"].title == "female"
    assert shown.panels_of("female")["readout"].title == "female"
