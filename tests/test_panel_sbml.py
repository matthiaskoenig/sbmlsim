"""The panel declaration in the annotation of the SBML model."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from sbmlsim.engine import JitOptimization
from sbmlsim.model.model_roadrunner import load_roadrunner
from sbmlsim.panel import (
    Choice,
    Knob,
    Panel,
    PanelCurve,
    PanelData,
    PanelSimulation,
)
from sbmlsim.panel_sbml import (
    PANEL_NS,
    PanelProblem,
    panels_from_sbml,
    panels_to_annotation,
    write_panels,
)
from sbmlsim.project import Project
from sbmlsim.resources import REPRESSILATOR_SBML

#: the model the panels are written into, a model of the package
SBML_IN = REPRESSILATOR_SBML

MODELS = Path(__file__).parent / "data" / "models"
#: two panels, the second one with a range knob without its bounds
BROKEN_SBML = MODELS / "panels_broken.xml"
#: an RDF annotation and a `panels` element of another namespace
FOREIGN_SBML = MODELS / "panels_foreign.xml"
#: two panels of the id `renal`
DUPLICATE_SBML = MODELS / "panels_duplicate.xml"

#: the panel of `BROKEN_SBML` which can be read, the documented XML shape
DOCUMENTED = Panel(
    id="ok",
    title="Renal impairment",
    subtitle="hctz, single oral dose",
    x_label="time [hr]",
    knobs=(
        Knob(
            id="dose",
            target="PODOSE_hctz",
            label="dose",
            unit="mg",
            kind="range",
            min=12.5,
            max=100.0,
            steps=9,
            default=25.0,
        ),
        Knob(
            id="impairment",
            target="f_renal_function",
            label="renal function",
            kind="choice",
            choices=(
                Choice(label="normal", value=1.0),
                Choice(label="severe", value=0.2),
            ),
            default=1.0,
        ),
    ),
    curves=(
        PanelCurve(id="Cve_hctz", label="hctz, plasma", unit="uM", scale="linear"),
    ),
    data=(
        PanelData(
            id="vormfelde2006",
            label="Vormfelde 2006",
            dataset="hctz_plasma.tsv",
            x="time",
            y="cp",
            yerr="cp_sd",
            unit="uM",
        ),
    ),
    simulation=PanelSimulation(
        variant="hctz_body",
        end=1440.0,
        points=200,
        x_unit="hr",
        preinit={"BW": 75.0},
    ),
)


@pytest.fixture
def panel() -> Panel:
    """A panel of the repressilator with every field set."""
    return Panel(
        id="oscillation",
        title="The repressilator",
        subtitle="the protein over the decay of its mRNA",
        x_label="time (min)",
        knobs=(
            Knob(
                id="beta",
                label="decay of the protein",
                target="beta",
                unit="dimensionless",
                kind="range",
                min=0.1,
                max=1.0,
                steps=5,
                default=0.2,
            ),
            Knob(
                id="n",
                label="hill coefficient",
                target="n",
                kind="choice",
                choices=(
                    Choice(label="two", value=2.0),
                    Choice(label="three", value=3.0),
                ),
                default=2.0,
            ),
        ),
        curves=(
            PanelCurve(id="[PX]", label="protein X", unit="dimensionless", scale="log"),
        ),
        data=(
            PanelData(
                id="elowitz2000",
                label="Elowitz 2000",
                dataset="elowitz2000.tsv",
                x="time",
                y="mean",
                yerr="mean_sd",
                unit="dimensionless",
            ),
        ),
        simulation=PanelSimulation(
            variant="default",
            end=100.0,
            points=200,
            x_unit="min",
            preinit={"beta": 0.2},
        ),
    )


@pytest.fixture
def minimal() -> Panel:
    """A panel of the required fields only, so the defaults round trip too."""
    return Panel(
        id="decay",
        title="decay",
        curves=(PanelCurve(id="[PX]", label="protein X"),),
        simulation=PanelSimulation(end=10.0),
    )


def test_a_panel_round_trips_through_the_annotation(
    tmp_path: Path, panel: Panel
) -> None:
    write_panels(SBML_IN, tmp_path / "m.xml", [panel])
    panels, problems = panels_from_sbml(tmp_path / "m.xml")
    assert problems == ()
    assert panels == {panel.id: panel}


def test_a_panel_of_the_defaults_round_trips_as_well(
    tmp_path: Path, panel: Panel, minimal: Panel
) -> None:
    write_panels(SBML_IN, tmp_path / "m.xml", [panel, minimal])
    panels, problems = panels_from_sbml(tmp_path / "m.xml")
    assert problems == ()
    assert panels == {panel.id: panel, minimal.id: minimal}
    assert list(panels) == [panel.id, minimal.id]


def test_the_sbml_itself_is_read_as_well(tmp_path: Path, panel: Panel) -> None:
    """A service reads the panels of the SBML of a model it has loaded."""
    write_panels(SBML_IN, tmp_path / "m.xml", [panel])
    assert panels_from_sbml((tmp_path / "m.xml").read_text(encoding="utf-8")) == (
        {panel.id: panel},
        (),
    )


def test_a_model_without_the_annotation_has_no_panels_and_no_problem() -> None:
    assert panels_from_sbml(REPRESSILATOR_SBML) == ({}, ())


def test_an_annotation_of_another_namespace_is_ignored() -> None:
    assert panels_from_sbml(FOREIGN_SBML) == ({}, ())


def test_a_malformed_panel_is_reported_by_id_and_the_others_are_read() -> None:
    panels, problems = panels_from_sbml(BROKEN_SBML)
    assert set(panels) == {"ok"}
    assert [problem.panel_id for problem in problems] == ["broken"]
    assert "bounds" in problems[0].reason


def test_the_documented_xml_shape_reads_into_the_panel_it_declares() -> None:
    """The XML of the documentation is the contract with the model factories."""
    panels, _ = panels_from_sbml(BROKEN_SBML)
    assert panels["ok"] == DOCUMENTED


def test_a_duplicate_panel_id_is_reported_and_neither_is_dropped_silently() -> None:
    panels, problems = panels_from_sbml(DUPLICATE_SBML)
    assert panels["renal"].title == "the first panel"
    assert [problem.panel_id for problem in problems] == ["renal"]
    assert "twice" in problems[0].reason


def test_writing_the_annotation_twice_replaces_it_instead_of_appending(
    tmp_path: Path, panel: Panel, minimal: Panel
) -> None:
    once, twice = tmp_path / "once.xml", tmp_path / "twice.xml"
    write_panels(SBML_IN, once, [panel])
    write_panels(once, twice, [minimal])
    panels, problems = panels_from_sbml(twice)
    assert problems == ()
    assert panels == {minimal.id: minimal}
    assert twice.read_text(encoding="utf-8").count(f'"{PANEL_NS}"') == 1


def test_writing_no_panel_removes_the_annotation(tmp_path: Path, panel: Panel) -> None:
    once, twice = tmp_path / "once.xml", tmp_path / "twice.xml"
    write_panels(SBML_IN, once, [panel])
    write_panels(once, twice, [])
    assert panels_from_sbml(twice) == ({}, ())
    assert PANEL_NS not in twice.read_text(encoding="utf-8")


def test_writing_the_annotation_keeps_the_rdf_of_the_model(
    tmp_path: Path, panel: Panel
) -> None:
    """The panels are a child of the annotation, the metadata stays."""
    out = tmp_path / "m.xml"
    write_panels(SBML_IN, out, [panel])
    written = out.read_text(encoding="utf-8")
    assert "http://www.w3.org/1999/02/22-rdf-syntax-ns#" in written
    assert "lenov@ebi.ac.uk" in written


def test_the_annotation_names_the_attributes_of_the_contract(panel: Panel) -> None:
    """`xLabel` and `xUnit` are the attributes which are not their field names."""
    annotation = panels_to_annotation([panel])
    assert annotation.startswith("<annotation>")
    assert f'<panels xmlns="{PANEL_NS}">' in annotation
    assert 'xLabel="time (min)"' in annotation
    assert 'xUnit="min"' in annotation
    assert "x_label" not in annotation
    assert "x_unit" not in annotation


def test_the_annotated_model_still_loads_in_roadrunner(
    tmp_path: Path, panel: Panel
) -> None:
    out = tmp_path / "m.xml"
    write_panels(SBML_IN, out, [panel])
    before = load_roadrunner(SBML_IN, JitOptimization.O0).simulate(0, 10, 11)
    after = load_roadrunner(out, JitOptimization.O0).simulate(0, 10, 11)
    assert after.colnames == before.colnames
    assert np.array_equal(np.asarray(after), np.asarray(before))


def test_a_path_which_is_no_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        panels_from_sbml(tmp_path / "does_not_exist.xml")


def test_a_file_which_is_no_model_is_an_error(tmp_path: Path) -> None:
    sbml = tmp_path / "m.xml"
    sbml.write_text("<not-an-sbml/>", encoding="utf-8")
    with pytest.raises(ValueError, match="SBML"):
        panels_from_sbml(sbml)


def _project(root: Path, panels: dict[str, Panel]) -> Project:
    """A project of one variant, the model of `root`."""
    return Project(
        name="panels",
        root=root,
        variants={"default": Path("m.xml")},
        panels=panels,
    )


def test_project_panels_override_the_annotation(
    tmp_path: Path, panel: Panel, minimal: Panel
) -> None:
    write_panels(SBML_IN, tmp_path / "m.xml", [panel, minimal])
    override = panel.model_copy(update={"title": "the panel of the project"})
    project = _project(tmp_path, {panel.id: override})
    assert project.panels_of() == {panel.id: override, minimal.id: minimal}
    assert project.panels_of("default")[panel.id].title == "the panel of the project"


def test_project_panels_of_a_model_without_the_annotation_are_the_declared_ones(
    tmp_path: Path, panel: Panel
) -> None:
    (tmp_path / "m.xml").write_bytes(SBML_IN.read_bytes())
    project = _project(tmp_path, {panel.id: panel})
    assert project.panels_of() == {panel.id: panel}


def test_the_annotated_panels_of_a_project_are_cached_outside_of_its_value(
    tmp_path: Path, panel: Panel
) -> None:
    write_panels(SBML_IN, tmp_path / "m.xml", [panel])
    project = _project(tmp_path, {})
    assert project.panels_of() == {panel.id: panel}
    (tmp_path / "m.xml").unlink()
    assert project.panels_of() == {panel.id: panel}
    assert project == _project(tmp_path, {})
    assert pickle.loads(pickle.dumps(project)) == project


def test_a_problem_of_the_annotation_of_a_project_is_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "m.xml").write_bytes(BROKEN_SBML.read_bytes())
    project = _project(tmp_path, {})
    assert set(project.panels_of()) == {"ok"}
    assert "broken" in caplog.text


def test_a_problem_is_a_value() -> None:
    assert PanelProblem("renal", "no bounds") == PanelProblem("renal", "no bounds")
    assert (
        len({PanelProblem("renal", "no bounds")} | {PanelProblem("renal", "no bounds")})
        == 1
    )
