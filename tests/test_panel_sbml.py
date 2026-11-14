"""The panel declaration in the annotation of the SBML model."""

import pickle
import re
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
    AT_ELEMENT,
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
    y_label="plasma concentration",
    y_max=2.0,
    x_max=24.0,
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
            id="crcl",
            label="creatinine clearance",
            unit="ml/min",
            kind="range",
            min=10.0,
            max=150.0,
            steps=8,
            choices=(
                Choice(label="normal", value=110.0),
                Choice(label="severe impairment", value=20.0),
            ),
            default=110.0,
        ),
    ),
    curves=(
        PanelCurve(id="[Cve_hctz]", label="hctz, plasma", unit="uM", scale="linear"),
    ),
    data=(
        PanelData(
            id="vormfelde2006",
            label="Vormfelde 2006",
            curve="[Cve_hctz]",
            dataset="Vormfelde2006_Fig2",
            x="time",
            y="mean",
            yerr="mean_sd",
            sel={"label": "hctz25"},
            at={"crcl": 110.0},
            conversion="1/(297.7 g/mole)",
        ),
    ),
    simulation=PanelSimulation(
        variant="hctz_body",
        end=1440.0,
        points=200,
        x_unit="hr",
        preinit={"BW": 75.0},
        derived={"f_renal_function": "min(crcl / 110, 1)"},
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
                curve="[PX]",
                dataset="Elowitz2000_Fig1",
                x="time",
                y="mean",
                yerr="mean_sd",
                sel={"label": "px", "individual": "S1"},
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


def test_a_range_knob_with_named_stops_round_trips_through_the_annotation(
    tmp_path: Path, panel: Panel
) -> None:
    """The stops of a range knob are its `option` children, as the choices are."""
    beta = panel.knobs[0].model_copy(
        update={
            "choices": (
                Choice(label="slow", value=0.1),
                Choice(label="fast", value=1.0),
            )
        }
    )
    stopped = panel.model_copy(update={"knobs": (beta, panel.knobs[1])})
    assert '<option label="slow" value="0.1"/>' in panels_to_annotation([stopped])
    write_panels(SBML_IN, tmp_path / "m.xml", [stopped])
    panels, problems = panels_from_sbml(tmp_path / "m.xml")
    assert problems == ()
    assert panels == {stopped.id: stopped}
    assert panels[stopped.id].knobs[0].kind == "range"


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


def test_the_documented_panel_round_trips_through_the_writer(tmp_path: Path) -> None:
    """The curve, the selection, the state and the conversion are written too."""
    annotation = panels_to_annotation([DOCUMENTED])
    assert '<select column="label" value="hctz25"/>' in annotation
    assert '<at knob="crcl" value="110.0"/>' in annotation
    assert (
        '<derived target="f_renal_function" formula="min(crcl / 110, 1)"/>'
        in annotation
    )
    assert 'curve="[Cve_hctz]"' in annotation
    assert 'conversion="1/(297.7 g/mole)"' in annotation
    write_panels(SBML_IN, tmp_path / "m.xml", [DOCUMENTED])
    assert panels_from_sbml(tmp_path / "m.xml") == ({DOCUMENTED.id: DOCUMENTED}, ())


def test_the_state_of_a_series_round_trips_through_the_annotation(
    tmp_path: Path,
) -> None:
    """A series names the state of two knobs it was measured under."""
    knobs = (
        Knob(
            id="dose",
            label="dose",
            target="PODOSE_hctz",
            unit="mg",
            kind="choice",
            choices=(
                Choice(label="25 mg", value=25.0),
                Choice(label="50 mg", value=50.0),
            ),
            default=25.0,
        ),
        Knob(
            id="impairment",
            label="renal function",
            target="f_renal_function",
            kind="choice",
            choices=(
                Choice(label="normal", value=1.0),
                Choice(label="severe", value=0.08),
            ),
            default=1.0,
        ),
    )
    measured = Panel(
        id="renal",
        title="Renal impairment",
        knobs=knobs,
        curves=(PanelCurve(id="[PX]", label="protein X"),),
        data=(
            PanelData(
                id="study2000",
                label="Study 2000",
                curve="[PX]",
                dataset="Study2000_Fig1",
                x="time",
                y="mean",
                at={"impairment": 0.08, "dose": 50.0},
            ),
        ),
        simulation=PanelSimulation(end=10.0),
    )
    annotation = panels_to_annotation([measured])
    assert '<at knob="impairment" value="0.08"/>' in annotation
    assert '<at knob="dose" value="50.0"/>' in annotation
    write_panels(SBML_IN, tmp_path / "m.xml", [measured])
    assert panels_from_sbml(tmp_path / "m.xml") == ({measured.id: measured}, ())


def test_a_series_without_a_state_reads_back_as_none(
    tmp_path: Path, panel: Panel
) -> None:
    """A series which was measured at no state writes no child, as `sel` does."""
    without = panel.data[0].model_copy(update={"sel": None})
    assert without.at is None
    written = panel.model_copy(update={"data": (without,)})
    write_panels(SBML_IN, tmp_path / "m.xml", [written])
    assert f"<{AT_ELEMENT} " not in (tmp_path / "m.xml").read_text(encoding="utf-8")
    panels, problems = panels_from_sbml(tmp_path / "m.xml")
    assert problems == ()
    assert panels[written.id].data[0].at is None
    assert panels[written.id].data[0].sel is None


def test_an_at_without_a_knob_is_reported_as_a_problem(tmp_path: Path) -> None:
    """A malformed child is a problem of its panel, as a malformed `select` is."""
    out = tmp_path / "m.xml"
    write_panels(SBML_IN, out, [DOCUMENTED])
    written = out.read_text(encoding="utf-8")
    assert '<at knob="crcl" value="110.0"/>' in written
    broken = written.replace('<at knob="crcl" value="110.0"/>', "<at/>")
    panels, problems = panels_from_sbml(broken)
    assert panels == {}
    assert [problem.panel_id for problem in problems] == [DOCUMENTED.id]
    assert "knob" in problems[0].reason
    assert "vormfelde2006" in problems[0].reason


@pytest.mark.parametrize(
    ("element", "match"),
    [
        ('<derived formula="min(crcl / 110, 1)"/>', "'' is no SId"),
        ('<derived target="f_renal_function"/>', "needs its formula"),
    ],
)
def test_a_derived_target_without_its_target_or_formula_is_a_problem(
    tmp_path: Path, element: str, match: str
) -> None:
    out = tmp_path / "m.xml"
    write_panels(SBML_IN, out, [DOCUMENTED])
    written = out.read_text(encoding="utf-8")
    derived = '<derived target="f_renal_function" formula="min(crcl / 110, 1)"/>'
    assert derived in written
    panels, problems = panels_from_sbml(written.replace(derived, element))
    assert panels == {}
    assert [problem.panel_id for problem in problems] == [DOCUMENTED.id]
    assert re.search(match, problems[0].reason)


def test_derived_targets_stops_a_targetless_knob_and_limits_round_trip(
    tmp_path: Path, panel: Panel
) -> None:
    """Every field of the extension of the contract is written and read back."""
    rate = Knob(
        id="rate",
        label="rate of decay",
        unit="1/min",
        kind="range",
        min=0.5,
        max=4.0,
        steps=4,
        choices=(Choice(label="slow", value=0.5), Choice(label="fast", value=4.0)),
        default=1.0,
    )
    extended = panel.model_copy(
        update={
            "knobs": (*panel.knobs, rate),
            "y_min": 1.0,
            "y_max": 5000.0,
            "x_max": 60.0,
            "simulation": panel.simulation.model_copy(
                update={"derived": {"alpha": "rate * 100", "alpha0": "rate / 1000"}}
            ),
        }
    )
    annotation = panels_to_annotation([extended])
    assert '<derived target="alpha" formula="rate * 100"/>' in annotation
    assert '<knob id="rate" label="rate of decay" unit="1/min"' in annotation
    write_panels(SBML_IN, tmp_path / "m.xml", [extended])
    panels, problems = panels_from_sbml(tmp_path / "m.xml")
    assert problems == ()
    assert panels == {extended.id: extended}
    read = panels[extended.id]
    assert read.knobs[-1].target is None
    assert read.simulation.derived == {"alpha": "rate * 100", "alpha0": "rate / 1000"}


def test_the_label_of_the_value_axis_round_trips_and_is_empty_without_one(
    tmp_path: Path, panel: Panel, minimal: Panel
) -> None:
    """`yLabel` is the attribute of `y_label`, as `xLabel` is of `x_label`."""
    assert minimal.y_label == ""
    labelled = panel.model_copy(update={"y_label": "protein X"})
    annotation = panels_to_annotation([labelled])
    assert 'yLabel="protein X"' in annotation
    assert "y_label" not in annotation
    write_panels(SBML_IN, tmp_path / "m.xml", [labelled, minimal])
    panels, problems = panels_from_sbml(tmp_path / "m.xml")
    assert problems == ()
    assert panels == {labelled.id: labelled, minimal.id: minimal}
    assert panels[minimal.id].y_label == ""


def test_a_select_without_a_column_is_reported_as_a_problem(tmp_path: Path) -> None:
    """A malformed child is a problem of its panel, as a malformed knob is."""
    out = tmp_path / "m.xml"
    write_panels(SBML_IN, out, [DOCUMENTED])
    written = out.read_text(encoding="utf-8")
    assert '<select column="label" value="hctz25"/>' in written
    broken = written.replace('<select column="label" value="hctz25"/>', "<select/>")
    panels, problems = panels_from_sbml(broken)
    assert panels == {}
    assert [problem.panel_id for problem in problems] == [DOCUMENTED.id]
    assert "column" in problems[0].reason
    assert "vormfelde2006" in problems[0].reason


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


def test_the_limits_of_the_axes_round_trip_through_the_annotation(
    tmp_path: Path, panel: Panel
) -> None:
    limited = panel.model_copy(update={"y_min": 0.01, "y_max": 50.0, "x_max": 80.0})
    annotation = panels_to_annotation([limited])
    assert 'yMin="0.01" yMax="50.0" xMax="80.0"' in annotation
    assert "y_min" not in annotation
    write_panels(SBML_IN, tmp_path / "m.xml", [limited])
    assert panels_from_sbml(tmp_path / "m.xml") == ({limited.id: limited}, ())
    # an axis without a limit is no attribute and reads back as none
    assert "yMin" not in panels_to_annotation([panel])


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


def test_the_panels_of_a_variant_are_laid_over_the_ones_of_every_variant(
    tmp_path: Path, panel: Panel, minimal: Panel
) -> None:
    """The declaration of a panel in a variant is the one of its SBML and overrides.

    The SBML of `default` carries the annotation and the one of `other` none;
    `panels` is the override of every variant and `variant_panels` the one of
    a variant, which wins.
    """
    write_panels(SBML_IN, tmp_path / "m.xml", [panel, minimal])
    (tmp_path / "other.xml").write_bytes(SBML_IN.read_bytes())
    every = minimal.model_copy(update={"title": "of every variant"})
    other = panel.model_copy(update={"title": "of the variant other"})
    only = minimal.model_copy(update={"id": "only"})
    project = Project(
        name="panels",
        root=tmp_path,
        variants={"default": Path("m.xml"), "other": Path("other.xml")},
        panels={minimal.id: every},
        variant_panels={"other": {panel.id: other, only.id: only}},
    )
    assert project.panels_of() == {panel.id: panel, minimal.id: every}
    assert project.panels_of("default") == project.panels_of()
    assert project.panels_of("other") == {
        minimal.id: every,
        panel.id: other,
        "only": only,
    }
    assert list(project.panels_of("other")) == [minimal.id, panel.id, "only"]
    # the default variant by its name takes its own overrides as well
    named = project.replace(variant_panels={"default": {only.id: only}})
    assert set(named.panels_of()) == {panel.id, minimal.id, "only"}
    assert set(named.panels_of("other")) == {minimal.id}


def test_the_panels_of_a_variant_the_project_has_not_are_refused(
    tmp_path: Path, panel: Panel
) -> None:
    with pytest.raises(ValueError, match=r"variants \['female'\], which it does not"):
        Project(
            name="panels",
            root=tmp_path,
            variants={"default": Path("m.xml")},
            variant_panels={"female": {panel.id: panel}},
        )


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
