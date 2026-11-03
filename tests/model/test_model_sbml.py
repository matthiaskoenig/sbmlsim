"""The model independent of its engine and the loading of a model of an engine."""

import logging
from pathlib import Path

import libsbml
import pytest

from sbmlsim.engine import Engine
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel, SBMLModel, load_model
from sbmlsim.model.symbols import EventSymbols
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.resources import DEMO_SBML, HCTZ_SBML, MIDAZOLAM_SBML, REPRESSILATOR_SBML
from sbmlsim.simulator.changes import triggers
from sbmlsim.units import Quantity
from tests.simulator.models import sbml as probe_sbml


def _with_membrane() -> str:
    """Get the probe model with a compartment without a size."""
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(probe_sbml())
    model: libsbml.Model = doc.getModel()
    membrane: libsbml.Compartment = model.createCompartment()
    membrane.setId("M")
    membrane.setSpatialDimensions(2)
    membrane.setConstant(True)
    membrane.setSize(float("nan"))
    return libsbml.writeSBMLToString(doc)


@pytest.mark.parametrize(
    "sbml", [REPRESSILATOR_SBML, DEMO_SBML, MIDAZOLAM_SBML, HCTZ_SBML]
)
def test_default_selections_are_the_ones_of_roadrunner(sbml: Path) -> None:
    """The selections from the SBML are the ones roadrunner selects by default."""
    model = RoadrunnerSBMLModel(source=sbml)
    assert model.selections is not None
    assert set(model.default_selections()) == set(model.selections)
    assert len(model.default_selections()) == len(model.selections)


def test_default_selections_without_a_compartment_without_a_size() -> None:
    """A compartment without a size has no value and is not selected."""
    model = RoadrunnerSBMLModel(source=_with_membrane())
    assert "M" not in model.default_selections()
    assert model.selections is not None
    assert set(model.default_selections()) == set(model.selections)


def test_default_selections_without_the_added_parameters() -> None:
    """The parameters added to a model are not selected by default."""
    model = RoadrunnerSBMLModel(source=probe_sbml(), parameters={"added": 2.0})
    assert "added" not in model.default_selections()
    assert model.selections is not None
    assert set(model.default_selections()) == set(model.selections)


def test_load_model_of_roadrunner_is_a_roadrunner_model() -> None:
    """The roadrunner engine loads a `RoadrunnerSBMLModel`."""
    model = load_model(AbstractModel(source=DEMO_SBML), Engine.ROADRUNNER)
    assert isinstance(model, RoadrunnerSBMLModel) and isinstance(model, SBMLModel)
    assert model.engine is Engine.ROADRUNNER


def test_load_model_of_a_path_and_of_the_model_of_the_engine() -> None:
    """A path is loaded, a model of the engine is the model itself."""
    model = load_model(DEMO_SBML, Engine.ROADRUNNER, jit_optimization="O0")
    assert isinstance(model, RoadrunnerSBMLModel)
    assert model.jit_optimization == "O0"
    assert load_model(model, Engine.ROADRUNNER) is model


def test_load_model_keeps_the_description() -> None:
    """Changes, parameters, selections and the id of a description are loaded."""
    description = AbstractModel(
        source=probe_sbml(),
        sid="probe",
        changes={"k1": 2.0},
        selections=["time", "A"],
        parameters={"added": 3.0},
    )
    model = load_model(description, Engine.ROADRUNNER)
    assert model.sid == "probe"
    assert model.selections == ["time", "A"]
    assert model.parameters == {"added": 3.0}
    change = model.changes["k1"]
    assert isinstance(change, Quantity) and change.magnitude == 2.0


def test_the_label_is_the_id_or_the_id_of_the_sbml_model() -> None:
    """A model is named by its id, else by the id of its SBML model."""
    assert RoadrunnerSBMLModel(source=REPRESSILATOR_SBML, sid="r").label == "r"
    model = RoadrunnerSBMLModel(source=REPRESSILATOR_SBML)
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(REPRESSILATOR_SBML))
    assert model.label == doc.getModel().getId()


def test_the_value_of_a_selection() -> None:
    """`value` is the current value, `has_selection` whether there is one."""
    model = RoadrunnerSBMLModel(source=probe_sbml())
    assert model.has_selection("k1")
    assert not model.has_selection("no_entity")
    assert model.value("k1") == model.loaded_value("k1")


def test_the_integrator_state_has_the_absolute_tolerance_of_the_model() -> None:
    """The settings of the integrator, the absolute tolerance as given."""
    model = RoadrunnerSBMLModel(source=probe_sbml())
    tolerance = AbsoluteTolerance(amount=1e-9)
    model.set_integrator_settings(absolute_tolerance=tolerance, relative_tolerance=1e-8)
    name, settings = model.integrator_state()
    assert name == "cvode"
    values = dict(settings)
    assert values["absolute_tolerance"] == tolerance
    assert values["relative_tolerance"] == 1e-8
    assert [k for k, _ in settings] == sorted(values)


def test_a_message_of_the_events_names_the_sbml_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A model without an id is named by the id of its SBML model."""
    math = libsbml.writeMathMLToString(libsbml.parseL3Formula("rateOf(A) > 0"))
    event = EventSymbols(eid="R", trigger=math, delayed=False, assignments=())
    model = RoadrunnerSBMLModel(source=probe_sbml())
    with caplog.at_level(logging.WARNING, logger="sbmlsim.simulator.changes"):
        assert triggers([event], model.r_loaded, 0.0, model) == [None]
    assert "of the model 'probe'" in caplog.records[0].getMessage()
