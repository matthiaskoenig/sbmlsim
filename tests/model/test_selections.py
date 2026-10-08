"""The default selections of a model are the entities which have a value."""

import libsbml

from sbmlsim.model import RoadrunnerSBMLModel
from tests.simulator.models import sbml


def _sbml_with_membrane() -> str:
    """Get the probe model with a membrane whose area the model does not use."""
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml())
    model: libsbml.Model = doc.getModel()
    membrane: libsbml.Compartment = model.createCompartment()
    membrane.setId("M")
    membrane.setSpatialDimensions(2)
    membrane.setConstant(True)
    membrane.setSize(float("nan"))
    return libsbml.writeSBMLToString(doc)


def test_a_compartment_without_a_size_is_not_selected() -> None:
    """A compartment whose size is `NaN` has no value to record."""
    model = RoadrunnerSBMLModel(source=_sbml_with_membrane())
    assert model.selections is not None
    assert "C" in model.selections
    assert "M" not in model.selections


def test_a_compartment_without_a_size_can_be_selected() -> None:
    """Selections which are given are kept as they are."""
    model = RoadrunnerSBMLModel(source=_sbml_with_membrane())
    selections = RoadrunnerSBMLModel.set_timecourse_selections(
        model.r_loaded, selections=["time", "M"]
    )
    assert selections == ["time", "M"]
