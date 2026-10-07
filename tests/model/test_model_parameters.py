"""Parameters which a problem adds to a model, e.g. of the parameter table of PEtab."""

import pytest

from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.model.symbols import TargetKind
from sbmlsim.simulator.plan import Assignment
from tests.simulator.models import sbml


def test_added_parameters_are_parameters_of_the_model() -> None:
    """An added parameter has its value, is a target and is not selected by default."""
    model = RoadrunnerSBMLModel(source=sbml(), parameters={"scale": 2.0})
    assert model.r_loaded["scale"] == pytest.approx(2.0)
    assert model.symbols.kind("scale") is TargetKind.PARAMETER
    assert model.selections is not None
    assert "scale" not in model.selections
    model.initialize([Assignment("scale", TargetKind.PARAMETER, value=3.0)])
    assert model.r_loaded["scale"] == pytest.approx(3.0)


def test_an_added_parameter_must_be_new() -> None:
    """An id of an entity of the model is not added."""
    with pytest.raises(ValueError, match="'k1'"):
        RoadrunnerSBMLModel(source=sbml(), parameters={"k1": 1.0})


def test_the_abstract_model_carries_the_parameters() -> None:
    """The parameters of an abstract model reach the loaded model."""
    abstract = AbstractModel(source=sbml(), parameters={"offset": 0.5})
    model = RoadrunnerSBMLModel.from_abstract_model(abstract)
    assert model.r_loaded["offset"] == pytest.approx(0.5)
