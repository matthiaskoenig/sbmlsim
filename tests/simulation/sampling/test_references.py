"""The references of the targets of a design."""

import pytest

from sbmlsim import Q
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Simulation
from sbmlsim.simulation.sampling import parameters_of, references
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml, sbml_minutes


def test_the_references_are_the_values_after_the_preinitialization() -> None:
    model = Simulator().load(sbml())
    refs = references(model, ["k1", "pinit", "X"])
    assert refs == {"k1": 0.8, "pinit": 4.0, "X": 12.0}
    changed = references(
        model, ["k1", "pinit"], Simulation(end=1, preinit_changes={"f": 5.0, "k1": 2.0})
    )
    # pinit = 2 f is an initial assignment of the changed f
    assert changed == {"k1": 2.0, "pinit": 10.0}


def test_the_changes_of_the_model_are_defaults() -> None:
    model = RoadrunnerSBMLModel(source=sbml(), changes={"f": 3.0})
    assert references(model, ["pinit"])["pinit"] == pytest.approx(6.0)


def test_a_reference_has_the_unit_of_its_target() -> None:
    refs = references(Simulator().load(sbml_minutes()), ["f"])
    assert refs["f"] == Q(2.0, "mg")


def test_an_unknown_target_raises() -> None:
    with pytest.raises(ValueError, match="'nope'"):
        references(Simulator().load(sbml()), ["nope"])


def test_the_parameters_of_a_model() -> None:
    model = Simulator().load(sbml())
    parameters = parameters_of(model)
    assert {"a0", "b0", "k1", "k2", "f"} <= set(parameters)
    assert not any(p.endswith("__initial") for p in parameters)
    assert "k1" not in parameters_of(model, exclude={"k1"})
    assert "k1" not in parameters_of(model, exclude=lambda pid: pid.startswith("k"))
    assert {"A", "B", "X"} <= set(parameters_of(model, species=True))
    zero = RoadrunnerSBMLModel(source=sbml(), changes={"k2": 0.0})
    assert "k2" not in parameters_of(zero)
    assert "k2" in parameters_of(zero, exclude_zero=False)


def test_a_model_path_is_loaded() -> None:
    assert references(sbml(), ["k1"]) == {"k1": 0.8}
