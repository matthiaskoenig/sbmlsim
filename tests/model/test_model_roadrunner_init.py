"""The model sets the initial values of a plan and restores what it does not set."""

import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.model.symbols import TargetKind
from sbmlsim.simulator.plan import Assignment
from tests.simulator.models import sbml


def _a(
    target: str, value: float, kind: TargetKind = TargetKind.PARAMETER
) -> Assignment:
    return Assignment(target=target, kind=kind, value=value)


def test_a_model_is_loaded_from_sbml() -> None:
    """The source of a model is a path or the SBML itself."""
    model = RoadrunnerSBMLModel(source=sbml())
    assert model.r_loaded["[B]"] == pytest.approx(1.0)
    assert model.symbols.kind("[B]") is TargetKind.SPECIES_CONCENTRATION


def test_preinit_reaches_initial_assignments_and_is_restored() -> None:
    """A value before the initialization reaches the initial assignments."""
    model = RoadrunnerSBMLModel(source=sbml())
    r = model.r_loaded
    model.set_initial_values([_a("f", 5.0), _a("b0", 0.0)])
    r.reset()
    assert r["[B]"] == pytest.approx(0.0)
    assert r["pinit"] == pytest.approx(10.0)
    assert r["X"] == pytest.approx(30.0)
    model.set_initial_values([])
    r.reset()
    assert r["[B]"] == pytest.approx(1.0)
    assert r["pinit"] == pytest.approx(4.0)
    assert r["X"] == pytest.approx(12.0)


def test_parameter_with_initial_assignment_can_be_set() -> None:
    """A parameter with an initial assignment is set before the initialization."""
    model = RoadrunnerSBMLModel(source=sbml())
    model.free_initial_assignments({"pinit"})
    model.set_initial_values([_a("pinit", 7.0)])
    model.r_loaded.reset()
    assert model.r_loaded["pinit"] == pytest.approx(7.0)
    assert model.r_loaded["X"] == pytest.approx(21.0)
    model.set_initial_values([_a("f", 3.0)])
    model.r_loaded.reset()
    assert model.r_loaded["pinit"] == pytest.approx(6.0)
    assert model.r_loaded["X"] == pytest.approx(18.0)


def test_species_with_initial_assignment_is_restored() -> None:
    """A species whose initial assignment was replaced gets it back."""
    model = RoadrunnerSBMLModel(source=sbml())
    model.free_initial_assignments({"B"})
    model.set_initial_values(
        [_a("[B]", 5.0, TargetKind.SPECIES_CONCENTRATION), _a("b0", 0.0)]
    )
    model.r_loaded.reset()
    assert model.r_loaded["[B]"] == pytest.approx(5.0)
    model.set_initial_values([_a("b0", 0.0)])
    model.r_loaded.reset()
    assert model.r_loaded["[B]"] == pytest.approx(0.0)
    model.set_initial_values([])
    model.r_loaded.reset()
    assert model.r_loaded["[B]"] == pytest.approx(1.0)


def test_derived_helpers_are_not_selected() -> None:
    """The helper of a freed initial assignment is not in the default selections."""
    model = RoadrunnerSBMLModel(source=sbml())
    model.free_initial_assignments({"pinit"})
    assert model.derived_initial == {"pinit": "pinit__initial"}
    assert model.selections is not None
    assert "pinit__initial" not in model.selections
