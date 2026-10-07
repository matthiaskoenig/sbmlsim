"""The model is initialized with the pre-initialization values of a plan."""

import time

import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.model.symbols import ModelSymbols, TargetKind
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
    model.initialize([_a("f", 5.0), _a("b0", 0.0)])
    assert r["[B]"] == pytest.approx(0.0)
    assert r["pinit"] == pytest.approx(10.0)
    assert r["X"] == pytest.approx(30.0)
    assert r["kk"] == pytest.approx(15.0)
    model.initialize([])
    assert r["[B]"] == pytest.approx(1.0)
    assert r["pinit"] == pytest.approx(4.0)
    assert r["X"] == pytest.approx(12.0)
    assert r["f"] == pytest.approx(2.0)


def test_parameter_with_initial_assignment_can_be_set() -> None:
    """A parameter with an initial assignment is set before the initialization."""
    model = RoadrunnerSBMLModel(source=sbml())
    model.initialize([_a("pinit", 7.0)])
    assert model.r_loaded["pinit"] == pytest.approx(7.0)
    assert model.r_loaded["X"] == pytest.approx(21.0)
    model.initialize([_a("f", 3.0)])
    assert model.r_loaded["pinit"] == pytest.approx(6.0)
    assert model.r_loaded["X"] == pytest.approx(18.0)


def test_species_with_initial_assignment() -> None:
    """A species set before the initialization replaces its initial assignment."""
    model = RoadrunnerSBMLModel(source=sbml())
    model.initialize([_a("[B]", 5.0, TargetKind.SPECIES_CONCENTRATION), _a("b0", 0.0)])
    assert model.r_loaded["[B]"] == pytest.approx(5.0)
    model.initialize([_a("b0", 0.0)])
    assert model.r_loaded["[B]"] == pytest.approx(0.0)
    model.initialize([])
    assert model.r_loaded["[B]"] == pytest.approx(1.0)


def test_compartment_before_initialization() -> None:
    """A compartment before the initialization keeps the initial concentrations."""
    conc = sbml(
        "model c\n  compartment V = 1\n  species S in V = 3\n"
        "  substanceOnly species N in V\n  N = 2\nend"
    )
    model = RoadrunnerSBMLModel(source=conc)
    model.initialize([_a("V", 4.0, TargetKind.COMPARTMENT)])
    r = model.r_loaded
    assert r["V"] == pytest.approx(4.0)
    assert r["[S]"] == pytest.approx(3.0)
    assert r["S"] == pytest.approx(12.0)
    assert r["N"] == pytest.approx(2.0)


def test_initial_assignment_order() -> None:
    """The initial assignments are evaluated after the ones they read."""
    chain = sbml("model ch\n  a = 1\n  b = 2*a\n  c = b + 1\n  d = c * a\nend")
    symbols = ModelSymbols.from_sbml(chain)
    order = symbols.initial_assignment_order
    assert order.index("b") < order.index("c") < order.index("d")
    model = RoadrunnerSBMLModel(source=chain)
    model.initialize([_a("a", 3.0)])
    assert model.r_loaded["d"] == pytest.approx((2 * 3 + 1) * 3)


def test_helpers_are_not_selected() -> None:
    """The helper of an initial assignment is not in the default selections."""
    model = RoadrunnerSBMLModel(source=sbml())
    assert model.selections is not None
    assert "pinit__initial" not in model.selections
    assert model.initial_helpers["pinit"] == "pinit__initial"


def test_initialize_is_fast() -> None:
    """An initialization sets values and does not regenerate the model."""
    model = RoadrunnerSBMLModel(source=sbml())
    start = time.perf_counter()
    for _ in range(100):
        model.initialize([_a("f", 5.0), _a("b0", 0.0)])
    assert (time.perf_counter() - start) / 100 < 0.005
