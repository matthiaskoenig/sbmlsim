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


T0_EVENT = """
model events_at_t0
  P1 = 3/2
  E0: at 2.5 after (P1 > 1), t0=false, fromTrigger=false: P1 = P1^2
  E1: at 1.3 after (P1 > 1), t0=false: P1 = 5
end
"""


def _p1(model: RoadrunnerSBMLModel) -> list[float]:
    import numpy as np

    r = model.r_loaded
    r.timeCourseSelections = ["time", "P1"]
    values = np.array(r.simulate(0, 5, 6))[:, 1].tolist()
    model.simulated()
    return values


def test_events_at_t0_fire_once() -> None:
    """A reset without a simulation after it does not fire the events at t0 again.

    roadrunner queues the events which fire at the time 0 with every reset, a
    loaded model counts as one (case 01757 of the SBML Test Suite).
    """
    expected = [1.5, 1.5, 5.0, 25.0, 25.0, 25.0]
    model = RoadrunnerSBMLModel(source=sbml(T0_EVENT))
    model.initialize([])
    assert _p1(model) == pytest.approx(expected)
    model.initialize([])
    model.initialize([])
    assert _p1(model) == pytest.approx(expected)


def test_values_of_an_initialization_without_simulation_are_restored() -> None:
    """Two initializations without a simulation do not leak into each other."""
    model = RoadrunnerSBMLModel(source=sbml())
    model.initialize(
        [_a("f", 5.0), _a("b0", 0.0), _a("C", 4.0, TargetKind.COMPARTMENT)]
    )
    model.initialize([_a("k1", 0.1)])
    r = model.r_loaded
    assert r["f"] == pytest.approx(2.0)
    assert r["X"] == pytest.approx(12.0)
    assert r["[B]"] == pytest.approx(1.0)
    assert r["C"] == pytest.approx(2.0)
    assert r["[A]"] == pytest.approx(1.0)
    assert r["k1"] == pytest.approx(0.1)
