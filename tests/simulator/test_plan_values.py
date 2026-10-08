"""A point of a scan is a plan with other values, see `Plan.with_values`."""

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Change, Simulation, SteadyState
from sbmlsim.simulator.executor import execute
from sbmlsim.simulator.plan import Plan, compile_simulation, model_time, target_values
from tests.simulator.models import sbml, sbml_minutes

SEL = ["time", "[A]", "[B]", "X", "C", "k1"]


@pytest.fixture
def model() -> RoadrunnerSBMLModel:
    return RoadrunnerSBMLModel(source=sbml())


def _plan(model: RoadrunnerSBMLModel, simulation: Simulation) -> Plan:
    return compile_simulation(simulation, model.symbols, model.uinfo)


def _same(model: RoadrunnerSBMLModel, plan: Plan, simulation: Simulation) -> None:
    """The plan gives the values of the compiled simulation."""
    expected = execute(_plan(model, simulation), model, SEL).values
    np.testing.assert_allclose(execute(plan, model, SEL).values, expected, rtol=1e-12)


def test_a_value_before_the_initialization(model: RoadrunnerSBMLModel) -> None:
    plan = _plan(model, Simulation(end=2, steps=4)).with_values({"a0": 3.0})
    _same(model, plan, Simulation(end=2, steps=4, preinit_changes={"a0": 3.0}))


def test_a_value_replaces_every_change_of_its_target(
    model: RoadrunnerSBMLModel,
) -> None:
    base = Simulation(end=2, steps=4, changes=[Change([0.5, 1.5], {"k1": 0.1})])
    plan = _plan(model, base).with_values({"k1": 2.0})
    _same(
        model,
        plan,
        Simulation(end=2, steps=4, changes=[Change([0.5, 1.5], {"k1": 2.0})]),
    )


def test_a_value_at_a_time_is_a_change(model: RoadrunnerSBMLModel) -> None:
    plan = _plan(model, Simulation(end=2, steps=4)).with_values({"k1": 2.0}, at=1.0)
    assert [event.time for event in plan.events] == [1.0]
    _same(model, plan, Simulation(end=2, steps=4, changes=[Change(1.0, {"k1": 2.0})]))


def test_a_value_at_the_time_of_a_change_merges_into_it(
    model: RoadrunnerSBMLModel,
) -> None:
    base = Simulation(end=2, steps=4, changes=[Change(1.0, {"k1": 0.5, "k2": 0.1})])
    plan = _plan(model, base).with_values({"k1": 2.0}, at=1.0)
    assert len(plan.events) == 1
    assert {a.target: a.value for a in plan.events[0].assignments} == {
        "k1": 2.0,
        "k2": 0.1,
    }
    _same(
        model,
        plan,
        Simulation(end=2, steps=4, changes=[Change(1.0, {"k1": 2.0, "k2": 0.1})]),
    )


def test_the_events_stay_sorted(model: RoadrunnerSBMLModel) -> None:
    base = Simulation(
        end=2, changes=[Change(0.5, {"k2": 0.1}), Change(1.5, {"k2": 0.2})]
    )
    plan = _plan(model, base).with_values({"k1": 2.0}, at=1.0)
    assert [event.time for event in plan.events] == [0.5, 1.0, 1.5]


def test_a_value_of_the_presimulation(model: RoadrunnerSBMLModel) -> None:
    def simulation(k1: float) -> Simulation:
        return Simulation(
            end=2, steps=4, presimulation=SteadyState(preinit_changes={"k1": k1})
        )

    _same(
        model, _plan(model, simulation(0.5)).with_values({"k1": 2.0}), simulation(2.0)
    )


def test_a_compartment_at_a_time(model: RoadrunnerSBMLModel) -> None:
    plan = _plan(model, Simulation(end=2, steps=4)).with_values({"C": 4.0}, at=1.0)
    _same(model, plan, Simulation(end=2, steps=4, changes=[Change(1.0, {"C": 4.0})]))


def test_a_time_outside_of_the_simulation_is_an_error(
    model: RoadrunnerSBMLModel,
) -> None:
    with pytest.raises(ValueError, match="outside of the simulation"):
        _plan(model, Simulation(end=2)).with_values({"k1": 2.0}, at=3.0)


def test_a_value_at_a_time_of_no_target_is_an_error(model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="nope"):
        _plan(model, Simulation(end=2)).with_values({"nope": 2.0}, at=1.0)


def test_no_values_are_the_plan(model: RoadrunnerSBMLModel) -> None:
    plan = _plan(model, Simulation(end=2))
    assert plan.with_values({}, at=1.0) is plan


def test_the_output_times(model: RoadrunnerSBMLModel) -> None:
    assert _plan(model, Simulation(end=2)).output_times() is None
    shifted = Simulation(end=2, steps=2, time_shift=1.0)
    assert _plan(model, shifted).output_times() == (1.0, 2.0, 3.0)
    steady = Simulation(end=2, times=[0, 2, np.inf])
    assert _plan(model, steady).output_times() == (0.0, 2.0, np.inf)


def test_the_values_of_a_target_in_the_unit_of_the_model() -> None:
    model = RoadrunnerSBMLModel(source=sbml_minutes())
    np.testing.assert_allclose(
        target_values("f", Q([1.0, 2.0], "g"), model.symbols, model.uinfo),
        [1000.0, 2000.0],
    )
    np.testing.assert_array_equal(
        target_values("f", [1.0, 2.0], model.symbols, model.uinfo), [1.0, 2.0]
    )
    with pytest.raises(ValueError, match="cannot be converted"):
        target_values("f", Q([1.0], "mmol"), model.symbols, model.uinfo)
    with pytest.raises(ValueError, match="nope"):
        target_values("nope", [1.0], model.symbols, model.uinfo)


def test_a_time_in_the_time_unit_of_the_model() -> None:
    model = RoadrunnerSBMLModel(source=sbml_minutes())
    hours = Simulation(time_unit="hr", end=2)
    assert model_time(hours, 1.0, model.symbols, model.uinfo) == pytest.approx(60.0)
    assert model_time(hours, Q(30, "s"), model.symbols, model.uinfo) == pytest.approx(
        0.5
    )


TIME_MODEL = """
model timedep
  compartment C = 1;
  species A in C; species B in C;
  k1 = 0.8; k2 = 0.6
  A = 1; B = 1
  J1: A -> B; k1*A*(1 + 0.5*time)
  J2: B -> A; k2*B
end
"""


@pytest.mark.parametrize(
    "source", [sbml(), sbml(TIME_MODEL)], ids=["local", "absolute"]
)
def test_a_value_at_a_time_equals_the_compiled_simulation(source: str) -> None:
    model = RoadrunnerSBMLModel(source=source)
    plan = _plan(model, Simulation(end=2, steps=8)).with_values({"k1": 2.0}, at=0.7)
    expected = _plan(
        model, Simulation(end=2, steps=8, changes=[Change(0.7, {"k1": 2.0})])
    )
    selections = ["time", "[A]", "[B]"]
    np.testing.assert_allclose(
        execute(plan, model, selections).values,
        execute(expected, model, selections).values,
        rtol=1e-12,
    )
    assert plan == expected
