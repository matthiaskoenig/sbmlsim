"""A simulation is compiled against a model into a plan without units."""

import pickle

import pytest

from sbmlsim import Q
from sbmlsim.model.symbols import ModelSymbols, TargetKind
from sbmlsim.simulation import Change, Simulation, SteadyState
from sbmlsim.simulator.plan import OutputMode, compile_simulation
from sbmlsim.units import UnitsInformation
from tests.simulator.models import sbml, sbml_minutes


@pytest.fixture(scope="module")
def probe() -> tuple[ModelSymbols, UnitsInformation]:
    """Get the symbols and the units of the probe model."""
    s = sbml()
    return ModelSymbols.from_sbml(s), UnitsInformation.from_sbml(s)


@pytest.fixture(scope="module")
def minutes() -> tuple[ModelSymbols, UnitsInformation]:
    """Get the symbols and the units of the probe model in minutes."""
    s = sbml_minutes()
    return ModelSymbols.from_sbml(s), UnitsInformation.from_sbml(s)


def test_events_are_sorted_and_merged(probe) -> None:
    """The changes become events sorted by time, one per time."""
    symbols, uinfo = probe
    plan = compile_simulation(
        Simulation(
            start=-10,
            end=10,
            changes=[
                Change([0, -10], {"k1": 1.0}),
                Change(0, {"[A]": "[A] + 1"}),
            ],
        ),
        symbols,
        uinfo,
    )
    assert [e.time for e in plan.events] == [-10.0, 0.0]
    assert {a.target for a in plan.events[1].assignments} == {"k1", "[A]"}
    formula = next(a for a in plan.events[1].assignments if a.target == "[A]")
    assert formula.kind is TargetKind.SPECIES_CONCENTRATION
    assert formula.formula == "[A] + 1"
    assert formula.value is None
    assert plan.output is OutputMode.INTEGRATOR
    assert plan.start == -10.0


def test_two_values_for_one_target_at_one_time_raise(probe) -> None:
    """Two values of one target at one time are ambiguous."""
    symbols, uinfo = probe
    with pytest.raises(ValueError, match=r"'k1'.*0"):
        compile_simulation(
            Simulation(end=1, changes=[Change(0, {"k1": 1.0}), Change(0, {"k1": 2.0})]),
            symbols,
            uinfo,
        )


def test_change_outside_of_the_interval_raises(probe) -> None:
    """A change after the end is never applied, which is an error."""
    symbols, uinfo = probe
    with pytest.raises(ValueError, match="outside"):
        compile_simulation(
            Simulation(end=1, changes=[Change(2, {"k1": 1.0})]), symbols, uinfo
        )


def test_output_time_outside_of_the_interval_raises(probe) -> None:
    """An output time after the end is an error."""
    symbols, uinfo = probe
    with pytest.raises(ValueError, match="outside"):
        compile_simulation(Simulation(end=1, times=[0, 2]), symbols, uinfo)


def test_steps_are_output_times(probe) -> None:
    """An equidistant grid is a set of exact output times."""
    symbols, uinfo = probe
    plan = compile_simulation(Simulation(end=4, steps=4), symbols, uinfo)
    assert plan.output is OutputMode.TIMES
    assert plan.times == (0.0, 1.0, 2.0, 3.0, 4.0)


def test_unknown_target_and_rule_target_raise(probe) -> None:
    """A target the model does not have, or a rule sets, is an error."""
    symbols, uinfo = probe
    with pytest.raises(ValueError, match="'nope'"):
        compile_simulation(
            Simulation(end=1, preinit_changes={"nope": 1.0}), symbols, uinfo
        )
    with pytest.raises(ValueError, match="'kk'"):
        compile_simulation(
            Simulation(end=1, preinit_changes={"kk": 1.0}), symbols, uinfo
        )


def test_formula_of_unknown_symbol_raises(probe) -> None:
    """A formula reads entities of the model and the time."""
    symbols, uinfo = probe
    with pytest.raises(ValueError, match="'nope'"):
        compile_simulation(
            Simulation(end=1, changes=[Change(0.5, {"k1": "nope + 1"})]),
            symbols,
            uinfo,
        )
    plan = compile_simulation(
        Simulation(end=1, changes=[Change(0.5, {"k1": "kk + time + [A]"})]),
        symbols,
        uinfo,
    )
    assert plan.events[0].assignments[0].formula == "kk + time + [A]"


def test_time_unit_and_quantities(minutes) -> None:
    """Times are converted into the time unit of the model, values into theirs."""
    symbols, uinfo = minutes
    plan = compile_simulation(
        Simulation(
            time_unit="hr",
            start=-1,
            end=2,
            preinit_changes={"f": Q(1, "g")},
            changes=[Change(Q(30, "min"), {"f": Q(2, "g")}), Change([1], {"k1": 2.0})],
            times=[0, 2],
        ),
        symbols,
        uinfo,
    )
    assert plan.start == -60.0
    assert plan.end == 120.0
    assert plan.times == (0.0, 120.0)
    assert [e.time for e in plan.events] == [30.0, 60.0]
    assert plan.preinit[0].value == pytest.approx(1000.0)
    assert plan.events[0].assignments[0].value == pytest.approx(2000.0)


def test_time_unit_of_the_wrong_dimension_raises(minutes) -> None:
    """A time unit which is not a time is an error."""
    symbols, uinfo = minutes
    with pytest.raises(ValueError, match="'mg'"):
        compile_simulation(Simulation(time_unit="mg", end=2), symbols, uinfo)


def test_value_of_the_wrong_dimension_raises(minutes) -> None:
    """A quantity which cannot be converted into the unit of its target is an error."""
    symbols, uinfo = minutes
    with pytest.raises(ValueError, match="'f'"):
        compile_simulation(
            Simulation(end=2, preinit_changes={"f": Q(1, "mol")}), symbols, uinfo
        )


def test_steady_state(probe) -> None:
    """The presimulation keeps its own pre-initialization and tolerances."""
    symbols, uinfo = probe
    plan = compile_simulation(
        Simulation(
            end=1,
            presimulation=SteadyState(preinit_changes={"k1": 0.1}, max_time=50),
        ),
        symbols,
        uinfo,
    )
    assert plan.steady_state is not None
    assert plan.steady_state.preinit[0].target == "k1"
    assert plan.steady_state.max_time == 50


def test_plan_is_picklable_and_takes_values(probe) -> None:
    """A plan is pickled for the workers and takes the values of a fit."""
    symbols, uinfo = probe
    plan = compile_simulation(
        Simulation(
            end=1, preinit_changes={"k1": 1.0}, changes=[Change([0, 0.5], {"f": 3.0})]
        ),
        symbols,
        uinfo,
    )
    again = pickle.loads(pickle.dumps(plan))
    assert again == plan
    overridden = plan.with_values({"f": 4.0, "k2": 0.1})
    assert all(a.value == 4.0 for e in overridden.events for a in e.assignments)
    assert {a.target: a.value for a in overridden.preinit} == {"k1": 1.0, "k2": 0.1}
    assert {a.target: a.value for a in plan.preinit} == {"k1": 1.0}
    with pytest.raises(ValueError, match="'nope'"):
        plan.with_values({"nope": 1.0})


def test_with_values_replaces_a_target_of_the_steady_state(probe) -> None:
    """A value of a target of the presimulation replaces it, it is not added."""
    symbols, uinfo = probe
    plan = compile_simulation(
        Simulation(end=1, presimulation=SteadyState(preinit_changes={"k1": 0.2})),
        symbols,
        uinfo,
    )
    changed = plan.with_values({"k1": 5.0})
    assert changed.preinit == ()
    assert changed.steady_state is not None
    assert [(a.target, a.value) for a in changed.steady_state.preinit] == [("k1", 5.0)]
