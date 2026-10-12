"""The executor runs a plan with the initialization and change semantics of PEtab v2."""

import logging
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.model.roottol import EPSILON
from sbmlsim.result import TimecourseResult
from sbmlsim.simulation import Change, Simulation, SteadyState
from sbmlsim.simulator.executor import (
    SteadyStateError,
    _exact_times,
    _to_the_end,
    execute,
    steady_state,
)
from sbmlsim.simulator.plan import compile_simulation, time_tolerance
from tests.simulator.models import SPECIES_KINDS, sbml, sbml_pk

SEL = ["time", "[A]", "[B]", "X", "C", "k1"]

#: A + B at steady state, A = k2 / (k1 + k2) of the total
A_STEADY = 0.6 / 1.4


def run(sim: Simulation, model: RoadrunnerSBMLModel | None = None) -> TimecourseResult:
    """Run a simulation of the probe model."""
    model = model or RoadrunnerSBMLModel(source=sbml())
    plan = compile_simulation(sim, model.symbols, model.uinfo)
    return execute(plan, model, SEL)


def _after(result: TimecourseResult) -> np.ndarray:
    """Get the rows with the state after a change at their time, a mask.

    The output has the state before and the state after a change at its
    time, the last row of a time is the state after the changes.
    """
    time = result.time
    return np.append(time[1:] != time[:-1], True)


def test_preinit_reaches_initial_assignment() -> None:
    """A parameter before the initialization reaches the initial assignment of B."""
    # the rates are per volume of the compartment C = 2, the relaxation time
    # is 1 / 0.7, so the time 40 is at steady state
    res = run(Simulation(end=40, preinit_changes={"b0": 0.0}, times=[0, 40]))
    assert res["[B]"][0] == pytest.approx(0.0)
    assert res["[A]"][-1] == pytest.approx(A_STEADY, rel=1e-3)


def test_second_run_on_the_same_instance_uses_the_model_value() -> None:
    """A value of an earlier simulation does not leak into the next one."""
    model = RoadrunnerSBMLModel(source=sbml())
    run(
        Simulation(
            end=1, preinit_changes={"b0": 0.0}, changes=[Change(0.5, {"k1": 3.0})]
        ),
        model,
    )
    res = run(Simulation(end=1, times=[0, 1]), model)
    assert res["[B]"][0] == pytest.approx(1.0)
    assert res["k1"][0] == pytest.approx(0.8)
    assert res["C"][0] == pytest.approx(2.0)


def test_change_at_start_and_output_at_the_time_of_a_change() -> None:
    """The output at the time of a change has the states before and after it."""
    res = run(
        Simulation(
            end=2,
            changes=[Change(0, {"[A]": 5.0}), Change(1, {"[A]": "[A] + 10"})],
            times=[0, 1, 2],
        )
    )
    assert res.time.tolist() == [0.0, 1.0, 1.0, 2.0]
    assert res["[A]"][0] == pytest.approx(5.0)
    before = run(Simulation(end=1, changes=[Change(0, {"[A]": 5.0})], times=[0, 1]))
    assert res["[A]"][1] == pytest.approx(before["[A]"][1])
    assert res["[A]"][2] == pytest.approx(before["[A]"][1] + 10.0)


def test_simultaneous_formulas_use_the_old_values() -> None:
    """All values of a change are evaluated before any of them is assigned."""
    res = run(
        Simulation(
            end=1, changes=[Change(0.5, {"[A]": "[B]", "[B]": "[A]"})], times=[0.5]
        )
    )
    ref = run(Simulation(end=0.5, times=[0.5]))
    assert res["[A]"][-1] == pytest.approx(ref["[B]"][0])
    assert res["[B]"][-1] == pytest.approx(ref["[A]"][0])


def test_formula_reads_the_time() -> None:
    """`time` in a formula is the time of the change."""
    res = run(Simulation(end=2, changes=[Change(1.5, {"k1": "time * 2"})], times=[2]))
    assert res["k1"][0] == pytest.approx(3.0)


def test_compartment_change_keeps_concentrations_and_amounts() -> None:
    """A concentration species keeps its concentration, an amount species its amount."""
    res = run(Simulation(end=1, changes=[Change(0.5, {"C": 4.0})], times=[0.5]))
    ref = run(Simulation(end=0.5, times=[0.5]))
    assert res["[A]"][-1] == pytest.approx(ref["[A]"][0])
    assert res["X"][-1] == pytest.approx(ref["X"][0])
    assert res["C"][-1] == pytest.approx(4.0)


def test_compartment_change_with_a_species_of_an_assignment_rule() -> None:
    """A species of an assignment rule follows its rule when its compartment changes.

    roadrunner refuses to set a species of an assignment rule (#300). Every
    other concentration species keeps its concentration, an amount species
    its amount.
    """
    model = RoadrunnerSBMLModel(source=sbml(SPECIES_KINDS))
    selections = ["time", "V", "[F]", "[R]", "NR", "[RR]", "[B]", "[I]", "N"]
    plan = compile_simulation(
        Simulation(end=2, changes=[Change(1, {"V": 30.0})], times=[1]),
        model.symbols,
        model.uinfo,
    )
    res = execute(plan, model, selections)
    assert res.time.tolist() == [1.0, 1.0]
    before = {s: res[s][0] for s in selections}
    after = {s: res[s][1] for s in selections}
    assert before["V"] == pytest.approx(15.0)
    assert after["V"] == pytest.approx(30.0)
    kept = ["[F]", "[RR]", "[B]", "[I]", "N"]
    assert {s: after[s] for s in kept} == pytest.approx({s: before[s] for s in kept})
    assert after["[R]"] == pytest.approx(3 * after["[F]"])
    assert after["NR"] == pytest.approx(2 * after["[F]"])


def test_compartment_and_species_in_one_change() -> None:
    """A species set together with its compartment gets the value it is set to."""
    res = run(
        Simulation(end=1, changes=[Change(0.5, {"C": 4.0, "[A]": 7.0})], times=[0.5])
    )
    assert res["[A]"][-1] == pytest.approx(7.0)


def test_negative_start_and_time_shift() -> None:
    """A simulation starts at a negative time, the time shift moves the result."""
    res = run(Simulation(start=-5, end=5, times=[-5, 0, 5], time_shift=5))
    np.testing.assert_allclose(res.time, [0.0, 5.0, 10.0])
    ref = run(Simulation(end=10, times=[0, 5, 10]))
    np.testing.assert_allclose(res["[A]"], ref["[A]"], rtol=1e-6)


def test_integrator_output_has_the_time_of_a_change_twice() -> None:
    """The steps of the integrator have the states before and after a change."""
    res = run(Simulation(end=2, changes=[Change(1, {"[A]": 5.0})]))
    times = res.time
    assert np.all(np.diff(times) >= 0)
    assert np.count_nonzero(np.diff(times) == 0) == 1
    k = np.flatnonzero(times == 1.0)
    assert k.size == 2
    assert res["[A]"][k[0]] != pytest.approx(5.0)
    assert res["[A]"][k[1]] == pytest.approx(5.0)
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(2.0)


def test_multiple_dosing() -> None:
    """A change at a vector of times is applied at each of them."""
    res = run(
        Simulation(
            end=3,
            changes=[Change([0, 1, 2], {"X": "X + 1"})],
            times=[0, 1, 2, 3],
        )
    )
    assert res.time.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 3.0]
    np.testing.assert_allclose(res["X"], [13.0, 13.0, 14.0, 14.0, 15.0, 15.0])


def test_change_at_the_end() -> None:
    """A change at the end is applied and is the last output."""
    res = run(Simulation(end=1, changes=[Change(1, {"X": 0.0})], times=[0, 1]))
    assert res["X"][-1] == pytest.approx(0.0)


def test_steady_state_presimulation() -> None:
    """A presimulation starts the simulation in the steady state."""
    res = run(Simulation(end=1, presimulation=SteadyState(), times=[0]))
    assert res["[A]"][0] == pytest.approx(A_STEADY * 2.0, rel=1e-5)


def test_steady_state_with_its_preinit() -> None:
    """The pre-initialization of the steady state enters the initialization."""
    res = run(
        Simulation(
            end=1,
            presimulation=SteadyState(preinit_changes={"b0": 0.0}),
            times=[0],
        )
    )
    assert res["[A]"][0] == pytest.approx(A_STEADY, rel=1e-5)


#: a reaction whose product has a species reference of a rate rule, as case
#: 01633 of the SBML Test Suite: roadrunner has a rate of change of
#: `B_ref`, but no name of it in `getRatesOfChangeNamedArray`
STOICHIOMETRY_RULE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="stoichiometry_rule">
    <listOfCompartments>
      <compartment id="C" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialAmount="1" hasOnlySubstanceUnits="true"
        boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C" initialAmount="0" hasOnlySubstanceUnits="true"
        boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="1" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="B_ref">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn> 0 </cn></math>
      </rateRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="J" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference id="B_ref" species="B" stoichiometry="2" constant="false"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def test_steady_state_of_a_model_with_a_rate_rule_of_a_stoichiometry() -> None:
    """The rates of change of a steady state are the ones of every state.

    roadrunner names only the species and the rate rules which are no species
    reference in `getRatesOfChangeNamedArray`, its values are the ones of
    every state: a steady state failed with a `ValueError` of numpy (#307,
    cases 01452, 01563 and 01633 of the SBML Test Suite with a steady state).
    """
    model = RoadrunnerSBMLModel(source=STOICHIOMETRY_RULE)
    plan = compile_simulation(
        Simulation(end=1, presimulation=SteadyState(), times=[0]),
        model.symbols,
        model.uinfo,
    )
    res = execute(plan, model, ["time", "A", "B"])
    assert res["A"][0] == pytest.approx(0.0, abs=1e-6)
    assert res["B"][0] == pytest.approx(2.0, rel=1e-6)


def test_steady_state_with_another_integrator() -> None:
    """A steady state counts the states of the model, not of CVODE.

    The tolerance vector of an integrator other than CVODE is empty, so the
    states of the steady state, counted by it, did not match the rates.
    """
    model = RoadrunnerSBMLModel(source=sbml())
    model.r_loaded.setIntegrator("rk45")
    res = run(Simulation(end=1, presimulation=SteadyState(), times=[0]), model)
    assert res["[A]"][0] == pytest.approx(A_STEADY * 2.0, rel=1e-4)


def test_steady_state_not_reached_raises() -> None:
    """A model which does not reach a steady state fails with the time."""
    growth = sbml("model g\n  x' = 1\n  x = 0\nend")
    model = RoadrunnerSBMLModel(source=growth)
    plan = compile_simulation(
        Simulation(end=1, presimulation=SteadyState(max_time=100)),
        model.symbols,
        model.uinfo,
    )
    with pytest.raises(SteadyStateError, match="100"):
        execute(plan, model, ["time", "x"])


def test_parameter_with_initial_assignment_preinit() -> None:
    """A parameter with an initial assignment is set before the initialization."""
    model = RoadrunnerSBMLModel(source=sbml())
    res = run(Simulation(end=1, preinit_changes={"pinit": 7.0}, times=[0]), model)
    assert res["X"][0] == pytest.approx(21.0)
    res = run(Simulation(end=1, times=[0]), model)
    assert res["X"][0] == pytest.approx(12.0)


def test_selections_without_time_get_it() -> None:
    """The result always has the time as its first column."""
    model = RoadrunnerSBMLModel(source=sbml())
    plan = compile_simulation(
        Simulation(end=1, times=[0, 1]), model.symbols, model.uinfo
    )
    res = execute(plan, model, ["[A]"])
    assert res.columns == ("time", "[A]")


def test_integrator_setting_is_restored() -> None:
    """The output mode does not change the setting of the model."""
    model = RoadrunnerSBMLModel(source=sbml())
    model.r_loaded.getIntegrator().setValue("variable_step_size", False)
    run(Simulation(end=1), model)
    assert model.r_loaded.getIntegrator().getValue("variable_step_size") is False


def test_change_at_the_end_keeps_the_earlier_output() -> None:
    """Only the output at the end is the state after a change at the end."""
    res = run(Simulation(end=1e6, changes=[Change(1e6, {"X": 0.0})], times=[0, 999995]))
    assert res.time.tolist() == [0.0, 999995.0]
    assert res["X"][1] == pytest.approx(12.0)


def test_steady_state_error_names_the_model() -> None:
    """The error of a steady state names the model even without an id."""
    growth = sbml("model g\n  x' = 1\n  x = 0\nend")
    model = RoadrunnerSBMLModel(source=growth)
    plan = compile_simulation(
        Simulation(end=1, presimulation=SteadyState(max_time=10)),
        model.symbols,
        model.uinfo,
    )
    with pytest.raises(SteadyStateError) as err:
        execute(plan, model, ["time", "x"])
    assert "'None'" not in str(err.value)


def test_preinit_formula_of_parameters() -> None:
    """A formula before the initialization reads the parameters as they are set."""
    res = run(Simulation(end=1, preinit_changes={"[B]": "a0 + b0"}, times=[0]))
    assert res["[B]"][0] == pytest.approx(2.0)
    res = run(
        Simulation(end=1, preinit_changes={"a0": 3.0, "[B]": "a0 + b0"}, times=[0])
    )
    assert res["[B]"][0] == pytest.approx(4.0)
    # the initial assignment A = a0 follows the parameter
    assert res["[A]"][0] == pytest.approx(3.0)


def test_preinit_formula_reads_only_parameters() -> None:
    """A formula before the initialization cannot read a species, which has no value yet."""
    model = RoadrunnerSBMLModel(source=sbml())
    with pytest.raises(ValueError, match=r"'\[A\]'.*parameter"):
        compile_simulation(
            Simulation(end=1, preinit_changes={"[B]": "[A] + 1"}),
            model.symbols,
            model.uinfo,
        )


def test_steady_state_output_at_infinity() -> None:
    """An output time `inf` is the steady state after the end of the simulation."""
    res = run(Simulation(end=1, times=[0, 1, np.inf]))
    assert res.time[-1] == np.inf
    assert res["[A]"][-1] == pytest.approx(A_STEADY * 2.0, rel=1e-5)
    assert res["[A]"][1] != pytest.approx(A_STEADY * 2.0, rel=1e-5)


def test_only_the_steady_state_is_an_output() -> None:
    """A simulation whose only output is the steady state has one row."""
    res = run(Simulation(end=1, times=[np.inf]))
    assert res.time.tolist() == [np.inf]


#: a concentration S with dS/dt = 1 in a compartment C, and an event which
#: doubles C once S exceeds 14 (case 0016 of the PEtab test suite)
EVENT_MODEL = """
model ev_change
  compartment C = 4
  species S in C = 3
  S' = 1
  E: at (S > 14), t0=true: C = 2 * C
end
"""


def test_a_change_triggers_an_event() -> None:
    """An event whose trigger a change makes true fires at the time of the change."""
    model = RoadrunnerSBMLModel(source=sbml(EVENT_MODEL))
    plan = compile_simulation(
        Simulation(
            end=15,
            preinit_changes={"[S]": 2.0},
            changes=[Change(10, {"[S]": "C + [S]", "C": 8.0})],
            times=[0, 5, 10, 15],
        ),
        model.symbols,
        model.uinfo,
    )
    res = execute(plan, model, ["time", "C", "[S]", "S"])
    after = _after(res)
    np.testing.assert_allclose(np.asarray(res["C"])[after], [4, 4, 16, 16])
    np.testing.assert_allclose(np.asarray(res["[S]"])[after], [2, 7, 8, 13])
    np.testing.assert_allclose(np.asarray(res["S"])[after], [8, 28, 128, 208])
    # the state before the change at 10 has not triggered the event
    np.testing.assert_allclose(res.values[~after][0], [10, 4, 12, 48])


def test_an_event_fires_once_after_a_change() -> None:
    """The event of a change does not fire again when the integration continues."""
    model = RoadrunnerSBMLModel(
        source=sbml(
            "model ev4\n  X = 0\n  Y = 0\n  E: at (X > 1), t0=true: Y = Y + 1\nend"
        )
    )
    plan = compile_simulation(
        Simulation(end=10, changes=[Change(5, {"X": 2.0})], times=[0, 5, 10]),
        model.symbols,
        model.uinfo,
    )
    res = execute(plan, model, ["time", "Y"])
    np.testing.assert_allclose(res["Y"], [0, 0, 1, 1])


@pytest.mark.parametrize("times", [[0, 5, 10, 15], None])
def test_an_event_at_the_time_of_a_change_fires_after_it(
    times: list[float] | None,
) -> None:
    """An event and a change at one time: the change first, then the event (case 0030).

    The steps of the integrator (`times=None`) end before the event as well.
    """
    model = RoadrunnerSBMLModel(
        source=sbml(
            "model ev_time\n  compartment C = 4\n  species S in C = 3\n  S' = 1\n"
            "  E: at (time >= 10), t0=true: C = 2 * C\nend"
        )
    )
    plan = compile_simulation(
        Simulation(
            end=15,
            preinit_changes={"[S]": 2.0},
            changes=[Change(10, {"[S]": "C + [S]", "C": 8.0})],
            times=times,
        ),
        model.symbols,
        model.uinfo,
    )
    res = execute(plan, model, ["time", "C", "[S]", "S"])
    rows = np.isin(res.time, [0, 10, 15]) & _after(res)
    np.testing.assert_allclose(np.asarray(res["C"])[rows], [4, 16, 16])
    np.testing.assert_allclose(np.asarray(res["S"])[rows], [8, 128, 208], rtol=1e-6)


#: an event which counts the crossings of `X` above 1
COUNTER = """
model counter
  var X = 0
  var Y = 0
  E1: at (X > 1): Y = Y + 1
end
"""


def test_an_event_of_a_change_at_the_start_after_a_pre_equilibration() -> None:
    """The change at the start after the steady state triggers the event."""
    model = RoadrunnerSBMLModel(source=sbml(COUNTER))
    simulation = Simulation(
        end=1, steps=2, presimulation=SteadyState(), changes=[Change(0, {"X": 2.0})]
    )
    plan = compile_simulation(simulation, model.symbols, model.uinfo)
    result = execute(plan, model, ["time", "X", "Y"])
    assert np.asarray(result["Y"]).tolist() == [1.0, 1.0, 1.0]


def test_an_event_of_a_change_at_the_end() -> None:
    """The change at the end triggers the event, its output is after both."""
    model = RoadrunnerSBMLModel(source=sbml(COUNTER))
    simulation = Simulation(end=1, steps=2, changes=[Change(1, {"X": 2.0})])
    plan = compile_simulation(simulation, model.symbols, model.uinfo)
    result = execute(plan, model, ["time", "X", "Y"])
    assert np.asarray(result["X"])[-1] == 2.0
    assert result.time.tolist() == [0.0, 0.5, 1.0, 1.0]
    assert np.asarray(result["Y"]).tolist() == [0.0, 0.0, 0.0, 1.0]


#: a change at the start, after it and at the end of the probe model
CHANGES = [
    Change(0, {"[A]": 5.0}),
    Change(1, {"[A]": "[A] + 10"}),
    Change(2, {"X": 0.0}),
]


def _rows_at(result: TimecourseResult, time: float) -> np.ndarray:
    """Get the rows of a result at a time."""
    return result.values[result.time == time]


def test_the_states_before_and_after_the_changes_after_the_start() -> None:
    """The state before a change is the end of the segment before it (#262)."""
    result = run(Simulation(end=2, changes=CHANGES, times=[0, 0.5, 1, 1.5, 2]))
    reference = run(Simulation(end=1, changes=CHANGES[:1], times=[0, 1]))
    np.testing.assert_array_equal(result.time, [0, 0.5, 1, 1, 1.5, 2, 2])
    # the change at the start is the initial condition, it has one row
    assert result["[A]"][0] == pytest.approx(5.0)
    before, after = _rows_at(result, 1.0)
    assert before[1] == pytest.approx(reference["[A]"][1], rel=1e-6)
    assert after[1] == pytest.approx(reference["[A]"][1] + 10.0, rel=1e-6)
    np.testing.assert_allclose(result["X"][-2:], [12.0, 0.0])


def test_the_states_at_a_change_from_a_negative_start() -> None:
    """The states of a simulation from a negative time in local time, shifted."""
    simulation = Simulation(
        start=-1,
        end=1,
        changes=[Change(0, {"[A]": "[A] + 10"})],
        times=[-1, 0, 1],
        time_shift=1,
    )
    result = run(simulation)
    reference = run(Simulation(end=1, times=[0, 1]))
    np.testing.assert_allclose(result.time, [0, 1, 1, 2])
    assert result["[A]"][1] == pytest.approx(reference["[A]"][1], rel=1e-6)
    assert result["[A]"][2] == pytest.approx(reference["[A]"][1] + 10.0, rel=1e-6)


def test_the_state_before_a_change_is_before_the_event_it_triggers() -> None:
    """An event a change triggers fires after the change, not before it."""
    model = RoadrunnerSBMLModel(source=sbml(COUNTER))
    simulation = Simulation(
        end=2, times=[0, 1, 2], changes=[Change(1, {"X": 2.0}), Change(2, {"X": 0.0})]
    )
    plan = compile_simulation(simulation, model.symbols, model.uinfo)
    result = execute(plan, model, ["time", "X", "Y"])
    assert result.time.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0]
    assert np.asarray(result["X"]).tolist() == [0.0, 0.0, 2.0, 2.0, 0.0]
    assert np.asarray(result["Y"]).tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize("times", [None, [0.5, 1, 2]])
def test_the_states_before_the_changes_are_the_ends_of_the_segments(
    times: list[float] | None,
) -> None:
    """The state before a change is the last row of a simulation which ends at it.

    For the steps of the integrator it is the row roadrunner gives at the
    end; the output has every change of the steps, and the changes at its
    times of an output of times.
    """
    model = RoadrunnerSBMLModel(source=sbml())
    simulation = Simulation(end=2, changes=CHANGES, times=times, time_shift=3)
    plan = compile_simulation(simulation, model.symbols, model.uinfo)
    assert plan.change_times == (1.0, 2.0)
    result = execute(plan, model, SEL)
    first = run(Simulation(end=1, changes=CHANGES[:1], times=times and times[:2]))
    second = run(Simulation(end=2, changes=CHANGES[:2], times=times))
    np.testing.assert_allclose(
        _rows_at(result, 4.0)[0, 1:], first.values[-1, 1:], rtol=1e-12
    )
    np.testing.assert_allclose(
        _rows_at(result, 5.0)[0, 1:], second.values[-1, 1:], rtol=1e-12
    )
    # the output at the end has the state after the change at the end
    assert result["X"][-2:].tolist() == pytest.approx([12.0, 0.0])


@pytest.mark.parametrize("times", [None, [0, 0.5, 1.5, 2]])
def test_the_states_after_the_changes(times: list[float] | None) -> None:
    """The state after a change is the start of the segment after it.

    An output of times without the time of a change has no row at it.
    """
    model = RoadrunnerSBMLModel(source=sbml())
    result = run(Simulation(end=2, changes=CHANGES, times=times), model)
    if times is None:
        before, after = _rows_at(result, 1.0)
        assert after[1] == pytest.approx(before[1] + 10.0)
    else:
        assert not np.isin(1.0, result.time)
    assert result["X"][-1] == 0.0


class _FailingIntegrator:
    """An integrator which fails at the end."""

    def setValue(self, key: str, value: object) -> None:
        """Ignore a setting."""

    def integrate(self, start: float, step: float) -> float:
        """Fail."""
        raise RuntimeError("CV_TOO_MUCH_WORK")


class _FailingRoadRunner:
    """A roadrunner instance whose integrator fails at the end."""

    def getIntegrator(self) -> _FailingIntegrator:
        """Get the integrator."""
        return _FailingIntegrator()


def test_a_failure_of_the_integrator_at_the_end_raises() -> None:
    """The integrator fails at the end: an error, or an event after the end."""
    r = _FailingRoadRunner()
    with pytest.raises(RuntimeError, match="CV_TOO_MUCH_WORK"):
        _to_the_end(r, 1.0, events=False)  # ty: ignore[invalid-argument-type]
    assert not _to_the_end(r, 1.0, events=True)  # ty: ignore[invalid-argument-type]


def test_a_plan_without_changes_after_the_start_has_one_row_per_time() -> None:
    """A change at the start is the initial condition, it has one row."""
    model = RoadrunnerSBMLModel(source=sbml())
    plan = compile_simulation(
        Simulation(end=1, changes=CHANGES[:1], times=[0, 1]), model.symbols, model.uinfo
    )
    assert execute(plan, model, SEL).time.tolist() == [0.0, 1.0]
    assert plan.at_change([0.0, 1.0]).tolist() == [-1, -1]


#: the oral dose of the PK model, in mg
PK_DOSE = Change(0, {"PODOSE": 100.0})
#: the selections of the PK model
PK_SEL = ["time", "[C]", "Aurine", "Aelim"]


def run_on(
    model: RoadrunnerSBMLModel, sim: Simulation, selections: list[str]
) -> TimecourseResult:
    """Run a simulation of a model."""
    plan = compile_simulation(sim, model.symbols, model.uinfo)
    return execute(plan, model, selections)


def test_a_change_to_its_own_value_changes_nothing() -> None:
    """The steps of the integrator continue from the state at the time of a change.

    roadrunner ends the last step of a segment after its end; the change at
    24 h of `ka` to its own value was applied to the state of that later time
    and moved the concentration after it by -1.4 %. CVODE starts again at a
    change, which moves the solution by its error, 1.4e-5 here, as with
    exact output times.
    """
    model = RoadrunnerSBMLModel(source=sbml_pk(urine=True))
    steps = run_on(
        model, Simulation(end=48, changes=[PK_DOSE, Change(24, {"ka": 1.0})]), PK_SEL
    )
    # the first steps are closer than the resolution of the time, the state
    # after the change at 24 h
    rows = (steps.time >= 1.0) & _after(steps)
    exact = run_on(
        model,
        Simulation(end=48, changes=[PK_DOSE], times=steps.time[rows].tolist()),
        PK_SEL,
    )
    np.testing.assert_array_equal(exact.time, steps.time[rows])
    np.testing.assert_allclose(steps.values[rows], exact.values, rtol=5e-5)


def test_the_steps_end_in_the_state_at_the_end() -> None:
    """The last output of the steps is the state at the end, not an interpolation."""
    model = RoadrunnerSBMLModel(source=sbml_pk(urine=True))
    steps = run_on(model, Simulation(end=24, changes=[PK_DOSE]), PK_SEL)
    exact = run_on(model, Simulation(end=24, changes=[PK_DOSE], times=[0, 24]), PK_SEL)
    assert steps.time[-1] == 24.0
    np.testing.assert_allclose(steps.values[-1], exact.values[-1], rtol=1e-9)


def test_the_states_before_the_changes_of_the_steps_are_exact() -> None:
    """The state before a change of the steps is the one of exact output times."""
    model = RoadrunnerSBMLModel(source=sbml_pk(urine=True))
    changes = [PK_DOSE, Change(24, {"Aurine": 0.0})]
    states = [
        _rows_at(
            run_on(model, Simulation(end=48, changes=changes, times=times), PK_SEL),
            24.0,
        )
        for times in (None, [0, 24, 48])
    ]
    np.testing.assert_allclose(states[0], states[1], rtol=1e-9)


#: a decay whose event counts the crossing of `S` below 1, at the time ln(10)
DECAY = """
model decay
  var S = 10
  var Y = 0
  k = 1
  S' = -k * S
  E: at (S < 1): Y = Y + 1
end
"""
#: the time of the event of the decay
DECAY_EVENT = float(np.log(10.0))


def test_an_event_after_the_end_of_a_segment_fires_at_its_time() -> None:
    """An event in the last step of roadrunner past a change fires after the change.

    The last step of the segment before the change at 2.25 ends after the
    event at ln(10); roadrunner fired it at the change, and the decay
    continued from the state at the event.
    """
    model = RoadrunnerSBMLModel(source=sbml(DECAY))
    result = run_on(
        model, Simulation(end=4, changes=[Change(2.25, {"k": 1.0})]), ["time", "S", "Y"]
    )
    fired = np.asarray(result["Y"]) > 0
    assert result.time[np.argmax(fired)] == pytest.approx(DECAY_EVENT, rel=1e-6)
    assert np.asarray(result["Y"])[-1] == 1.0
    assert result["S"][-1] == pytest.approx(10.0 * np.exp(-4.0), rel=1e-5)


def test_a_simulation_leaves_no_event_of_roadrunner_pending() -> None:
    """The next simulation of the model does not fire an event of the last one.

    The last step of the simulation to 2.25 ends after the event at ln(10),
    which roadrunner kept and fired at the start of the next simulation, in
    an output row of its own.
    """
    model = RoadrunnerSBMLModel(source=sbml(DECAY))
    run_on(model, Simulation(end=2.25), ["time", "S", "Y"])
    second = run_on(model, Simulation(end=1), ["time", "S", "Y"])
    fresh = run_on(
        RoadrunnerSBMLModel(source=sbml(DECAY)), Simulation(end=1), ["time", "S", "Y"]
    )
    np.testing.assert_array_equal(second.values, fresh.values)


def test_the_last_output_of_the_steps_is_at_the_end() -> None:
    """The last output of the steps is at the end, also in local time.

    The last segment from 0.2 to 0.9 starts at the time 0 of roadrunner, its
    end shifted back is `(0.9 - 0.2) + 0.2 = 0.8999999999999999`; the output
    at the end is the state after the change at the end.
    """
    model = RoadrunnerSBMLModel(source=sbml())
    assert not model.symbols.time_dependent
    result = run_on(
        model,
        Simulation(end=0.9, changes=[Change(0.2, {"X": 1.0}), Change(0.9, {"X": 0.0})]),
        SEL,
    )
    assert (0.9 - 0.2) + 0.2 != 0.9
    assert result.time[-1] == 0.9
    assert result["X"][-1] == 0.0


#: a sawtooth: `X` grows with the rate `k` and an event resets it at 1 and
#: counts the resets in `Y`
SAWTOOTH = """
model sawtooth
  var X = 0
  var Y = 0
  k = 1
  X' = k
  E: at (X >= 1): X = 0, Y = Y + 1
end
"""


def _count_runs(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the integrations of a plan, see `executor._simulate`."""
    from sbmlsim.simulator import executor

    runs: list[int] = []
    simulate = executor._simulate

    def counted(*args: object, **kwargs: object) -> np.ndarray:
        runs.append(1)
        return simulate(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(executor, "_simulate", counted)
    return runs


def test_segments_past_their_end_are_integrated_again_at_most_twice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The plan runs three times however many segments roadrunner integrated past their end.

    A change of `k` to its own value 0.002 before every reset: the last step
    of roadrunner before every change ends after the reset. The second run
    finds the second segment past its end, the third one keeps the state at
    the start of every later segment in a temporary file, which it removes;
    `saveStateS` of roadrunner 2.10 keeps every state in memory and is not
    used.
    """
    import roadrunner

    def leaking(self: object) -> bytes:
        raise AssertionError("saveStateS keeps every state in memory")

    monkeypatch.setattr(roadrunner.RoadRunner, "saveStateS", leaking)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    runs = _count_runs(monkeypatch)
    model = RoadrunnerSBMLModel(source=sbml(SAWTOOTH))
    times = [j - 0.002 for j in range(1, 11)]
    result = run_on(
        model,
        Simulation(end=10.5, changes=[Change(times, {"k": 1.0})]),
        ["time", "X", "Y"],
    )
    assert len(runs) == 3
    assert list(tmp_path.iterdir()) == []
    at = np.isin(result.time, times) & _after(result)
    np.testing.assert_allclose(np.asarray(result["X"])[at], 0.998, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(result["Y"])[at], np.arange(10))
    assert result["Y"][-1] == 10.0
    # the states of the third run are the ones of exact output times
    exact = run_on(
        RoadrunnerSBMLModel(source=sbml(SAWTOOTH)),
        Simulation(end=10.5, changes=[Change(times, {"k": 1.0})], times=times),
        ["time", "X", "Y"],
    )
    np.testing.assert_allclose(
        np.asarray(result["X"])[at], np.asarray(exact["X"])[_after(exact)], rtol=1e-9
    )


def test_a_model_without_events_is_integrated_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A segment shorter than the resolution of the time is no event past its end."""
    runs = _count_runs(monkeypatch)
    model = RoadrunnerSBMLModel(source=sbml())
    run_on(
        model,
        Simulation(
            end=2, changes=[Change(1, {"X": 1.0}), Change(1 + 1e-12, {"X": 2.0})]
        ),
        SEL,
    )
    assert len(runs) == 1


def test_an_exact_segment_keeps_the_states_before_and_after_an_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A segment integrated again with exact output keeps an event inside as two rows.

    The segment up to the change at 1.998 has the reset at 1 inside, and its
    last step ends after the reset at 2: it is integrated again with exact
    output, which has the state before the reset at 1 and the state after it,
    as the steps of the integrator have them.
    """
    runs = _count_runs(monkeypatch)
    model = RoadrunnerSBMLModel(source=sbml(SAWTOOTH))
    result = run_on(
        model,
        Simulation(end=2.5, changes=[Change(1.998, {"k": 1.0})]),
        ["time", "X", "Y"],
    )
    assert len(runs) == 2
    time, x = result.time, np.asarray(result["X"])
    jump = np.flatnonzero(
        (x[:-1] > 0.999) & (x[1:] < 1e-3) & (np.abs(time[:-1] - 1) < 1e-6)
    )
    assert len(jump) == 1
    assert time[jump[0] + 1] == pytest.approx(1.0, abs=1e-6)
    assert np.asarray(result["Y"])[-1] == 2.0


def test_the_exact_times_of_a_segment_keep_the_events() -> None:
    """Steps closer than the resolution of the time are an event, before and after it.

    A step closer to the end than the resolution is an event roadrunner fired
    at the end and is left out.
    """
    times, labels = _exact_times(
        0.0,
        np.array([0.003, 0.004, 0.5, 1.0, 1.0, 1.0 + 1e-12, 1.7, 1.98, 1.995]),
        2.0,
        0.01,
    )
    # the integration a resolution before and after an event, the rows at the
    # times of the steps; a cluster at the start keeps the state after it
    np.testing.assert_allclose(times, [0.0, 0.014, 0.5, 0.99, 1.01, 1.7, 1.98, 2.0])
    np.testing.assert_allclose(
        labels, [0.0, 0.004, 0.5, 1.0, 1.0 + 1e-12, 1.7, 1.98, 2.0]
    )


def test_the_exact_times_of_two_close_events_are_sorted() -> None:
    """Two events less than two resolutions apart share the time between them.

    A resolution after the first event is after the time a resolution before
    the second one, which roadrunner rejects; both are half way between the
    events, the state between them.
    """
    times, labels = _exact_times(
        0.0, np.array([0.5, 1.0, 1.0, 1.015, 1.015, 1.5]), 2.0, 0.01
    )
    np.testing.assert_allclose(times, [0.0, 0.5, 0.99, 1.0075, 1.0075, 1.025, 1.5, 2.0])
    np.testing.assert_allclose(labels, [0.0, 0.5, 1.0, 1.0, 1.015, 1.015, 1.5, 2.0])
    assert np.all(np.diff(times) >= 0.0)


class _Folded:
    """A roadrunner instance which roadrunner left at a time in a state."""

    def __init__(self, time: float, state: list[float]) -> None:
        self.time, self.state = time, state
        self.model = self

    def getTime(self) -> float:
        return self.time

    def getSelectedValues(self) -> list[float]:
        return self.state


@pytest.mark.parametrize(
    ("end", "model_time", "offset", "folded"),
    [
        (2.0, 1.9999995, 1e-6, True),  # the root of a trigger, `roottol` before
        (2.0, 1.9999995, EPSILON, True),  # an event at the end of a step of CVODE
        (5.0, 4.4999995, 0.0, True),  # from 4 on the epsilon rounds away
        (10.0, 9.9999995, 0.0, True),
        (2.0, 2.0000004, 1e-6, True),  # an event after the end
        (2.0, 2.0000004, 1.0000004, False),  # a step of CVODE past the end
        (2.0, 1.9999995, 3e-6, False),  # another distance: a step of CVODE
    ],
)
def test_an_event_whose_row_after_it_roadrunner_folded_into_the_end(
    end: float, model_time: float, offset: float, folded: bool
) -> None:
    """roadrunner left the model at the event, `roottol` after its row before it."""
    from sbmlsim.model.roottol import RootTolerance
    from sbmlsim.simulator.executor import _folded_event

    block = np.array(
        [[0.0, 0.0], [1.0, 1.0], [model_time - offset, 2.0], [end + 1e-16, 0.7]]
    )
    event = _folded_event(
        _Folded(model_time, [model_time, 0.0]),  # ty: ignore[invalid-argument-type]
        block,
        end,
        RootTolerance(1e-6, 1e-19),
        1e-8,
    )
    assert event == (model_time if folded else None)


@pytest.mark.parametrize(("last", "paired"), [(2.0, False), (2.0 - 1e-6, True)])
def test_the_last_row_of_roadrunner_is_no_row_of_an_event(
    last: float, paired: bool
) -> None:
    """The row roadrunner interpolated to the end is left out of the events.

    CVODE located an event a few ulps after the end; roadrunner's last row is
    `roottol` after the row before the event. A last row before the end is a
    step, roadrunner stopped at the most rows of its output.
    """
    from sbmlsim.model.roottol import RootTolerance
    from sbmlsim.simulator.executor import _at_event_times

    roottol = RootTolerance(1e-6, 1e-19)
    block = np.array([[0.0, 0.0], [1.0, 1.0], [last - 1e-6, 2.0], [last, 0.7]])
    _at_event_times(block, 2.0, roottol, 1e-8)
    if paired:
        assert block[2, 0] == block[3, 0] == last - 1e-6
    else:
        np.testing.assert_array_equal(block[:, 0], [0.0, 1.0, 2.0 - 1e-6, 2.0])


def test_an_event_after_the_end_which_roadrunner_applied_is_past_the_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An event at a change fires after it: the segment gets exact output.

    roadrunner applied the event, `roottol` after its row before it, and the
    row before it is no step of CVODE, so it is left out of the steps.
    """
    from sbmlsim.model.roottol import RootTolerance
    from sbmlsim.simulator import executor

    monkeypatch.setattr(executor, "root_tolerance", lambda: RootTolerance(1e-6, 1e-19))
    end = float(np.nextafter(2.0, -np.inf))
    block = np.array([[0.0, 0.0], [1.0, 1.0], [2.0 + 1e-13 - 1e-6, 2.0], [end, 0.7]])
    out, inside = executor._events_at_the_end(
        _Folded(2.0 + 1e-13, [2.0 + 1e-13, 0.0]),  # ty: ignore[invalid-argument-type]
        block.copy(),
        end,
        1e-8,
    )
    assert not inside
    np.testing.assert_array_equal(out, block[[0, 1, 3]])


def test_an_event_within_the_resolution_before_the_end_gets_its_row_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The state of the model is the row after the event, the steps do not go on."""
    from sbmlsim.model.roottol import RootTolerance
    from sbmlsim.simulator import executor

    monkeypatch.setattr(executor, "root_tolerance", lambda: RootTolerance(1e-6, 1e-19))
    event = 2.0 - 5e-9
    block = np.array([[0.0, 0.0], [1.0, 1.0], [event - 1e-6, 2.0], [2.0, 0.7]])
    out, inside = executor._events_at_the_end(
        _Folded(event, [event, 0.0]),  # ty: ignore[invalid-argument-type]
        block,
        2.0,
        1e-8,
    )
    assert inside
    np.testing.assert_array_equal(
        out, [[0, 0], [1, 1], [event, 2], [event, 0], [2.0, 0.7]]
    )


#: two events of the time less than two resolutions of the time of a simulation
#: to 2.5 apart, and a reset of `X` at 2, after a change at 1.998
TWINS = """
model twins
  var X = 0
  var Y = 0
  k = 1
  X' = k
  E1: at (time >= 1): Y = Y + 1
  E2: at (time >= {second!r}): Y = Y + 10
  E3: at (X >= 2): X = 0
end
"""


def test_two_close_events_in_a_segment_past_its_end(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A segment past its end with two events closer than two resolutions runs again.

    The last step of roadrunner before the change at 1.998 ends after the
    reset at 2, the segment is integrated again with exact output at its
    steps, which have the events at 1 and 1.5 resolutions after it.
    """
    runs = _count_runs(monkeypatch)
    end = 2.5
    second = 1.0 + 1.5 * time_tolerance(0.0, end)
    model = RoadrunnerSBMLModel(source=sbml(TWINS.format(second=second)))
    assert model.symbols.time_dependent
    with caplog.at_level(logging.DEBUG, logger="sbmlsim.simulator.executor"):
        result = run_on(
            model,
            Simulation(end=end, changes=[Change(1.998, {"k": 1.0})]),
            ["time", "X", "Y"],
        )
    assert len(runs) == 2
    assert "the plan runs again with exact output in it (run 2)" in caplog.text
    near = np.abs(result.time - 1.0) < 1e-6
    # the states before the first event, between the events and after the second
    np.testing.assert_array_equal(np.asarray(result["Y"])[near], [0, 1, 1, 11])
    assert result.time[near][-1] == pytest.approx(second, abs=1e-10)
    assert np.all(np.diff(result.time) >= 0.0)
    assert result["Y"][-1] == 11.0
    assert result["X"][-1] == pytest.approx(0.5, rel=1e-6)


def test_a_steady_state_ends_at_the_end_of_its_last_horizon() -> None:
    """The presimulation ends in the state at the end of its last horizon."""
    model = RoadrunnerSBMLModel(source=sbml())
    plan = compile_simulation(
        Simulation(end=1, presimulation=SteadyState()), model.symbols, model.uinfo
    )
    assert plan.steady_state is not None
    model.initialize([])
    time = steady_state(model, plan.steady_state)
    model.simulated()
    assert model.r_loaded.model.getTime() == time


#: a model which reads the time, integrated in absolute time
CLOCK = """
model clock
  var X = 0
  X' = 1 + 0 * time
end
"""


@pytest.mark.parametrize(
    "simulation",
    [
        Simulation(start=-3, end=-1),
        Simulation(start=-3, end=1, changes=[Change(-1, {"X": 0.0})]),
    ],
)
def test_the_steps_of_a_negative_end_warn_nothing(simulation: Simulation) -> None:
    """roadrunner warns that it did not reach a negative end, which it reached."""
    model = RoadrunnerSBMLModel(source=sbml(CLOCK))
    assert model.symbols.time_dependent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = run_on(model, simulation, ["time", "X"])
    assert result.time[-1] == simulation.end


#: `X` grows with the rate 1, a change resets it at 2
LINEAR = "model linear\n  var X = 0\n  X' = 1\nend"
#: a dose of `A` at the time 2 by an event of the model, `B` keeps the steps of
#: the integrator short
DOSE_EVENT = """
model dose
  var A = 0
  var B = 1
  k = 0.1
  A' = -k * A
  B' = -B
  E: at (time >= 2.0): A = A + 10
end
"""


@pytest.mark.parametrize("times", [None, [0, 1, 2, 3, 4]])
def test_a_change_after_the_start_has_the_states_before_and_after_it(
    times: list[float] | None,
) -> None:
    """The output has two rows at the time of a change, before and after it.

    The steps of the integrator have every change, an output of times the
    changes at its times; a change at the start is the initial condition and
    has one row.
    """
    model = RoadrunnerSBMLModel(source=sbml(LINEAR))
    result = run_on(
        model,
        Simulation(
            end=4,
            changes=[Change(0, {"X": 0.5}), Change(2, {"X": 0.0})],
            times=times,
        ),
        ["time", "X"],
    )
    time, x = result.time, np.asarray(result["X"])
    assert np.all(np.diff(time) >= 0)
    assert np.count_nonzero(time == 0.0) == 1
    assert x[0] == 0.5
    at = np.flatnonzero(time == 2.0)
    assert at.tolist() == [at[0], at[0] + 1]
    assert x[at].tolist() == pytest.approx([2.5, 0.0])
    if times is not None:
        assert time.tolist() == [0.0, 1.0, 2.0, 2.0, 3.0, 4.0]
        np.testing.assert_allclose(x, [0.5, 1.5, 2.5, 0.0, 1.0, 2.0])


@pytest.mark.parametrize("times", [None, [0, 1, 2]])
def test_a_change_at_the_end_has_the_states_before_and_after_it(
    times: list[float] | None,
) -> None:
    """The last two rows are the states before and after the change at the end."""
    model = RoadrunnerSBMLModel(source=sbml(LINEAR))
    result = run_on(
        model,
        Simulation(end=2, changes=[Change(2, {"X": 0.0})], times=times),
        ["time", "X"],
    )
    assert result.time[-2:].tolist() == [2.0, 2.0]
    assert np.asarray(result["X"])[-2:].tolist() == pytest.approx([2.0, 0.0])


#: an event of the model at the end of a simulation, the column and its states
#: before and after the event: the second reset of the sawtooth at 2, which
#: CVODE locates a few ulps after 2, and a dose by a trigger of the time
END_EVENTS = {
    "reset": (SAWTOOTH, 2.0, "X", 1.0, 0.0),
    "dose": (DOSE_EVENT, 2.0, "A", 0.0, 10.0),
}


@pytest.mark.parametrize("case", END_EVENTS)
def test_an_event_at_the_end_has_the_states_before_and_after_it(case: str) -> None:
    """The steps of the integrator have an event at the end as two rows.

    The states before and after the event, as an event inside and a change at
    the end have them. An output of times has one row at the end, the state
    after the event, which a time of a grid at an event sees.
    """
    sbml_model, end, column, before, after = END_EVENTS[case]
    model = RoadrunnerSBMLModel(source=sbml(sbml_model))
    steps = run_on(model, Simulation(end=end), ["time", column])
    assert steps.time[-2:].tolist() == [end, end]
    assert steps.time[-3] < end
    assert np.asarray(steps[column])[-2:].tolist() == pytest.approx(
        [before, after], abs=1e-9
    )
    times = run_on(
        model, Simulation(end=end, times=[0, end / 4, end]), ["time", column]
    )
    assert times.time.tolist() == [0.0, end / 4, end]
    assert times[column][-1] == pytest.approx(after, abs=1e-9)


@pytest.mark.parametrize("times", [None, [0, 1, 2]])
def test_an_event_at_a_change_at_the_end_fires_after_the_change(
    times: list[float] | None,
) -> None:
    """The dose at the end adds to the value the change at the end sets.

    The output has the state before both and the state after both; a dose
    before the change would have been overwritten by it.
    """
    model = RoadrunnerSBMLModel(source=sbml(DOSE_EVENT))
    result = run_on(
        model,
        Simulation(end=2, changes=[Change(2, {"A": 5.0})], times=times),
        ["time", "A"],
    )
    assert result.time[-2:].tolist() == [2.0, 2.0]
    assert result.time[-3] < 2.0
    assert np.asarray(result["A"])[-2:].tolist() == [0.0, 15.0]


@pytest.mark.parametrize("times", [None, [0, 1, 2]])
def test_an_event_of_the_dynamics_at_a_change_at_the_end_fires(
    times: list[float] | None,
) -> None:
    """The second reset of the sawtooth at the end fires after a change there.

    No change makes its trigger true, CVODE locates it a few ulps after the
    end, which is at the end.
    """
    model = RoadrunnerSBMLModel(source=sbml(SAWTOOTH))
    result = run_on(
        model,
        Simulation(end=2, changes=[Change(2, {"k": 2.0})], times=times),
        ["time", "X", "Y", "k"],
    )
    assert result.time[-2:].tolist() == [2.0, 2.0]
    np.testing.assert_allclose(
        result.values[-2:, 1:], [[1.0, 1.0, 1.0], [0.0, 2.0, 2.0]], atol=1e-9
    )


@pytest.mark.parametrize("delta", [1e-12, 1e-7])
def test_an_event_after_the_end_fires_once_before_the_steady_state(
    delta: float,
) -> None:
    """The steady state after the end starts after the events at the end.

    The event at the end, within the resolution of the time after it, and an
    event after it which the integration after the end found, fired once: the
    steady state starts at the time of the model, not at the end again.
    """
    model = RoadrunnerSBMLModel(
        source=sbml(
            f"model m\n  var Y = 0\n  X = 1\n  X' = -X\n"
            f"  E: at (time >= 2 + {delta!r}): Y = Y + 1\nend"
        )
    )
    result = run_on(model, Simulation(end=2, times=[0, 2, np.inf]), ["time", "Y"])
    assert result.time.tolist() == [0.0, 2.0, np.inf]
    assert np.asarray(result["Y"]).tolist() == [0.0, 1.0 if delta < 1e-9 else 0.0, 1.0]


#: a dose by an event of the time at `te`, a model in absolute time, which a
#: PK simulation from a negative start often has at 0
DOSE_AT = """
model dose
  var A = 1
  var Y = 0
  te = 0
  k = 0.5
  A' = -k * A
  E: at (time >= te): A = A + 10, Y = Y + 1
end
"""

#: `A` at 0 from 1 at -4, before the dose
A_AT_0 = float(np.exp(-0.5 * 4.0))


@pytest.mark.parametrize("times", [None, [-4.0, -2.0, 0.0]])
def test_a_dose_at_the_end_0_from_a_negative_start(times: list[float] | None) -> None:
    """The integration stops a normal time before an end at 0.

    The stop one unit in the last place before 0 was a subnormal number,
    from which CVODE failed to integrate across the dose at 0
    (CV_CONV_FAILURE).
    """
    model = RoadrunnerSBMLModel(source=sbml(DOSE_AT))
    assert model.symbols.time_dependent
    result = run_on(model, Simulation(start=-4, end=0, times=times), ["time", "A", "Y"])
    at = result.time == 0.0
    expected = [A_AT_0, A_AT_0 + 10.0] if times is None else [A_AT_0 + 10.0]
    np.testing.assert_allclose(np.asarray(result["A"])[at], expected, rtol=1e-4)
    assert result["Y"][-1] == 1.0


@pytest.mark.parametrize("times", [None, [-4.0, -2.0, 0.0, 4.0]])
@pytest.mark.parametrize("end", [4.0, 0.0])
def test_a_dose_at_a_change_at_0_from_a_negative_start(
    times: list[float] | None, end: float
) -> None:
    """A change at 0 with the dose at 0: the dose fires once, after the change.

    The segment before the change stopped at a subnormal time: CVODE failed
    for a change at 0 inside the simulation, and a change at the end 0 gave
    `NaN` before it and the dose twice.
    """
    model = RoadrunnerSBMLModel(source=sbml(DOSE_AT))
    output = None if times is None else [t for t in times if t <= end]
    result = run_on(
        model,
        Simulation(start=-4, end=end, changes=[Change(0.0, {"A": 2.0})], times=output),
        ["time", "A", "Y"],
    )
    at = result.time == 0.0
    np.testing.assert_allclose(np.asarray(result["A"])[at], [A_AT_0, 12.0], rtol=1e-4)
    assert np.asarray(result["Y"])[at].tolist() == [0.0, 1.0]
    assert result["Y"][-1] == 1.0


@pytest.mark.parametrize("times", [False, True])
@pytest.mark.parametrize("at", ["before", "change", "end"])
@pytest.mark.parametrize("ulps", [1, 2, 4, 9])
@pytest.mark.parametrize("end", [2.0, 0.0])
def test_a_change_a_few_units_in_the_last_place_before_the_end(
    end: float, ulps: int, at: str, times: bool
) -> None:
    """A segment too short for CVODE keeps its state, its events of the time fire.

    A change a few units in the last place before the end of a model in
    absolute time failed in CVODE (CV_TOO_CLOSE); a dose at the change fires
    once, a dose at the end after the change, a dose before them before.
    """
    start = end - 2.0
    c = end
    for _ in range(ulps):
        c = float(np.nextafter(c, -np.inf))
    te = {"before": start + 1.0, "change": c, "end": end}[at]
    model = RoadrunnerSBMLModel(source=sbml(DOSE_AT))
    result = run_on(
        model,
        Simulation(
            start=start,
            end=end,
            preinit_changes={"te": te},
            changes=[Change(c, {"A": 0.0})],
            times=[start, start + 1.0, end] if times else None,
        ),
        ["time", "A", "Y"],
    )
    assert result.time[-1] == end
    assert np.all(np.diff(result.time) >= 0.0)
    assert result["Y"][-1] == 1.0
    assert result["A"][-1] == pytest.approx(0.0 if at == "before" else 10.0)


def _ulps_after(time: float, ulps: int) -> float:
    """Get the time a number of units in the last place after a time."""
    for _ in range(ulps):
        time = float(np.nextafter(time, np.inf))
    return time


@pytest.mark.parametrize("times", [False, True])
@pytest.mark.parametrize("at", ["first", "between", "second"])
@pytest.mark.parametrize("ulps", [1, 3, 9])
@pytest.mark.parametrize("c1", [1.0, 0.0, -1.0])
def test_a_dose_at_two_changes_a_few_units_in_the_last_place_apart(
    c1: float, ulps: int, at: str, times: bool
) -> None:
    """A dose at the first, between or at the second of two close changes fires once.

    The segment between the changes is too short for CVODE and not
    integrated; the triggers before the second change were evaluated at the
    time the integration stopped before the first one, so the dose fired at
    the first change fired again at the second.
    """
    c2 = _ulps_after(c1, ulps)
    te = {"first": c1, "between": _ulps_after(c1, 1), "second": c2}[at]
    start, end = c1 - 1.0, c1 + 1.0
    model = RoadrunnerSBMLModel(source=sbml(DOSE_AT))
    result = run_on(
        model,
        Simulation(
            start=start,
            end=end,
            preinit_changes={"te": te},
            changes=[Change(c1, {"A": 2.0}), Change(c2, {"k": 0.2})],
            times=[start, c1, end] if times else None,
        ),
        ["time", "A", "Y"],
    )
    assert result["Y"][-1] == 1.0
    assert result["A"][-1] == pytest.approx(12.0 * np.exp(-0.2 * (end - c2)), rel=1e-5)
    assert np.all(np.diff(result.time) >= 0.0)


def test_a_dose_a_few_units_in_the_last_place_before_a_change_fires_once() -> None:
    """A dose between the stop before a change and the change fires once.

    roadrunner integrated the stop four units in the last place before the
    change at 1.6 + 3 units to one unit after it, where the dose at 1.6
    fired, and the triggers before the change, evaluated at the stop, fired
    it a second time after the change; doses from 1 to 3.
    """
    wrong = []
    for te in np.round(np.linspace(1.0, 3.0, 41), 6)[::4]:
        te = float(te)
        model = RoadrunnerSBMLModel(
            source=sbml(
                f"model d\n  var A = 0\n  var Y = 0\n  k = 0.1\n  A' = -k * A\n"
                f"  E: at (time >= {te!r}): A = A + 10, Y = Y + 1\nend"
            )
        )
        for ulps in (1, 3):
            c = _ulps_after(te, ulps)
            for times in (None, [0.0, c, c + 1.0]):
                result = run_on(
                    model,
                    Simulation(
                        end=c + 1.0, changes=[Change(c, {"k": 0.2})], times=times
                    ),
                    ["time", "A", "Y"],
                )
                if result["Y"][-1] != 1.0 or np.any(np.diff(result.time) < 0.0):
                    wrong.append((te, ulps, times is None))
    assert not wrong


@pytest.mark.parametrize("times", [None, [0.0, 0.3, 1.0]])
def test_a_dose_at_a_change_and_a_change_of_a_grid_one_unit_after_it(
    times: list[float] | None,
) -> None:
    """A grid of changes from `np.arange` has 0.30000000000000004 next to 0.3.

    The dose at the change at 0.3 fired twice, at 0.3 and at the change of
    the grid one unit in the last place after it.
    """
    grid = np.arange(0.0, 1.0, 0.1)
    assert grid[3] == _ulps_after(0.3, 1)
    model = RoadrunnerSBMLModel(
        source=sbml(
            "model dose\n  var A = 1\n  var Y = 0\n  te = 0.3\n  k = 0.5\n"
            "  r = 0\n  A' = -k * A + r\n"
            "  E: at (time >= te): A = A + 10, Y = Y + 1\nend"
        )
    )
    dose = Change(0.3, {"k": 0.4})
    reference = run_on(
        model, Simulation(end=1.0, changes=[dose], times=[0.0, 1.0]), ["time", "A"]
    )
    result = run_on(
        model,
        Simulation(
            end=1.0,
            changes=[dose, Change(list(grid[1:]), {"r": "r + 0"})],
            times=times,
        ),
        ["time", "A", "Y"],
    )
    assert result["Y"][-1] == 1.0
    assert result["A"][-1] == pytest.approx(reference["A"][-1], rel=1e-5)


@pytest.mark.parametrize("at", ["start", "change"])
@pytest.mark.parametrize("start", [-2.0, 1.0])
def test_a_change_a_few_units_in_the_last_place_after_the_start(
    start: float, at: str
) -> None:
    """A first segment too short for CVODE: the doses are the ones of an integrated one.

    The triggers before the change were evaluated at the time 0 of the
    reset: a dose at the change from the start -2 was dropped, a dose at the
    start 1, which roadrunner does not fire at the start, fired.
    """
    model = RoadrunnerSBMLModel(source=sbml(DOSE_AT))
    results = []
    for ulps in (1, 3, 40):
        c1 = _ulps_after(start, ulps)
        te = start if at == "start" else c1
        results.append(
            run_on(
                model,
                Simulation(
                    start=start,
                    end=start + 2.0,
                    preinit_changes={"te": te},
                    changes=[Change(c1, {"k": 0.2})],
                ),
                ["time", "A", "Y"],
            )
        )
    # the change 40 units in the last place after the start is integrated
    assert results[-1]["Y"][-1] == (0.0 if at == "start" else 1.0)
    for result in results[:-1]:
        assert result["Y"][-1] == results[-1]["Y"][-1]
        assert result["A"][-1] == pytest.approx(results[-1]["A"][-1], rel=1e-9)


#: an event of the time within the resolution of the time before the end 2.5
LATE_EVENT = """
model late
  var X = 0
  var Y = 0
  k = 1
  X' = k
  E: at (time >= {time!r}): X = 0, Y = Y + 1
end
"""


@pytest.mark.parametrize("delta", [1e-9, 2e-9])
@pytest.mark.parametrize("change", [False, True])
def test_an_event_within_the_resolution_before_the_end_is_in_the_chain_at_it(
    delta: float, change: bool
) -> None:
    """An event closer to the end of a segment than the resolution is at its end.

    The steps of the integrator end closer to the end than the resolution,
    the segment gets exact output without them: the rows at the end, also the
    state before the change at the end, are the state after the event, the
    chain of the event and the end, as `grid_weights` sees it.
    """
    assert delta < time_tolerance(0.0, 2.5)
    model = RoadrunnerSBMLModel(source=sbml(LATE_EVENT.format(time=2.5 - delta)))
    changes = [Change(2.5, {"k": 2.0})] if change else []
    result = run_on(model, Simulation(end=2.5, changes=changes), ["time", "X", "Y"])
    near = result.time > 2.5 - 1e-6
    assert result.time[near].tolist() == ([2.5, 2.5] if change else [2.5])
    np.testing.assert_allclose(np.asarray(result["X"])[near], delta, rtol=1e-3)
    assert np.all(np.asarray(result["Y"])[near] == 1.0)


def _states_at_a_change(c: float) -> list[tuple[float, bool, float, float]]:
    """The states before and after a change at `c` with a dose event at `c`.

    The steps of the integrator and an output of times with the change, the
    dose of 10 fires after the change.
    """
    model = RoadrunnerSBMLModel(
        source=sbml(
            f"model d\n  var A = 0\n  k = 0.1\n  A' = -k * A\n"
            f"  E: at (time >= {c!r}): A = A + 10\nend"
        )
    )
    states = []
    for times in (None, [0.0, c / 3, c, c + 1.0]):
        result = run_on(
            model,
            Simulation(end=c + 1.0, changes=[Change(c, {"k": 0.2})], times=times),
            ["time", "A"],
        )
        at = np.flatnonzero(result.time == c)
        states.append((c, times is None, *np.asarray(result["A"])[at].tolist()))
    return states


def _wrong(states: list[tuple[float, bool, float, float]]) -> list:
    """The states at a change which are not 0 before it and 10 after it."""
    return [row for row in states if row[2] != 0.0 or row[3] != pytest.approx(10.0)]


#: times of the change for which roadrunner integrated the last output time
#: before it to the change, of the steps of the integrator (1, 1.525, 1.775, 2)
#: and of an output of times (1.69, 1.75, 1.775, 1.98), and two times for which
#: it did not
TOUT_ROUNDED = [1.0, 1.525, 1.69, 1.75, 1.775, 1.98, 2.0, 1.005, 2.5]


@pytest.mark.parametrize("c", TOUT_ROUNDED)
def test_the_state_before_a_change_is_before_the_event_at_its_time(c: float) -> None:
    """An event of the model at the time of a change fires after the change.

    roadrunner integrates the last output time of a segment from the one
    before it, `tout = t + (end - t)`, which rounds up to the time of the
    change for some times below half of it: the event fired before the change.
    """
    assert not _wrong(_states_at_a_change(c))


@pytest.mark.slow
def test_the_state_before_a_change_is_before_the_event_at_every_time() -> None:
    """The rule of `test_the_state_before_a_change_is_before_the_event_at_its_time`.

    401 times of a change from 1 to 3; without the time between of
    `_simulate_times` 27 states at a change were wrong.
    """
    states = []
    for c in np.round(np.linspace(1.0, 3.0, 401), 6):
        states.extend(_states_at_a_change(float(c)))
    assert not _wrong(states)
