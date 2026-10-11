"""The executor runs a plan with the initialization and change semantics of PEtab v2."""

import warnings

import numpy as np
import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.result import TimecourseResult
from sbmlsim.simulation import Change, Simulation, SteadyState
from sbmlsim.simulator.executor import (
    SteadyStateError,
    _exact_times,
    _to_the_end,
    execute,
    execute_with_changes,
    steady_state,
)
from sbmlsim.simulator.plan import compile_simulation
from tests.simulator.models import sbml, sbml_pk

SEL = ["time", "[A]", "[B]", "X", "C", "k1"]

#: A + B at steady state, A = k2 / (k1 + k2) of the total
A_STEADY = 0.6 / 1.4


def run(sim: Simulation, model: RoadrunnerSBMLModel | None = None) -> TimecourseResult:
    """Run a simulation of the probe model."""
    model = model or RoadrunnerSBMLModel(source=sbml())
    plan = compile_simulation(sim, model.symbols, model.uinfo)
    return execute(plan, model, SEL)


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
    """The output at the time of a change is the state after the change."""
    res = run(
        Simulation(
            end=2,
            changes=[Change(0, {"[A]": 5.0}), Change(1, {"[A]": "[A] + 10"})],
            times=[0, 1, 2],
        )
    )
    assert res["[A]"][0] == pytest.approx(5.0)
    before = run(Simulation(end=1, changes=[Change(0, {"[A]": 5.0})], times=[0, 1]))
    assert res["[A]"][1] == pytest.approx(before["[A]"][1] + 10.0)


def test_simultaneous_formulas_use_the_old_values() -> None:
    """All values of a change are evaluated before any of them is assigned."""
    res = run(
        Simulation(
            end=1, changes=[Change(0.5, {"[A]": "[B]", "[B]": "[A]"})], times=[0.5]
        )
    )
    ref = run(Simulation(end=0.5, times=[0.5]))
    assert res["[A]"][0] == pytest.approx(ref["[B]"][0])
    assert res["[B]"][0] == pytest.approx(ref["[A]"][0])


def test_formula_reads_the_time() -> None:
    """`time` in a formula is the time of the change."""
    res = run(Simulation(end=2, changes=[Change(1.5, {"k1": "time * 2"})], times=[2]))
    assert res["k1"][0] == pytest.approx(3.0)


def test_compartment_change_keeps_concentrations_and_amounts() -> None:
    """A concentration species keeps its concentration, an amount species its amount."""
    res = run(Simulation(end=1, changes=[Change(0.5, {"C": 4.0})], times=[0.5]))
    ref = run(Simulation(end=0.5, times=[0.5]))
    assert res["[A]"][0] == pytest.approx(ref["[A]"][0])
    assert res["X"][0] == pytest.approx(ref["X"][0])
    assert res["C"][0] == pytest.approx(4.0)


def test_compartment_and_species_in_one_change() -> None:
    """A species set together with its compartment gets the value it is set to."""
    res = run(
        Simulation(end=1, changes=[Change(0.5, {"C": 4.0, "[A]": 7.0})], times=[0.5])
    )
    assert res["[A]"][0] == pytest.approx(7.0)


def test_negative_start_and_time_shift() -> None:
    """A simulation starts at a negative time, the time shift moves the result."""
    res = run(Simulation(start=-5, end=5, times=[-5, 0, 5], time_shift=5))
    np.testing.assert_allclose(res.time, [0.0, 5.0, 10.0])
    ref = run(Simulation(end=10, times=[0, 5, 10]))
    np.testing.assert_allclose(res["[A]"], ref["[A]"], rtol=1e-6)


def test_integrator_output_has_the_time_of_a_change_once() -> None:
    """The steps of the integrator are increasing and contain the change."""
    res = run(Simulation(end=2, changes=[Change(1, {"[A]": 5.0})]))
    times = res.time
    assert np.all(np.diff(times) > 0)
    k = int(np.flatnonzero(np.isclose(times, 1.0))[0])
    assert res["[A]"][k] == pytest.approx(5.0)
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
    np.testing.assert_allclose(res["X"], [13.0, 14.0, 15.0, 15.0])


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
    np.testing.assert_allclose(res["C"], [4, 4, 16, 16])
    np.testing.assert_allclose(res["[S]"], [2, 7, 8, 13])
    np.testing.assert_allclose(res["S"], [8, 28, 128, 208])


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
    np.testing.assert_allclose(res["Y"], [0, 1, 1])


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
    rows = np.isin(res.time, [0, 10, 15])
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
    assert np.asarray(result["Y"]).tolist() == [0.0, 0.0, 1.0]


#: a change at the start, after it and at the end of the probe model
CHANGES = [
    Change(0, {"[A]": 5.0}),
    Change(1, {"[A]": "[A] + 10"}),
    Change(2, {"X": 0.0}),
]


def run_before(
    sim: Simulation, model: RoadrunnerSBMLModel | None = None
) -> tuple[TimecourseResult, TimecourseResult]:
    """Run a simulation of the probe model with the states before the changes."""
    model = model or RoadrunnerSBMLModel(source=sbml())
    plan = compile_simulation(sim, model.symbols, model.uinfo)
    result, before, _ = execute_with_changes(plan, model, SEL)
    return result, before


def test_the_states_before_the_changes_after_the_start() -> None:
    """The state before a change is the end of the segment before it (#262)."""
    result, before = run_before(
        Simulation(end=2, changes=CHANGES, times=[0, 0.5, 1, 1.5, 2])
    )
    reference = run(Simulation(end=1, changes=CHANGES[:1], times=[0, 1]))
    # the output is the one of a run, the state after a change at its time
    np.testing.assert_array_equal(
        result.values,
        run(Simulation(end=2, changes=CHANGES, times=[0, 0.5, 1, 1.5, 2])).values,
    )
    assert result["[A]"][2] == pytest.approx(reference["[A]"][1] + 10.0, rel=1e-6)
    assert result["X"][-1] == pytest.approx(0.0)
    # the change at the start is the initial condition, it has no state before
    np.testing.assert_allclose(before.time, [1.0, 2.0])
    assert before["[A]"][0] == pytest.approx(reference["[A]"][1], rel=1e-6)
    assert before["X"][1] == pytest.approx(12.0)


def test_the_states_before_the_changes_from_a_negative_start() -> None:
    """The states of a simulation from a negative time in local time, shifted."""
    simulation = Simulation(
        start=-1,
        end=1,
        changes=[Change(0, {"[A]": "[A] + 10"})],
        times=[-1, 0, 1],
        time_shift=1,
    )
    result, before = run_before(simulation)
    reference = run(Simulation(end=1, times=[0, 1]))
    np.testing.assert_allclose(result.time, [0, 1, 2])
    np.testing.assert_allclose(before.time, [1.0])
    assert before["[A]"][0] == pytest.approx(reference["[A]"][1], rel=1e-6)
    assert result["[A]"][1] == pytest.approx(reference["[A]"][1] + 10.0, rel=1e-6)


def test_the_state_before_a_change_is_before_the_event_it_triggers() -> None:
    """An event a change triggers fires after the change, not before it."""
    model = RoadrunnerSBMLModel(source=sbml(COUNTER))
    simulation = Simulation(
        end=2, times=[0, 1, 2], changes=[Change(1, {"X": 2.0}), Change(2, {"X": 0.0})]
    )
    plan = compile_simulation(simulation, model.symbols, model.uinfo)
    result, before, _ = execute_with_changes(plan, model, ["time", "X", "Y"])
    assert np.asarray(result["X"]).tolist() == [0.0, 2.0, 0.0]
    assert np.asarray(result["Y"]).tolist() == [0.0, 1.0, 1.0]
    assert np.asarray(before["X"]).tolist() == [0.0, 2.0]
    assert np.asarray(before["Y"]).tolist() == [0.0, 1.0]


@pytest.mark.parametrize("times", [None, [0.5, 1, 2]])
def test_the_states_before_the_changes_next_to_the_output(
    times: list[float] | None,
) -> None:
    """The states before the changes after the start come with the output.

    The state before a change is the end of the integration up to it, the
    last row of a simulation which ends at the change: for the steps of the
    integrator the row roadrunner gives at the end.
    """
    model = RoadrunnerSBMLModel(source=sbml())
    simulation = Simulation(end=2, changes=CHANGES, times=times, time_shift=3)
    plan = compile_simulation(simulation, model.symbols, model.uinfo)
    assert plan.change_times == (1.0, 2.0)
    result, before, _ = execute_with_changes(plan, model, SEL)
    np.testing.assert_array_equal(result.values, execute(plan, model, SEL).values)
    np.testing.assert_allclose(before.time, [4.0, 5.0])
    first = run(Simulation(end=1, changes=CHANGES[:1], times=times and times[:2]))
    second = run(Simulation(end=2, changes=CHANGES[:2], times=times))
    np.testing.assert_allclose(before.values[0, 1:], first.values[-1, 1:], rtol=1e-12)
    np.testing.assert_allclose(before.values[1, 1:], second.values[-1, 1:], rtol=1e-12)
    # the output at the end is the state after the change at the end
    assert result["X"][-1] == pytest.approx(0.0)
    assert before["X"][1] == pytest.approx(12.0)


@pytest.mark.parametrize("times", [None, [0, 0.5, 1.5, 2]])
def test_the_states_after_the_changes_next_to_the_output(
    times: list[float] | None,
) -> None:
    """The state after a change is the start of the segment after it.

    The output has it at the time of the change if it has the time, the
    steps of the integrator always; an output without the time drops it.
    """
    model = RoadrunnerSBMLModel(source=sbml())
    simulation = Simulation(end=2, changes=CHANGES, times=times)
    plan = compile_simulation(simulation, model.symbols, model.uinfo)
    result, before, after = execute_with_changes(plan, model, SEL)
    np.testing.assert_array_equal(after.time, [1.0, 2.0])
    assert after["[A]"][0] == pytest.approx(before["[A]"][0] + 10.0)
    assert after["X"][1] == 0.0
    if times is None:
        rows = np.searchsorted(result.time, after.time)
        np.testing.assert_array_equal(result.values[rows], after.values)
    else:
        assert not np.isin(1.0, result.time)


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


def test_a_plan_without_changes_has_no_states_before_them() -> None:
    """Without a change after the start, there is no state before one."""
    model = RoadrunnerSBMLModel(source=sbml())
    plan = compile_simulation(
        Simulation(end=1, changes=CHANGES[:1], times=[0, 1]), model.symbols, model.uinfo
    )
    _, before, _ = execute_with_changes(plan, model, SEL)
    assert before.values.shape == (0, len(SEL))
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
    # the first steps are closer than the resolution of the time
    rows = steps.time >= 1.0
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
    states = []
    for times in (None, [0, 24, 48]):
        plan = compile_simulation(
            Simulation(end=48, changes=changes, times=times), model.symbols, model.uinfo
        )
        _, before, _ = execute_with_changes(plan, model, PK_SEL)
        states.append(before.values)
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


def test_segments_past_their_end_are_integrated_again_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plan runs twice however many segments roadrunner integrated past their end.

    A change of `k` to its own value 0.002 before every reset: the last step
    of roadrunner before every change ends after the reset. The rest of the
    plan from the first such segment on runs again with exact output.
    """
    runs = _count_runs(monkeypatch)
    model = RoadrunnerSBMLModel(source=sbml(SAWTOOTH))
    times = [j - 0.002 for j in range(1, 11)]
    result = run_on(
        model,
        Simulation(end=10.5, changes=[Change(times, {"k": 1.0})]),
        ["time", "X", "Y"],
    )
    assert len(runs) == 2
    at = np.isin(result.time, times)
    np.testing.assert_allclose(np.asarray(result["X"])[at], 0.998, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(result["Y"])[at], np.arange(10))
    assert result["Y"][-1] == 10.0


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
    times = _exact_times(
        0.0,
        np.array([0.5, 1.0, 1.0, 1.0 + 1e-12, 1.7, 1.98, 1.995]),
        2.0,
        0.01,
    )
    np.testing.assert_allclose(times, [0.0, 0.5, 0.99, 1.01, 1.7, 1.98, 2.0])


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
