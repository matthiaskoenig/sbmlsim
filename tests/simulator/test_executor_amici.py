"""The executor of AMICI runs a plan with the semantics of the executor of roadrunner.

Every test runs a plan on both engines and compares them on the output
times, see `tests.simulator.engine_comparison`; the steps of the integrator
of AMICI are a grid with the events, see `sbmlsim.simulator.executor_amici`.
"""

from collections.abc import Sequence
from pathlib import Path

import libsbml
import numpy as np
import pytest

pytest.importorskip("amici")

from sbmlsim.model.model_amici import AmiciSBMLModel
from sbmlsim.resources import (
    DEMO_SBML,
    HCTZ_SBML,
    MIDAZOLAM_SBML,
    REPRESSILATOR_SBML,
)
from sbmlsim.result import TimecourseResult
from sbmlsim.simulation import Change, Simulation, SteadyState
from sbmlsim.simulator import engines, executor_amici
from sbmlsim.simulator.executor import SteadyStateError
from tests.simulator import engine_comparison
from tests.simulator.engine_comparison import (
    AT_SIX,
    AT_THE_START,
    COUNTER,
    M1,
    PRESIMULATED,
    REARM,
    SEL,
    SETTINGS,
    STALE,
    TIME_EVENT,
    assert_equal_results,
    assert_equal_rows,
    plan_of,
    rows_at,
)
from tests.simulator.models import sbml, sbml_pk

#: an event of the start whose trigger is false before the time 0, which
#: assigns the state a change at the start sets
AT_THE_START_CHANGED = """
model start_changed
  X = INIT
  X' = 1
  Y = 0
  n = 0
  E: at (X >= 5), t0=false, fromTrigger=FROM: X = X + 100, Y = Y + 1, n = X
end
"""

#: events of the start in the order of their priorities, `F` triggered by `E`
CASCADE = """
model cascade
  X = 10
  X' = 1
  Y = 0
  n = 0
  E: at (X >= 5), t0=false, priority=PE: n = 2 * n + 1, Y = Y + 1
  F: at (Y > 0.5), t0=true, priority=1: n = 2 * n + 10
  G: at (X >= 8), t0=false, priority=PG: n = 3 * n + 100
end
"""

#: events of strict triggers of the time, false at a start or a change
STRICT_TIME = """
model strict_time
  X = 0
  X' = 1
  n = 0
  E: at (time > 0), t0=true: n = n + 1
  F: at (time > 1), t0=true: n = n + 10
  G: at (time >= 2), t0=true: n = n + 100
end
"""

#: an event of a strict trigger of a state, which a change sets to its
#: threshold
STRICT_STATE = """
model strict_state
  X = 0
  X' = 1
  n = 0
  F: at (X > 2), t0=true: n = n + 1
  G: at (X > 3 && X != 3.5), t0=true: n = n + 10
end
"""

#: a parameter whose initial assignment reads a parameter an event sets, which
#: keeps its initial value (AMICI-dev/AMICI#3506)
INITIAL_OF_AN_EVENT_TARGET = """
model initial_of_target
  X = 0
  X' = 1
  k = 1
  p = 2 * k
  E: at (X >= 2), t0=true: k = 5
end
"""


def both(
    source: str | Path, simulation: Simulation, selections: Sequence[str]
) -> tuple[TimecourseResult, TimecourseResult]:
    """Run a simulation with AMICI and with roadrunner, see `engine_comparison.both`."""
    return engine_comparison.both(
        AmiciSBMLModel, executor_amici.execute, source, simulation, selections
    )


def assert_agree(
    amici: TimecourseResult,
    roadrunner: TimecourseResult,
    times: list[float] | None,
    changes: Sequence[float] = (),
    events: Sequence[float] = (),
) -> None:
    """Compare an output of times on all rows, the steps at the changes, the events and the end."""
    if times is None:
        assert_equal_rows(amici, roadrunner, changes=changes, events=events)
    else:
        assert_equal_results(amici, roadrunner)


@pytest.mark.parametrize(
    ("source", "simulation"),
    [
        (REPRESSILATOR_SBML, Simulation(end=200, steps=100)),
        (DEMO_SBML, Simulation(end=100, steps=50)),
        (
            HCTZ_SBML,
            Simulation(
                end=24 * 60,
                steps=96,
                changes=[Change(60, {"PODOSE_hctz": 25.0})],
            ),
        ),
        (MIDAZOLAM_SBML, Simulation(end=24 * 60, steps=96)),
    ],
    ids=["repressilator", "demo", "hctz", "midazolam"],
)
def test_output_times_of_the_packaged_models(
    source: Path, simulation: Simulation
) -> None:
    """The packaged models agree with roadrunner at every output time."""
    selections = AmiciSBMLModel(source=source).selections
    assert selections is not None
    amici, roadrunner = both(source, simulation, selections)
    assert_equal_results(amici, roadrunner)


def test_the_steps_of_a_packaged_model() -> None:
    """The substitute of the steps: a grid per segment, two rows at a change."""
    selections = ["time", "[Cve_hctz]", "PODOSE_hctz"]
    simulation = Simulation(end=24 * 60, changes=[Change(60, {"PODOSE_hctz": 25.0})])
    amici, roadrunner = both(HCTZ_SBML, simulation, selections)
    assert amici.time.size == 2 * (executor_amici.STEPS + 1)
    np.testing.assert_allclose(
        amici.time[: executor_amici.STEPS + 1],
        np.linspace(0, 60, executor_amici.STEPS + 1),
    )
    assert_equal_rows(amici, roadrunner, changes=[60.0])


@pytest.mark.parametrize("times", [[0, 0.25, 0.5, 1, 2], None])
def test_a_constant_parameter_changed_after_the_start_takes_effect(
    times: list[float] | None,
) -> None:
    """A change of a constant parameter after the start changes the rates."""
    simulation = Simulation(end=2, changes=[Change(0.5, {"k1": 3.0})], times=times)
    amici, roadrunner = both(sbml(), simulation, SEL)
    assert np.all(np.asarray(amici["k1"])[amici.time > 0.5] == 3.0)
    assert_agree(amici, roadrunner, times, changes=[0.5])


def test_a_compartment_change_keeps_concentrations() -> None:
    """A change of a compartment keeps the concentrations of its species."""
    simulation = Simulation(end=2, changes=[Change(1, {"C": 5.0})], times=[0, 1, 2])
    amici, roadrunner = both(sbml(), simulation, [*SEL, "A", "B"])
    before, after = rows_at(amici, 1.0)
    np.testing.assert_allclose(after[1:3], before[1:3], rtol=1e-12)
    assert after[4] == 5.0
    np.testing.assert_allclose(after[6:8], before[6:8] / 2.0 * 5.0, rtol=1e-12)
    assert_equal_results(amici, roadrunner)


@pytest.mark.parametrize("times", [[0, 1, 2], None])
def test_a_change_at_an_output_time_has_two_rows(times: list[float] | None) -> None:
    """The output at a change after the start has the states before and after it."""
    simulation = Simulation(
        end=2, changes=[Change(1, {"[A]": "[A] + 10"})], times=times
    )
    amici, roadrunner = both(sbml(), simulation, SEL)
    before, after = rows_at(amici, 1.0)
    assert after[1] == pytest.approx(before[1] + 10.0, rel=1e-12)
    if times is not None:
        assert amici.time.tolist() == [0.0, 1.0, 1.0, 2.0]
    assert_agree(amici, roadrunner, times, changes=[1.0])


@pytest.mark.parametrize("times", [[0, 2, 3, 5], None])
def test_an_event_a_change_triggers_fires_after_it(times: list[float] | None) -> None:
    """An event whose trigger a change makes true fires once, after the change."""
    simulation = Simulation(end=5, changes=[Change(3, {"X": 10.0})], times=times)
    amici, roadrunner = both(sbml(M1), simulation, ["time", "X", "n"])
    before, after = rows_at(amici, 3.0)
    assert before[1:].tolist() == pytest.approx([5.0, 0.0])
    assert after[1:].tolist() == [0.0, 1.0]
    assert amici["n"][-1] == 1.0
    assert_agree(amici, roadrunner, times, changes=[3.0])


@pytest.mark.parametrize("times", [[0, 5, 10, 15], None])
def test_an_event_at_the_time_of_a_change_fires_after_it(
    times: list[float] | None,
) -> None:
    """An event and a change at one time: the change first, then the event."""
    simulation = Simulation(
        end=15,
        preinit_changes={"[S]": 2.0},
        changes=[Change(10, {"[S]": "C + [S]", "C": 8.0})],
        times=times,
    )
    amici, roadrunner = both(sbml(TIME_EVENT), simulation, ["time", "C", "[S]", "S"])
    before, after = rows_at(amici, 10.0)
    assert before[1] == 4.0
    assert after[1] == 16.0
    assert_agree(amici, roadrunner, times, changes=[10.0])


def test_an_event_of_the_model_is_two_rows_in_the_steps() -> None:
    """The steps have an event as two rows at its time, the states before and after it."""
    amici, roadrunner = both(sbml(M1), Simulation(end=10), ["time", "X", "n"])
    # X grows from 2 by 1, the event is at 6
    rows = amici.values[np.abs(amici.time - 6.0) <= 1e-9]
    assert rows.shape[0] == 2
    assert rows[0, 0] == rows[1, 0]
    np.testing.assert_allclose(rows[:, 1:], [[8.0, 0.0], [0.0, 1.0]], atol=1e-6)
    assert_equal_rows(amici, roadrunner, events=[6.0])


def test_an_event_of_the_model_at_the_end() -> None:
    """An event at the end is two rows in the steps and the state after it at a time."""
    steps, roadrunner = both(sbml(AT_SIX), Simulation(end=6), ["time", "X", "n"])
    assert steps.time[-2:].tolist() == [6.0, 6.0]
    np.testing.assert_allclose(
        steps.values[-2:, 1:], [[6.0, 0.0], [0.0, 1.0]], atol=1e-6
    )
    assert_equal_rows(steps, roadrunner, events=[6.0])
    times, roadrunner = both(
        sbml(AT_SIX), Simulation(end=6, times=[0, 6]), ["time", "n"]
    )
    assert times["n"].tolist() == [0.0, 1.0]
    assert_equal_results(times, roadrunner)


def test_an_event_of_the_model_at_an_output_time() -> None:
    """An output time at an event of the model has the state after it."""
    amici, roadrunner = both(
        sbml(AT_SIX), Simulation(end=10, times=[0, 3, 6, 10]), ["time", "X", "n"]
    )
    np.testing.assert_allclose(amici.values[2], [6.0, 0.0, 1.0], atol=1e-9)
    assert_equal_results(amici, roadrunner)


@pytest.mark.parametrize("times", [[0, 1], None])
def test_an_event_at_the_start(times: list[float] | None) -> None:
    """An event whose trigger is true at the start and false before it fires there."""
    amici, roadrunner = both(
        sbml(AT_THE_START), Simulation(end=1, times=times), ["time", "X", "n"]
    )
    assert amici.values[0].tolist() == [0.0, 0.0, 1.0]
    assert_agree(amici, roadrunner, times)
    np.testing.assert_array_equal(amici.values[0], roadrunner.values[0])


@pytest.mark.parametrize("init", [10, 0])
@pytest.mark.parametrize("change", [None, 0.0, 7.0, 10.0])
@pytest.mark.parametrize("from_trigger", ["true", "false"])
def test_an_event_at_the_start_fires_before_a_change_there(
    init: int, change: float | None, from_trigger: str
) -> None:
    """An event of the start fires in the initial state, before the change at the start.

    A change at the start which makes a trigger true or false does not fire
    an event, as in roadrunner.
    """
    model = AT_THE_START_CHANGED.replace("INIT", str(init)).replace(
        "FROM", from_trigger
    )
    changes = [] if change is None else [Change(0, {"X": change})]
    amici, roadrunner = both(
        sbml(model),
        Simulation(end=1, times=[0, 1], changes=changes),
        ["time", "X", "Y", "n"],
    )
    assert amici["n"][0] == (10.0 if init == 10 else 0.0)
    assert_equal_results(amici, roadrunner)


@pytest.mark.parametrize(("pe", "pg"), [(2, 3), (3, 2)])
@pytest.mark.parametrize("times", [[0, 1], None])
def test_events_of_the_start_fire_in_the_order_of_their_priorities(
    pe: int, pg: int, times: list[float] | None
) -> None:
    """The events of the start fire by priority, with the events they trigger."""
    model = CASCADE.replace("PE", str(pe)).replace("PG", str(pg))
    amici, roadrunner = both(
        sbml(model), Simulation(end=1, times=times), ["time", "X", "Y", "n"]
    )
    assert_agree(amici, roadrunner, times)
    np.testing.assert_array_equal(amici.values[0], roadrunner.values[0])


def test_an_event_of_a_change_at_the_start_after_a_presimulation() -> None:
    """The change at the start after the steady state triggers the event."""
    simulation = Simulation(
        end=1, steps=2, presimulation=SteadyState(), changes=[Change(0, {"X": 2.0})]
    )
    amici, roadrunner = both(sbml(COUNTER), simulation, ["time", "X", "Y"])
    assert amici["Y"].tolist() == [1.0, 1.0, 1.0]
    assert_equal_results(amici, roadrunner)


@pytest.mark.parametrize("times", [[0, 6, 12, 18, 24, 36], None])
def test_multiple_dosing(times: list[float] | None) -> None:
    """A dose at a vector of times is applied at each of them."""
    simulation = Simulation(
        end=36,
        changes=[
            Change(0, {"PODOSE": 100.0}),
            Change([6, 12, 18, 24], {"PODOSE": "PODOSE + 100"}),
        ],
        times=times,
    )
    amici, roadrunner = both(
        sbml_pk(urine=True), simulation, ["time", "[C]", "Aurine", "Aelim", "PODOSE"]
    )
    if times is not None:
        assert amici.time.tolist() == [0, 6, 6, 12, 12, 18, 18, 24, 24, 36]
    assert_agree(amici, roadrunner, times, changes=[6, 12, 18, 24])


@pytest.mark.parametrize("times", [[0, 3, 5], None])
@pytest.mark.parametrize(("t0", "fired"), [("true", 1.0), ("false", 2.0)])
def test_an_event_a_change_makes_true_fires_once(
    times: list[float] | None, t0: str, fired: float
) -> None:
    """A trigger which stays true across the restart after a change fires once."""
    simulation = Simulation(end=5, changes=[Change(3, {"X": 0.0})], times=times)
    amici, roadrunner = both(
        sbml(STALE.replace("T0", t0)), simulation, ["time", "X", "Z", "n"]
    )
    assert amici["n"][-1] == fired
    assert_agree(amici, roadrunner, times, changes=[3.0])


@pytest.mark.parametrize("times", [[0, 3, 6], None])
def test_an_event_whose_trigger_a_change_sets_back_fires_again(
    times: list[float] | None,
) -> None:
    """An event whose trigger a change makes false fires at its next crossing."""
    simulation = Simulation(end=6, changes=[Change(3, {"X": 0.0})], times=times)
    amici, roadrunner = both(sbml(REARM), simulation, ["time", "X", "n"])
    assert amici["n"][-1] == 2.0
    assert_agree(amici, roadrunner, times, changes=[3.0], events=[2.0, 5.0])


@pytest.mark.parametrize("times", [[0, 1, 3, 5], None])
def test_an_event_of_the_time_fires_after_a_presimulation(
    times: list[float] | None,
) -> None:
    """The time set back after the presimulation does not keep its triggers."""
    simulation = Simulation(end=5, presimulation=SteadyState(), times=times)
    amici, roadrunner = both(sbml(PRESIMULATED), simulation, ["time", "n"])
    assert amici["n"][0] == 1.0
    assert amici["n"][-1] == 2.0
    assert_agree(amici, roadrunner, times, events=[2.0])


@pytest.mark.parametrize("times", [[0, 0.5, 1, 1.5, 2.5], None])
def test_a_strict_trigger_of_the_time_at_a_start_or_a_change_fires_after_it(
    times: list[float] | None,
) -> None:
    """`time > 0` at the start and `time > 1` at a change fire right after them.

    AMICI evaluates a trigger at a root of the value 0 at the start of an
    integration as true; the executor fires the event after the row of the
    time, as roadrunner in its first step.
    """
    simulation = Simulation(end=2.5, changes=[Change(1, {"X": 5.0})], times=times)
    amici, roadrunner = both(sbml(STRICT_TIME), simulation, ["time", "X", "n"])
    assert amici["n"][0] == 0.0
    assert amici["n"][-1] == 111.0
    if times is not None:
        assert rows_at(amici, 1.0)[:, 2].tolist() == [1.0, 1.0]
    assert_agree(amici, roadrunner, times, changes=[1.0], events=[0.0, 2.0])


@pytest.mark.parametrize("value", [2.0, 3.5])
@pytest.mark.parametrize("times", [[0, 1, 3, 5], None])
def test_a_strict_trigger_of_a_state_set_to_its_threshold(
    value: float, times: list[float] | None
) -> None:
    """A change which sets a state to the threshold of a strict trigger fires it after it."""
    simulation = Simulation(end=5, changes=[Change(1, {"X": value})], times=times)
    amici, roadrunner = both(sbml(STRICT_STATE), simulation, ["time", "X", "n"])
    assert amici["n"][-1] == 11.0
    assert_agree(amici, roadrunner, times, changes=[1.0])


def test_the_relaxed_trigger_is_the_one_of_amici() -> None:
    """A trigger as AMICI evaluates it at a root: strict comparisons are not strict."""
    model = AmiciSBMLModel(source=sbml(STRICT_STATE))
    relaxed = [
        libsbml.formulaToL3String(
            libsbml.readMathMLFromString(executor_amici._relaxed(str(e.trigger)))
        )
        for e in model.symbols.events
    ]
    assert relaxed == ["X >= 2", "(X >= 3) && true"]


def test_formula_values_of_a_change_are_evaluated_simultaneously() -> None:
    """All values of a change are evaluated before any of them is assigned."""
    simulation = Simulation(
        end=1, changes=[Change(0.5, {"[A]": "[B]", "[B]": "[A]"})], times=[0.5, 1]
    )
    amici, roadrunner = both(sbml(), simulation, SEL)
    before, after = rows_at(amici, 0.5)
    assert after[1] == before[2]
    assert after[2] == before[1]
    assert_equal_results(amici, roadrunner)


@pytest.mark.parametrize("times", [[-1, 0, 1], None])
def test_negative_start_and_time_shift(times: list[float] | None) -> None:
    """A simulation from a negative time, its output shifted."""
    simulation = Simulation(
        start=-1,
        end=1,
        changes=[Change(0, {"[A]": "[A] + 10"})],
        times=times,
        time_shift=1,
    )
    amici, roadrunner = both(sbml(), simulation, SEL)
    assert amici.time[0] == 0.0
    assert amici.time[-1] == 2.0
    if times is not None:
        assert amici.time.tolist() == [0.0, 1.0, 1.0, 2.0]
    assert_agree(amici, roadrunner, times, changes=[1.0])


def test_a_time_dependent_model_from_a_negative_start() -> None:
    """A model which reads the time is integrated in absolute time."""
    model = "model t\n  X = 0\n  X' = time\nend"
    simulation = Simulation(start=-2, end=2, times=[-2, -1, 0, 1, 2])
    amici, roadrunner = both(sbml(model), simulation, ["time", "X"])
    np.testing.assert_allclose(amici["X"], [0.0, -1.5, -2.0, -1.5, 0.0], atol=1e-9)
    assert_equal_results(amici, roadrunner)


def test_changes_closer_than_the_integrator_resolves() -> None:
    """A segment too short to integrate keeps the state at its start."""
    simulation = Simulation(
        end=2,
        changes=[
            Change(1, {"k1": 2.0}),
            Change(1 + 2 * float(np.spacing(1.0)), {"k1": 3.0}),
        ],
        times=[0, 1, 2],
    )
    amici, roadrunner = both(sbml(), simulation, SEL)
    assert amici["k1"][-1] == 3.0
    assert_equal_results(amici, roadrunner)


def test_a_change_at_the_end() -> None:
    """A change at the end has the states before and after it, and its events."""
    simulation = Simulation(end=5, changes=[Change(5, {"X": 10.0})], times=[0, 5])
    amici, roadrunner = both(sbml(M1), simulation, ["time", "X", "n"])
    assert amici.time.tolist() == [0.0, 5.0, 5.0]
    assert amici["n"].tolist() == [0.0, 0.0, 1.0]
    assert_equal_results(amici, roadrunner)
    steps, roadrunner = both(
        sbml(M1), Simulation(end=5, changes=[Change(5, {"X": 10.0})]), ["time", "n"]
    )
    assert steps["n"][-1] == 1.0
    assert_equal_rows(steps, roadrunner, changes=[5.0])


def test_steady_state_presimulation_and_after_the_end() -> None:
    """The presimulation and the steady state after the end, at the time `inf`."""
    simulation = Simulation(
        end=1,
        preinit_changes={"b0": 0.0},
        presimulation=SteadyState(preinit_changes={"k1": 0.4}),
        changes=[Change(0, {"k1": 0.8})],
        times=[0, 1, np.inf],
    )
    amici, roadrunner = both(sbml(), simulation, SEL)
    assert amici.time[-1] == np.inf
    # A + B at steady state: A is k2 / (k1 + k2) of the total concentration 1
    assert amici["[A]"][0] == pytest.approx(0.6 / 1.0, rel=1e-6)
    assert amici["[A]"][-1] == pytest.approx(0.6 / 1.4, rel=1e-6)
    assert_equal_results(amici, roadrunner)


def test_steady_state_not_reached_raises() -> None:
    """A model which does not reach a steady state fails with the time."""
    growth = sbml("model g\n  x' = 1\n  x = 0\nend")
    model = AmiciSBMLModel(source=growth)
    plan = plan_of(growth, Simulation(end=1, presimulation=SteadyState(max_time=100)))
    with pytest.raises(SteadyStateError, match=r"'g'.*100"):
        executor_amici.execute(plan, model, ["time", "x"])


def test_a_failed_integration_raises_runtime_error() -> None:
    """A failure of the integrator raises with the messages of AMICI."""
    model = AmiciSBMLModel(source=sbml(), settings={"maximum_num_steps": 2})
    plan = plan_of(sbml(), Simulation(end=10, times=[0, 10]))
    with pytest.raises(RuntimeError, match=r"AMICI_TOO_MUCH_WORK.*mxstep"):
        executor_amici.execute(plan, model, SEL)


def test_selections_without_time_get_it() -> None:
    """`time` is the first column of the result."""
    model = AmiciSBMLModel(source=sbml())
    plan = plan_of(sbml(), Simulation(end=1, times=[0, 1]))
    result = executor_amici.execute(plan, model, ["[A]"])
    assert result.columns == ("time", "[A]")


def test_a_simulation_does_not_depend_on_the_one_before() -> None:
    """A simulation on a model gives the result of a fresh model."""
    source = sbml(M1)
    model = AmiciSBMLModel(source=source, settings=SETTINGS)
    first = plan_of(source, Simulation(end=7, changes=[Change(3, {"k": 2.0})]))
    second = plan_of(source, Simulation(end=10, times=[0, 5, 10]))
    executor_amici.execute(first, model, ["time", "X", "n"])
    again = executor_amici.execute(second, model, ["time", "X", "n"])
    fresh = executor_amici.execute(
        second, AmiciSBMLModel(source=source, settings=SETTINGS), ["time", "X", "n"]
    )
    np.testing.assert_array_equal(again.values, fresh.values)


@pytest.mark.parametrize("times", [[0, 1, 3, 4], None])
def test_an_initial_assignment_of_an_event_target_keeps_its_value(
    times: list[float] | None,
) -> None:
    """A parameter whose initial assignment reads a parameter an event sets keeps its value.

    AMICI makes such a parameter follow its initial assignment after the
    event (AMICI-dev/AMICI#3506); sbmlsim makes it a state of AMICI.
    """
    amici, roadrunner = both(
        sbml(INITIAL_OF_AN_EVENT_TARGET),
        Simulation(end=4, times=times),
        ["time", "X", "k", "p"],
    )
    assert amici["k"][-1] == 5.0
    assert amici["p"].tolist() == [2.0] * amici.time.size
    assert_agree(amici, roadrunner, times, events=[2.0])


#: a species of a conversion factor of the model and one of its own
CONVERSION = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="conversion" conversionFactor="cf">
    <listOfCompartments>
      <compartment id="c" spatialDimensions="3" size="2" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="3"
        hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialAmount="0"
        hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"
        conversionFactor="cfB"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="cf" value="0.5" constant="true"/>
      <parameter id="cfB" value="3" constant="true"/>
      <parameter id="k" value="0.7" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="J" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="2" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> k </ci><ci> A </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


@pytest.mark.parametrize("times", [[0, 0.5, 1, 2, 4], None])
def test_conversion_factors_scale_the_rates_as_in_roadrunner(
    times: list[float] | None,
) -> None:
    """The conversion factors of the model and of a species, also after a change of one."""
    simulation = Simulation(
        end=4,
        preinit_changes={"cf": 0.25},
        changes=[Change(1, {"cfB": 1.5})],
        times=times,
    )
    amici, roadrunner = both(CONVERSION, simulation, ["time", "A", "[A]", "B", "J"])
    assert amici["B"][-1] > 0.0
    assert_agree(amici, roadrunner, times, changes=[1.0])


def test_a_change_of_a_compartment_which_is_no_number_raises() -> None:
    """A compartment whose size is not a number is refused in a plan which sets it."""
    doc = libsbml.readSBMLFromString(sbml())
    doc.getModel().getCompartment("C").setSize(float("nan"))
    doc.getModel().getSpecies("A").setInitialAmount(1.0)
    doc.getModel().getSpecies("B").setInitialAmount(1.0)
    source = libsbml.writeSBMLToString(doc)
    model = AmiciSBMLModel(source=source)
    plan = plan_of(sbml(), Simulation(end=1, changes=[Change(0.5, {"C": 2.0})]))
    with pytest.raises(NotImplementedError, match=r"\['C'\].*not a number"):
        engines.check_plan(plan, model)
    with pytest.raises(NotImplementedError, match=r"\['C'\].*not a number"):
        executor_amici.execute(plan, model, ["time"])


def test_the_engine_runs_a_plan_with_the_executor_of_amici() -> None:
    """`execute_plan` and `check_plan` run a model of AMICI with its executor."""
    source = sbml()
    model = AmiciSBMLModel(source=source, settings=SETTINGS)
    plan = plan_of(source, Simulation(end=1, times=[0, 1]))
    engines.check_plan(plan, model)
    result = engines.execute_plan(plan, model, SEL)
    expected = executor_amici.execute(
        plan, AmiciSBMLModel(source=source, settings=SETTINGS), SEL
    )
    np.testing.assert_array_equal(result.values, expected.values)


def test_the_steps_of_events_which_fire_more_often_than_the_grid() -> None:
    """The intervals of the grid are split until every event is two rows at its time.

    `X` is set back to 0 at 0.003, so the event fires about three times in
    every interval of the grid of `STEPS` intervals.
    """
    model = "model sawtooth\n  X = 0\n  X' = 1\n  n = 0\n"
    model += "  E: at (X >= 0.003), t0=true: X = 0, n = n + 1\nend"
    amici, roadrunner = both(sbml(model), Simulation(end=1), ["time", "X", "n"])
    fired = int(amici["n"][-1])
    assert fired == roadrunner["n"][-1] == 333
    pairs = np.flatnonzero(np.diff(amici.time) == 0.0)
    assert pairs.size == fired
    np.testing.assert_allclose(amici["X"][pairs], 0.003, rtol=1e-6)
    np.testing.assert_allclose(amici["X"][pairs + 1], 0.0, atol=1e-8)
    np.testing.assert_array_equal(np.diff(amici["n"][pairs]), 1.0)
    np.testing.assert_allclose(amici.values[-1], roadrunner.values[-1], atol=1e-6)


def test_an_event_within_the_resolution_after_a_change_in_the_steps() -> None:
    """An event right after a change is two rows after the two rows of the change.

    AMICI fires the event in the integration after the change, not at its
    start, so the rows at the change are the ones of roadrunner.
    """
    model = "model near\n  X = 0\n  X' = 1\n  n = 0\n"
    model += "  E: at (time >= 1.000000000001), t0=true: n = n + 1, X = 10\nend"
    amici, roadrunner = both(
        sbml(model),
        Simulation(end=2, changes=[Change(1, {"X": 5.0})]),
        ["time", "X", "n"],
    )
    near_amici = amici.values[np.abs(amici.time - 1.0) < 1e-6]
    near_roadrunner = roadrunner.values[np.abs(roadrunner.time - 1.0) < 1e-6]
    assert near_amici[:, 0].tolist() == [1.0, 1.0, 1.000000000001, 1.000000000001]
    np.testing.assert_allclose(near_amici, near_roadrunner, rtol=1e-6, atol=1e-6)


#: a state whose rate is 0 at the time a change sets it to the threshold of a
#: strict trigger, `RISE` the time from which it rises
FLAT = """
model flat
  X = 0
  X' = RATE
  n = 0
  E: at (X > 2), t0=true: n = n + 1
end
"""


@pytest.mark.parametrize("times", [[0, 1, 3, 5], None])
def test_a_strict_trigger_at_its_threshold_with_a_rate_of_zero(
    times: list[float] | None,
) -> None:
    """The trigger becomes true later than the resolution after the change; it fires."""
    model = FLAT.replace("RATE", "(time - 1)^2")
    simulation = Simulation(end=5, changes=[Change(1, {"X": 2.0})], times=times)
    amici, roadrunner = both(sbml(model), simulation, ["time", "X", "n"])
    assert amici["n"][-1] == 1.0
    assert_agree(amici, roadrunner, times, changes=[1.0])


def test_a_strict_trigger_which_stays_at_its_threshold_raises() -> None:
    """A trigger at its threshold longer than the look ahead raises, naming the event.

    `X` stays 2 until 1.5 and rises after it: roadrunner fires the event
    after 1.5, AMICI would miss it.
    """
    model = FLAT.replace("RATE", "piecewise(0, time < 1.5, 1)")
    source = sbml(model)
    plan = plan_of(source, Simulation(end=5, changes=[Change(1, {"X": 2.0})]))
    with pytest.raises(NotImplementedError, match=r"\['E'\].*'flat'.*time 1"):
        executor_amici.execute(plan, AmiciSBMLModel(source=source), ["time", "n"])


@pytest.mark.parametrize("trigger", ["time > 10", "time >= 10", "time > 5"])
def test_a_strict_trigger_at_a_horizon_of_the_steady_state(trigger: str) -> None:
    """The horizons of a steady state start with the events AMICI misses at their threshold.

    The horizons end at 1, 10, 100, ... after the start of the presimulation.
    """
    model = f"model horizon\n  X = 1\n  X' = -X\n  n = 0\n  E: at ({trigger}), t0=true: n = n + 1\nend"
    simulation = Simulation(end=1, times=[0, 1], presimulation=SteadyState())
    amici, roadrunner = both(sbml(model), simulation, ["time", "X", "n"])
    assert_equal_results(amici, roadrunner)


@pytest.mark.parametrize("times", [[0, 5], None])
def test_a_strict_trigger_at_a_change_at_the_end(times: list[float] | None) -> None:
    """An event right after a change at the end has the state after it at the end."""
    model = "model last\n  X = 0\n  X' = 1\n  n = 0\n"
    model += "  E: at (time > 5), t0=true: n = n + 1, X = X + 100\nend"
    simulation = Simulation(end=5, changes=[Change(5, {"X": 1.0})], times=times)
    amici, roadrunner = both(sbml(model), simulation, ["time", "X", "n"])
    assert amici.values[-1].tolist() == [5.0, 101.0, 1.0]
    assert_agree(amici, roadrunner, times, changes=[5.0])
