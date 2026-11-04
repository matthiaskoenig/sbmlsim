"""The executor of COPASI runs a plan with the semantics of the executor of roadrunner.

Every test runs a plan on both engines and compares them on the output
times: all rows of an output of times, and for the steps of the integrator,
which differ between the engines, the rows at the changes, at the events
and at the end.
"""

import contextlib
import gc
import locale
from collections.abc import Generator, Sequence
from pathlib import Path

import numpy as np
import pytest

from sbmlsim.engine import import_copasi

with contextlib.suppress(ImportError):
    # COPASI imported with the locale of the process kept, see `import_copasi`
    import_copasi()
COPASI = pytest.importorskip("COPASI")

from sbmlsim.model import RoadrunnerSBMLModel  # noqa: E402
from sbmlsim.model.model_copasi import (  # noqa: E402
    COPASI_ATOL_FACTOR,
    CopasiSBMLModel,
)
from sbmlsim.resources import (  # noqa: E402
    DEMO_SBML,
    HCTZ_SBML,
    MIDAZOLAM_SBML,
    REPRESSILATOR_SBML,
)
from sbmlsim.result import TimecourseResult  # noqa: E402
from sbmlsim.simulation import Change, Simulation, SteadyState  # noqa: E402
from sbmlsim.simulator import executor_copasi  # noqa: E402
from sbmlsim.simulator.executor import (  # noqa: E402
    SteadyStateError,
    execute,
    segment_stop,
)
from sbmlsim.simulator.plan import compile_simulation, time_tolerance  # noqa: E402
from tests.simulator import engine_comparison  # noqa: E402
from tests.simulator.engine_comparison import (  # noqa: E402
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
from tests.simulator.models import sbml, sbml_pk  # noqa: E402

#: a delayed event of the model, which COPASI schedules itself
DELAYED = """
model delayed
  X = 0
  X' = 1
  Y = 0
  E: at 1 after (X > 0.5), t0=true: Y = Y + 1
end
"""


def both(
    source: str | Path, simulation: Simulation, selections: Sequence[str]
) -> tuple[TimecourseResult, TimecourseResult]:
    """Run a simulation with COPASI and with roadrunner, see `engine_comparison.both`."""
    return engine_comparison.both(
        CopasiSBMLModel, executor_copasi.execute, source, simulation, selections
    )


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
    selections = CopasiSBMLModel(source=source).selections
    assert selections is not None
    copasi, roadrunner = both(source, simulation, selections)
    assert_equal_results(copasi, roadrunner)


#: an oral dose of the HCTZ model after the start
HCTZ_DOSE = Simulation(
    end=24 * 60, steps=96, changes=[Change(60, {"PODOSE_hctz": 25.0})]
)


def test_releasing_a_model_does_not_change_another_model() -> None:
    """A released model removes its own data model of COPASI and nothing else.

    A model of HCTZ simulates bitwise the same before and after another model
    of HCTZ, loaded before it, is released.
    """
    released = CopasiSBMLModel(source=HCTZ_SBML, settings=SETTINGS)
    model = CopasiSBMLModel(source=HCTZ_SBML, settings=SETTINGS)
    plan = plan_of(HCTZ_SBML, HCTZ_DOSE)
    selections = model.selections
    assert selections is not None
    before = executor_copasi.execute(plan, model, selections)
    released.close()
    gc.collect()
    after = executor_copasi.execute(plan, model, selections)
    np.testing.assert_array_equal(after.values, before.values)


def test_a_model_loaded_after_a_release_agrees_with_roadrunner() -> None:
    """A model of HCTZ loaded after one was released simulates as the first.

    The release frees the memory of COPASI, which the next load reuses, so
    COPASI sums the rates of the reactions of a species in another order, see
    the module of the model: the two loads agree within the integration and
    not bitwise, and both agree with roadrunner. Measured at `SETTINGS`, 22
    loads after a release: at most 31 times `1e-12 + 1e-10 * |x|`, the
    tolerance of the settings, which is about 1000 times.
    """
    roadrunner_model = RoadrunnerSBMLModel(source=HCTZ_SBML, settings=SETTINGS)
    plan = compile_simulation(
        HCTZ_DOSE, roadrunner_model.symbols, roadrunner_model.uinfo
    )
    released = CopasiSBMLModel(source=HCTZ_SBML, settings=SETTINGS)
    selections = released.selections
    assert selections is not None
    first = executor_copasi.execute(plan, released, selections)
    released.close()
    gc.collect()
    model = CopasiSBMLModel(source=HCTZ_SBML, settings=SETTINGS)
    copasi = executor_copasi.execute(plan, model, selections)
    np.testing.assert_allclose(copasi.values, first.values, rtol=1e-7, atol=1e-10)
    assert_equal_results(copasi, execute(plan, roadrunner_model, selections))


@pytest.mark.parametrize("times", [[0, 0.25, 0.5, 1, 2], None])
def test_a_constant_parameter_changed_after_the_start_takes_effect(
    times: list[float] | None,
) -> None:
    """A change of a constant parameter after the start changes the rates."""
    simulation = Simulation(end=2, changes=[Change(0.5, {"k1": 3.0})], times=times)
    copasi, roadrunner = both(sbml(), simulation, SEL)
    after = np.asarray(copasi["k1"])[copasi.time > 0.5]
    assert np.all(after == 3.0)
    unchanged, _ = both(sbml(), Simulation(end=2, times=times), SEL)
    assert copasi["[A]"][-1] != pytest.approx(unchanged["[A]"][-1], rel=1e-3)
    if times is None:
        assert_equal_rows(copasi, roadrunner, changes=[0.5])
    else:
        assert_equal_results(copasi, roadrunner)


def test_a_compartment_change_keeps_concentrations() -> None:
    """A change of a compartment keeps the concentrations of its species."""
    simulation = Simulation(end=2, changes=[Change(1, {"C": 5.0})], times=[0, 1, 2])
    copasi, roadrunner = both(sbml(), simulation, [*SEL, "A", "B"])
    before, after = rows_at(copasi, 1.0)
    np.testing.assert_allclose(after[1:3], before[1:3], rtol=1e-12)
    assert after[4] == 5.0
    np.testing.assert_allclose(after[6:8], before[6:8] / 2.0 * 5.0, rtol=1e-12)
    assert_equal_results(copasi, roadrunner)


@pytest.mark.parametrize("times", [[0, 1, 2], None])
def test_a_change_at_an_output_time_has_two_rows(times: list[float] | None) -> None:
    """The output at a change after the start has the states before and after it."""
    simulation = Simulation(
        end=2, changes=[Change(1, {"[A]": "[A] + 10"})], times=times
    )
    copasi, roadrunner = both(sbml(), simulation, SEL)
    before, after = rows_at(copasi, 1.0)
    assert after[1] == pytest.approx(before[1] + 10.0, rel=1e-12)
    if times is None:
        assert_equal_rows(copasi, roadrunner, changes=[1.0])
    else:
        assert copasi.time.tolist() == [0.0, 1.0, 1.0, 2.0]
        assert_equal_results(copasi, roadrunner)


@pytest.mark.parametrize("times", [[0, 2, 3, 5], None])
def test_an_event_a_change_triggers_fires_after_it(times: list[float] | None) -> None:
    """An event whose trigger a change makes true fires after the change."""
    simulation = Simulation(end=5, changes=[Change(3, {"X": 10.0})], times=times)
    copasi, roadrunner = both(sbml(M1), simulation, ["time", "X", "n"])
    before, after = rows_at(copasi, 3.0)
    assert before[1:].tolist() == pytest.approx([5.0, 0.0])
    assert after[1:].tolist() == [0.0, 1.0]
    assert copasi["n"][-1] == 1.0
    if times is None:
        assert_equal_rows(copasi, roadrunner, changes=[3.0])
    else:
        assert_equal_results(copasi, roadrunner)


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
    copasi, roadrunner = both(sbml(TIME_EVENT), simulation, ["time", "C", "[S]", "S"])
    before, after = rows_at(copasi, 10.0)
    assert before[1] == 4.0
    assert after[1] == 16.0
    if times is None:
        assert_equal_rows(copasi, roadrunner, changes=[10.0])
    else:
        assert_equal_results(copasi, roadrunner)


def test_an_event_of_the_model_is_two_rows_in_the_steps() -> None:
    """The steps of the integrator have an event as two rows at its time."""
    copasi, roadrunner = both(sbml(M1), Simulation(end=10), ["time", "X", "n"])
    # X grows from 2 by 1, the event is at 6
    # COPASI locates the root of the trigger within the rounding of the time
    rows = copasi.values[np.abs(copasi.time - 6.0) <= 1e-12]
    assert rows.shape[0] == 2
    assert rows[0, 0] == rows[1, 0]
    np.testing.assert_allclose(rows[:, 1:], [[8.0, 0.0], [0.0, 1.0]], atol=1e-9)
    assert_equal_rows(copasi, roadrunner, events=[6.0])


def test_an_event_of_the_model_at_the_end() -> None:
    """An event at the end is two rows in the steps and the state after it at a time."""
    steps, roadrunner = both(sbml(AT_SIX), Simulation(end=6), ["time", "X", "n"])
    assert steps.time[-2:].tolist() == [6.0, 6.0]
    np.testing.assert_allclose(steps.values[-2:, 1:], [[6.0, 0.0], [0.0, 1.0]])
    assert_equal_rows(steps, roadrunner, events=[6.0])
    times, roadrunner = both(
        sbml(AT_SIX), Simulation(end=6, times=[0, 6]), ["time", "n"]
    )
    assert times["n"].tolist() == [0.0, 1.0]
    assert_equal_results(times, roadrunner)


def test_an_event_of_the_model_at_an_output_time() -> None:
    """An output time at an event of the model has the state after it."""
    copasi, roadrunner = both(
        sbml(AT_SIX), Simulation(end=10, times=[0, 3, 6, 10]), ["time", "X", "n"]
    )
    assert copasi.values[2].tolist() == [6.0, 0.0, 1.0]
    assert_equal_results(copasi, roadrunner)


@pytest.mark.parametrize("times", [[0, 1], None])
def test_an_event_at_the_start(times: list[float] | None) -> None:
    """An event whose trigger is true at the start and false before it fires there."""
    copasi, roadrunner = both(
        sbml(AT_THE_START), Simulation(end=1, times=times), ["time", "X", "n"]
    )
    assert copasi.values[0].tolist() == [0.0, 0.0, 1.0]
    if times is None:
        assert_equal_rows(copasi, roadrunner)
        np.testing.assert_array_equal(copasi.values[0], roadrunner.values[0])
    else:
        assert_equal_results(copasi, roadrunner)


def test_an_event_of_a_change_at_the_start_after_a_presimulation() -> None:
    """The change at the start after the steady state triggers the event."""
    simulation = Simulation(
        end=1, steps=2, presimulation=SteadyState(), changes=[Change(0, {"X": 2.0})]
    )
    copasi, roadrunner = both(sbml(COUNTER), simulation, ["time", "X", "Y"])
    assert copasi["Y"].tolist() == [1.0, 1.0, 1.0]
    assert_equal_results(copasi, roadrunner)


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
    copasi, roadrunner = both(
        sbml_pk(urine=True), simulation, ["time", "[C]", "Aurine", "Aelim", "PODOSE"]
    )
    if times is None:
        assert_equal_rows(copasi, roadrunner, changes=[6, 12, 18, 24])
    else:
        assert copasi.time.tolist() == [0, 6, 6, 12, 12, 18, 18, 24, 24, 36]
        assert_equal_results(copasi, roadrunner)


#: delayed events which COPASI schedules as roadrunner in a plan without
#: changes after the start: a delay, assignments of the trigger and of the
#: execution and a decay
DELAYED_EVENTS = {
    "delay": DELAYED,
    "values of the trigger": DELAYED.replace(
        "t0=true: Y = Y + 1", "t0=true, fromTrigger=true: Y = X"
    ),
    "values of the execution": DELAYED.replace(
        "t0=true: Y = Y + 1", "t0=true, fromTrigger=false: Y = X"
    ),
    "decay": """
    model decay
      X = 1
      X' = -X
      Y = 0
      E: at 2 after (X < 0.5), t0=true: Y = 1
    end
    """,
}


@pytest.mark.parametrize("name", DELAYED_EVENTS)
@pytest.mark.parametrize("times", [[0, 1, 1.4, 1.6, 2, 3, 5], [0, 2, 5, np.inf], None])
def test_delayed_events_without_writes(name: str, times: list[float] | None) -> None:
    """COPASI schedules the delayed events of a plan without writes itself.

    A change at the start is no write between two runs. roadrunner fires a
    delayed event late in the steps of the integrator (#1382), the values of
    its execution are compared in an output of times only.
    """
    if times is None and name == "values of the execution":
        pytest.skip("roadrunner #1382")
    source = sbml(DELAYED_EVENTS[name])
    for changes in ([], [Change(0, {"Y": 0.0})]):
        simulation = Simulation(end=5, changes=changes, times=times)
        copasi, roadrunner = both(source, simulation, ["time", "X", "Y"])
        if times is None:
            assert_equal_rows(copasi, roadrunner)
        else:
            assert_equal_results(copasi, roadrunner)


@pytest.mark.parametrize(("persistent", "fired"), [("false", 0.0), ("true", 1.0)])
def test_a_delayed_event_whose_trigger_becomes_false(
    persistent: str, fired: float
) -> None:
    """A non-persistent delayed event is cancelled, a persistent one fires.

    The trigger is true from 0.5 to 1.2, the event is due at 1.5. The output
    times keep the steps of CVODE small enough that roadrunner sees the
    trigger, see `test_known_difference_an_event_whose_trigger_is_true_within_a_step`.
    """
    source = sbml(
        DELAYED.replace(
            "(X > 0.5), t0=true:",
            f"(X > 0.5 && X < 1.2), t0=true, persistent={persistent}:",
        )
    )
    simulation = Simulation(end=5, times=[0, 1, 1.4, 1.6, 2, 3, 5])
    copasi, roadrunner = both(source, simulation, ["time", "X", "Y"])
    assert copasi["Y"][-1] == fired
    assert_equal_results(copasi, roadrunner)


@pytest.mark.parametrize("times", [[0, 1, 2, 3], None])
@pytest.mark.parametrize(
    ("changes", "presimulation", "feature"),
    [
        ([Change(1, {"X": 0.0})], None, "a change after the start"),
        ([Change(3, {"W": 1.0})], None, "a change after the start"),
        ([], SteadyState(max_time=1), "a presimulation"),
    ],
)
def test_a_delayed_event_with_writes_raises(
    times: list[float] | None,
    changes: list[Change],
    presimulation: SteadyState | None,
    feature: str,
) -> None:
    """A delayed event in a plan with a change after the start or a presimulation.

    COPASI keeps the event pending across the values the executor sets
    between two runs, where roadrunner cancels or schedules it again.
    """
    source = sbml(DELAYED.replace("Y = 0", "Y = 0\n  W = 0"))
    model = CopasiSBMLModel(source=source)
    plan = plan_of(
        source,
        Simulation(end=3, changes=changes, presimulation=presimulation, times=times),
    )
    with pytest.raises(NotImplementedError, match=rf"\['E'\].*{feature}"):
        executor_copasi.execute(plan, model, ["time", "Y"])


#: a delayed event of the time, `X` approaches 10, so a steady state follows
#: the end
DELAYED_AT = """
model delayed_at
  X = 0.5
  X' = 10 - X
  p = 0
  E: at DELAY after (time >= TRIGGER): p = p + 1
end
"""


@pytest.mark.parametrize("times", [[0, 2, 4], [0, 2, 4, np.inf], None])
@pytest.mark.parametrize(
    ("trigger", "delay"),
    [
        pytest.param(3.9, 0.2, id="pending across the end"),
        pytest.param(3.75, 0.25, id="at the end"),
        pytest.param(3.75, 0.25 + 1e-13, id="within the resolution after the end"),
        pytest.param(3.5, 0.49999999999999956, id="between the stop and the end"),
    ],
)
def test_a_delayed_event_pending_across_the_stop_before_the_end(
    trigger: float, delay: float, times: list[float] | None
) -> None:
    """A delayed event pending at the stop before the end fires as in roadrunner.

    The last segment stops just before the end and the events at the end are
    integrated from the stop (`event_at_the_end`), the steady state after the
    end from there; the states of the triggers are set from the current values
    before every run (`resetEventsFound`), which keeps the pending event. An
    event at the end, closer to the stop than LSODA integrates, fires at the
    start of the run.
    """
    # the numbers of the math exactly, antimony writes 15 digits
    source = (
        sbml(DELAYED_AT.replace("DELAY", "123.5").replace("TRIGGER", "654.5"))
        .replace(" 123.5 ", f" {delay!r} ")
        .replace(" 654.5 ", f" {trigger!r} ")
    )
    simulation = Simulation(start=0, end=4, times=times)
    copasi, roadrunner = both(source, simulation, ["time", "X", "p"])
    fired = 0.0 if trigger + delay > 4 + 1e-12 else 1.0
    if times is None:
        assert_equal_rows(copasi, roadrunner, events=[4.0] if fired else [])
        assert copasi["p"][-1] == fired
        if fired:
            # the states before and after the event at the end
            np.testing.assert_array_equal(copasi.time[-2:], [4.0, 4.0])
            np.testing.assert_array_equal(copasi["p"][-2:], [0.0, 1.0])
    else:
        assert_equal_results(copasi, roadrunner)
        assert copasi["p"][times.index(4)] == fired
        if np.inf in times:
            # the event pending across the end fires in the steady state
            assert copasi["p"][-1] == 1.0


def test_known_difference_an_event_whose_trigger_is_true_within_a_step() -> None:
    """Pin a known difference: roadrunner misses a trigger true within one step.

    `X > 0.5 && X < 0.8` is true for 0.3 of the time, within one step of
    CVODE, which finds the roots of both comparisons in it; roadrunner does
    not fire the event, COPASI does, as SBML.
    """
    model = "model narrow\n  X = 0\n  X' = 1\n  Y = 0\n"
    model += "  E: at (X > 0.5 && X < 0.8), t0=true: Y = Y + 1\nend"
    copasi, roadrunner = both(
        sbml(model), Simulation(end=3, times=[0, 3]), ["time", "Y"]
    )
    assert copasi["Y"].tolist() == [0.0, 1.0]
    assert roadrunner["Y"].tolist() == [0.0, 0.0]


def test_known_defect_lsoda_fails_on_a_delayed_event_in_its_steps() -> None:
    """Pin a known defect of COPASI: its steps fail on a delayed event.

    The event `X = 0` 0.3 after `X > 1` is scheduled at 1.3, which LSODA has
    stepped past when it sets the time as its critical time: COPASI fails
    (`TCRIT behind TCUR`), the executor raises its message; an output of
    times agrees with roadrunner.
    """
    model = "model sawtooth\n  X = 0\n  X' = 1\n  n = 0\n"
    model += "  E: at 0.3 after (X > 1), t0=true: X = 0, n = n + 1\nend"
    source = sbml(model)
    plan = plan_of(source, Simulation(end=5))
    with pytest.raises(RuntimeError, match="TCRIT"):
        executor_copasi.execute(plan, CopasiSBMLModel(source=source), ["time"])
    copasi, roadrunner = both(
        source, Simulation(end=5, times=[0, 1, 2, 3, 5]), ["time", "X", "n"]
    )
    assert_equal_results(copasi, roadrunner)


def test_a_pending_event_is_not_carried_into_the_next_simulation() -> None:
    """An event pending at the end of a simulation does not fire in the next one."""
    model = CopasiSBMLModel(source=sbml(DELAYED), settings=SETTINGS)
    first = plan_of(sbml(DELAYED), Simulation(end=1, times=[0, 1]))
    executor_copasi.execute(first, model, ["time", "Y"])
    second = plan_of(
        sbml(DELAYED), Simulation(end=2, preinit_changes={"X": -10.0}, times=[0, 2])
    )
    assert executor_copasi.execute(second, model, ["time", "Y"])["Y"].tolist() == [
        0.0,
        0.0,
    ]


def test_formula_values_of_a_change_are_evaluated_simultaneously() -> None:
    """All values of a change are evaluated before any of them is assigned."""
    simulation = Simulation(
        end=1, changes=[Change(0.5, {"[A]": "[B]", "[B]": "[A]"})], times=[0.5, 1]
    )
    copasi, roadrunner = both(sbml(), simulation, SEL)
    before, after = rows_at(copasi, 0.5)
    assert after[1] == before[2]
    assert after[2] == before[1]
    assert_equal_results(copasi, roadrunner)


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
    copasi, roadrunner = both(sbml(), simulation, SEL)
    assert copasi.time[0] == 0.0
    assert copasi.time[-1] == 2.0
    if times is None:
        assert_equal_rows(copasi, roadrunner, changes=[1.0])
    else:
        assert copasi.time.tolist() == [0.0, 1.0, 1.0, 2.0]
        assert_equal_results(copasi, roadrunner)


def test_a_time_dependent_model_from_a_negative_start() -> None:
    """A model which reads the time is integrated in absolute time."""
    model = "model t\n  X = 0\n  X' = time\nend"
    simulation = Simulation(start=-2, end=2, times=[-2, -1, 0, 1, 2])
    copasi, roadrunner = both(sbml(model), simulation, ["time", "X"])
    np.testing.assert_allclose(copasi["X"], [0.0, -1.5, -2.0, -1.5, 0.0], atol=1e-9)
    assert_equal_results(copasi, roadrunner)


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
    copasi, roadrunner = both(sbml(), simulation, SEL)
    assert copasi["k1"][-1] == 3.0
    assert_equal_results(copasi, roadrunner)


def test_a_change_at_the_end() -> None:
    """A change at the end has the states before and after it, and its events."""
    simulation = Simulation(end=5, changes=[Change(5, {"X": 10.0})], times=[0, 5])
    copasi, roadrunner = both(sbml(M1), simulation, ["time", "X", "n"])
    assert copasi.time.tolist() == [0.0, 5.0, 5.0]
    assert copasi["n"].tolist() == [0.0, 0.0, 1.0]
    assert_equal_results(copasi, roadrunner)
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
    copasi, roadrunner = both(sbml(), simulation, SEL)
    assert copasi.time[-1] == np.inf
    # A + B at steady state: A is k2 / (k1 + k2) of the total concentration 1
    assert copasi["[A]"][0] == pytest.approx(0.6 / 1.0, rel=1e-6)
    assert copasi["[A]"][-1] == pytest.approx(0.6 / 1.4, rel=1e-6)
    assert_equal_results(copasi, roadrunner)


def test_steady_state_not_reached_raises() -> None:
    """A model which does not reach a steady state fails with the time."""
    growth = sbml("model g\n  x' = 1\n  x = 0\nend")
    model = CopasiSBMLModel(source=growth)
    plan = plan_of(growth, Simulation(end=1, presimulation=SteadyState(max_time=100)))
    with pytest.raises(SteadyStateError, match=r"'g'.*100"):
        executor_copasi.execute(plan, model, ["time", "x"])


def test_a_failed_integration_raises_runtime_error() -> None:
    """A failure of the integrator raises with the message of COPASI."""
    model = CopasiSBMLModel(source=sbml(), settings={"maximum_num_steps": 2})
    plan = plan_of(sbml(), Simulation(end=10, times=[0, 10]))
    with pytest.raises(RuntimeError, match="MXSTEP"):
        executor_copasi.execute(plan, model, SEL)


def test_selections_without_time_get_it() -> None:
    """`time` is the first column of the result."""
    model = CopasiSBMLModel(source=sbml())
    plan = plan_of(sbml(), Simulation(end=1, times=[0, 1]))
    result = executor_copasi.execute(plan, model, ["[A]"])
    assert result.columns == ("time", "[A]")


@contextlib.contextmanager
def _decimal_comma() -> Generator[None]:
    """Set a numeric locale with a decimal comma, skip the test without one."""
    saved = locale.setlocale(locale.LC_NUMERIC)
    for name in ("de_DE.UTF-8", "de_DE.utf8", "en_DK.UTF-8", "en_DK.utf8"):
        try:
            locale.setlocale(locale.LC_NUMERIC, name)
        except locale.Error:
            continue
        if locale.localeconv()["decimal_point"] == ",":
            break
    else:
        locale.setlocale(locale.LC_NUMERIC, saved)
        pytest.skip("no locale with a decimal comma")
    try:
        yield
    finally:
        locale.setlocale(locale.LC_NUMERIC, saved)


def test_the_output_times_do_not_depend_on_the_locale() -> None:
    """The output times passed to COPASI are read the same with a decimal comma."""
    simulation = Simulation(end=1, times=[0, 1 / 3, 0.123456789123, 1])
    model = CopasiSBMLModel(source=sbml())
    plan = plan_of(sbml(), simulation)
    expected = executor_copasi.execute(plan, model, SEL)
    with _decimal_comma():
        result = executor_copasi.execute(plan, model, SEL)
    np.testing.assert_array_equal(result.time, [0, 0.123456789123, 1 / 3, 1])
    np.testing.assert_array_equal(result.values, expected.values)


def test_a_simulation_does_not_depend_on_the_one_before() -> None:
    """Two runs of a plan on one model are equal bitwise."""
    model = CopasiSBMLModel(source=sbml(M1), settings=SETTINGS)
    plan = plan_of(sbml(M1), Simulation(end=10, changes=[Change(3, {"X": 10.0})]))
    first = executor_copasi.execute(plan, model, ["time", "X", "n"])
    executor_copasi.execute(
        plan_of(sbml(M1), Simulation(end=4, preinit_changes={"k": 2.0})),
        model,
        ["time", "X"],
    )
    second = executor_copasi.execute(plan, model, ["time", "X", "n"])
    np.testing.assert_array_equal(first.values, second.values)


def test_the_event_of_the_targets_does_not_change_the_steps() -> None:
    """The event which makes the targets states does not change the steps of LSODA.

    The model is loaded into COPASI once with the event (`TARGETS_EVENT`) and
    once as it is; the steps of the integrator are equal bitwise.
    """
    model = CopasiSBMLModel(source=REPRESSILATOR_SBML, settings=SETTINGS)
    plan = plan_of(REPRESSILATOR_SBML, Simulation(end=100))
    selections = ["time", "PX", "PY", "PZ", "X", "Y", "Z"]
    with_event = executor_copasi.execute(plan, model, selections)

    dm = COPASI.CRootContainer.addDatamodel()
    try:
        assert dm.importSBMLFromString(model.sbml)
        cmodel = dm.getModel()
        assert cmodel.getNumEvents() == 0
        task = dm.getTask("Time-Course")
        task.setMethodType(COPASI.CTaskEnum.Method_deterministic)
        method = task.getMethod()
        method.getParameter("Relative Tolerance").setDblValue(1e-10)
        # the absolute tolerance the model gives LSODA
        method.getParameter("Absolute Tolerance").setDblValue(
            SETTINGS["absolute_tolerance"] * COPASI_ATOL_FACTOR
        )
        task.setUpdateModel(False)
        problem = task.getProblem()
        problem.setTimeSeriesRequested(False)
        problem.setOutputStartTime(0.0)
        problem.setDuration(100.0)
        problem.setOutputEvent(True)
        problem.setAutomaticStepSize(True)
        problem.setUseValues(False)
        species = {
            cmodel.getMetabolite(k).getSBMLId(): cmodel.getMetabolite(k)
            for k in range(cmodel.getNumMetabs())
        }
        handler = COPASI.CDataHandler()
        handler.addDuringName(
            COPASI.CRegisteredCommonName(
                cmodel.getObject(COPASI.CCommonName("Reference=Time"))
                .getCN()
                .getString()
            )
        )
        for sid in selections[1:]:
            handler.addDuringName(
                COPASI.CRegisteredCommonName(
                    species[sid].getValueReference().getCN().getString()
                )
            )
        dm.addInterface(handler)
        task.initializeRawWithOutputHandler(COPASI.CCopasiTask.OUTPUT_DURING, handler)
        assert task.processRaw(True)
        task.restore(True)
        dm.removeInterface(handler)
        rows = np.array(
            [list(handler.getNthRow(k)) for k in range(handler.getNumRowsDuring())]
        )
    finally:
        COPASI.CRootContainer.removeDatamodel(dm)
    without_event = model.output_transform(selections)(rows)
    np.testing.assert_array_equal(with_event.values, without_event)


@pytest.mark.parametrize(
    ("start", "end", "rounded"),
    [
        (0.0, 1.0, False),
        (-4898.619485211566, -4.0, False),
        (-4898.619485211566, -0.007312715117751976, True),
    ],
)
def test_a_run_of_copasi_ends_at_the_end(
    start: float, end: float, rounded: bool
) -> None:
    """COPASI ends a run at `start + duration`, the end where a sum reaches it.

    A start of a larger magnitude than the end reaches no sum at the end: the
    run ends at the closest time before the end, or after it.
    """
    before = start + executor_copasi._duration(start, end, past=False)
    after = start + executor_copasi._duration(start, end, past=True)
    if not rounded:
        assert before == end and after == end
    else:
        spacing = abs(float(np.spacing(start)))
        assert end - spacing < before < end < after < end + spacing


def test_a_negative_start_of_a_larger_magnitude_than_the_end() -> None:
    """A run from a start of a coarser rounding than the end ends at the end."""
    start, end = -4898.619485211566, -0.007312715117751976
    for times in ([start, -1.0, end], None):
        simulation = Simulation(
            start=start, end=end, changes=[Change(-1.0, {"k1": 3.0})], times=times
        )
        copasi, roadrunner = both(sbml(), simulation, SEL)
        assert copasi.time[-1] == end
        if times is None:
            assert_equal_rows(copasi, roadrunner, changes=[-1.0])
        else:
            assert_equal_results(copasi, roadrunner)


#: a time whose stop before it COPASI integrates to from `ROUNDED_FROM` as the
#: time itself, `ROUNDED_FROM + (stop - ROUNDED_FROM)`
ROUNDED_TO = 59.71883543307713
ROUNDED_FROM = 26.330557610867654


def test_the_stop_before_a_change_is_not_rounded_to_the_change() -> None:
    """The integration to the stop before a change ends at the stop.

    The output time before the stop is less than half of it, COPASI
    integrates from it to the stop to the time of the change; a time between
    them keeps the stop, whose row is the state before the change.
    """
    model = CopasiSBMLModel(source=sbml(), settings=SETTINGS)
    model.initialize([])
    course = executor_copasi._TimeCourse(model, ["time", "[A]"])
    stop = segment_stop(0.0, ROUNDED_TO)
    assert ROUNDED_FROM + (stop - ROUNDED_FROM) == ROUNDED_TO
    values = course.times(0.0, np.array([ROUNDED_FROM, stop]), ROUNDED_TO)
    assert values[:, 0].tolist() == [0.0, ROUNDED_FROM, stop]
    assert model.value("time") == stop


def test_known_defect_an_event_at_a_change_within_the_root_tolerance() -> None:
    """Pin a known defect of both engines: an event at a change fires before it.

    PEtab v2 applies the change first and then the event (`C` 4 before and
    16 after the change). LSODA, as CVODE, finds a root within about a
    hundred units in the last place of the time and the step of the
    integrator, so the event of the time fires at the stop one unit before
    the change, in both engines (`C` 8 before and after). The defect is in
    the executor of roadrunner as well (`segment_stop`) and out of the scope
    of the engine of COPASI; the test pins that both engines agree.
    """
    model = f"""
    model rounding
      C = 4
      X = 0
      X' = 1
      E: at (time >= {ROUNDED_TO!r}), t0=true: C = 2 * C
    end
    """
    simulation = Simulation(
        end=60,
        changes=[Change(ROUNDED_TO, {"C": 8.0})],
        times=[0, ROUNDED_FROM, ROUNDED_TO, 60],
    )
    copasi, roadrunner = both(sbml(model), simulation, ["time", "C", "X"])
    assert rows_at(copasi, ROUNDED_TO)[:, 1].tolist() == [8.0, 8.0]
    assert_equal_results(copasi, roadrunner)


@pytest.mark.parametrize("times", [[0, 3, 5], None])
@pytest.mark.parametrize(("t0", "fired"), [("true", 1.0), ("false", 2.0)])
def test_an_event_a_change_makes_true_fires_once(
    times: list[float] | None, t0: str, fired: float
) -> None:
    """COPASI does not fire an event again which the executor fired after a change."""
    simulation = Simulation(end=5, changes=[Change(3, {"X": 0.0})], times=times)
    copasi, roadrunner = both(
        sbml(STALE.replace("T0", t0)), simulation, ["time", "X", "Z", "n"]
    )
    assert copasi["n"][-1] == fired
    if times is None:
        assert_equal_rows(copasi, roadrunner, changes=[3.0])
    else:
        assert_equal_results(copasi, roadrunner)


@pytest.mark.parametrize("times", [[0, 3, 6], None])
def test_an_event_whose_trigger_a_change_sets_back_fires_again(
    times: list[float] | None,
) -> None:
    """An event whose trigger a change makes false fires at its next crossing."""
    simulation = Simulation(end=6, changes=[Change(3, {"X": 0.0})], times=times)
    copasi, roadrunner = both(sbml(REARM), simulation, ["time", "X", "n"])
    assert copasi["n"][-1] == 2.0
    if times is None:
        assert_equal_rows(copasi, roadrunner, changes=[3.0], events=[2.0, 5.0])
    else:
        assert_equal_results(copasi, roadrunner)


@pytest.mark.parametrize("times", [[0, 1, 3, 5], None])
def test_an_event_of_the_time_fires_after_a_presimulation(
    times: list[float] | None,
) -> None:
    """The time set back after the presimulation does not keep its triggers."""
    simulation = Simulation(end=5, presimulation=SteadyState(), times=times)
    copasi, roadrunner = both(sbml(PRESIMULATED), simulation, ["time", "n"])
    assert copasi["n"][0] == 1.0
    assert copasi["n"][-1] == 2.0
    if times is None:
        assert_equal_rows(copasi, roadrunner, events=[2.0])
    else:
        assert_equal_results(copasi, roadrunner)


def test_an_event_at_the_end_in_the_rounding_of_the_state() -> None:
    """A known difference: an event within the rounding of the state at the end.

    `X >= 8` of `X' = 1` from 2 at the end 6: roadrunner fires it before the
    end and its steps have one row at the end, the state after it; COPASI
    has the states before and after it at the end. The states at the end
    agree.
    """
    copasi, roadrunner = both(sbml(M1), Simulation(end=6), ["time", "X", "n"])
    np.testing.assert_allclose(
        copasi.values[-2:], [[6.0, 8.0, 0.0], [6.0, 0.0, 1.0]], atol=1e-9
    )
    assert roadrunner.time[-2] < 6.0
    np.testing.assert_allclose(copasi.values[-1], roadrunner.values[-1], atol=1e-9)


@pytest.mark.parametrize("change", [-0.007312715117751976, -3.3])
@pytest.mark.parametrize("times", [True, False])
def test_an_event_at_a_change_from_a_start_of_a_coarser_rounding(
    change: float, times: bool
) -> None:
    """The segment from a start of a larger magnitude stops before the change.

    COPASI ends a run of output times at its last time, the stop, not at
    `start + duration`, which may be after it.
    """
    start = -4898.619485211566
    model = f"""
    model coarse
      C = 4
      X = 0
      X' = 1
      E: at (time >= {change!r}), t0=true: C = 2 * C
    end
    """
    simulation = Simulation(
        start=start,
        end=0.0,
        changes=[Change(change, {"C": 8.0})],
        times=[start, change, 0.0] if times else None,
    )
    copasi, roadrunner = both(sbml(model), simulation, ["time", "C"])
    assert rows_at(copasi, change)[:, 1].tolist() == [4.0, 16.0]
    if times:
        assert_equal_results(copasi, roadrunner)
    else:
        assert_equal_rows(copasi, roadrunner, changes=[change])
    stopped = CopasiSBMLModel(source=sbml(model), settings=SETTINGS)
    stopped.initialize([])
    course = executor_copasi._TimeCourse(stopped, ["time", "C"])
    stop = segment_stop(start, change)
    course.times(start, np.array([stop]), change)
    assert stopped.value("time") == stop


#: the rate of `X` of the models at a threshold: held until 1.001, a plateau
#: after a change at 1, then rising
PLATEAU = "piecewise(0, time < 1.001, 1)"


def at_threshold(
    trigger: str,
    rate: str = PLATEAU,
    x0: float = 0.0,
    t0: str = "true",
    persistent: str = "true",
) -> str:
    """Get the SBML of a model whose event counts in `n` and keeps its first time in `T1`."""
    return sbml(
        f"""
        model at_threshold
          X = {x0}
          X' = {rate}
          n = 0
          T1 = -1
          E: at ({trigger}), t0={t0}, persistent={persistent}: n = n + 1, T1 = piecewise(time, n < 0.5, T1)
        end
        """
    )


@pytest.mark.parametrize("times", [[0, 1, 1.001, 1.002, 5], None])
@pytest.mark.parametrize(
    ("trigger", "rate", "x0", "t0", "persistent", "fired"),
    [
        ("X > 2", PLATEAU, 0.0, "true", "true", 1.0),
        ("X > 2", PLATEAU, 0.0, "false", "true", 1.0),
        ("X > 2", PLATEAU, 0.0, "true", "false", 1.0),
        ("X > 2", "piecewise(0, time < 1.5, 1)", 0.0, "true", "true", 1.0),
        ("X > 2", "1", 0.0, "true", "true", 1.0),
        ("X > 2", "piecewise(0, time < 1, 1)", 0.0, "true", "true", 1.0),
        ("X >= 2", PLATEAU, 0.0, "true", "true", 1.0),
        ("X < 2", "piecewise(0, time < 1.001, -1)", 3.0, "true", "true", 1.0),
        ("X > 2", "piecewise(0, time < 1.001, -1)", 5.0, "true", "true", 0.0),
        ("X <= 2", "piecewise(0, time < 1.001, -1)", 5.0, "false", "true", 1.0),
    ],
    ids=[
        "plateau",
        "plateau-t0-false",
        "plateau-not-persistent",
        "plateau-to-an-exact-switch",
        "rising",
        "rising-from-a-switch",
        "geq-at-the-change",
        "less-falling",
        "falling-away",
        "leq-falling",
    ],
)
def test_a_trigger_a_change_leaves_at_its_threshold_fires_when_it_turns(
    trigger: str,
    rate: str,
    x0: float,
    t0: str,
    persistent: str,
    fired: float,
    times: list[float] | None,
) -> None:
    """A trigger a change leaves exactly at its threshold fires when it turns later.

    The change sets `X = 2`; LSODA finds no root of a root function which
    is 0 when it starts, the comparison is shifted, see
    `CopasiSBMLModel.shift_comparisons`. `X >= 2` fires at the change.
    """
    simulation = Simulation(end=5, changes=[Change(1, {"X": 2.0})], times=times)
    copasi, roadrunner = both(
        at_threshold(trigger, rate, x0, t0, persistent),
        simulation,
        ["time", "X", "n", "T1"],
    )
    assert roadrunner["n"][-1] == fired
    assert copasi["n"][-1] == fired
    if times is None:
        assert_equal_rows(copasi, roadrunner, changes=[1.0])
    else:
        assert_equal_results(copasi, roadrunner)


@pytest.mark.parametrize("times", [[0, 1, 1.001, 1.002, 5], None])
def test_an_event_triggered_by_an_event_at_a_threshold(
    times: list[float] | None,
) -> None:
    """An event whose trigger reads the counter of an event a change left at its threshold."""
    model = """
    model chained
      X = 0
      X' = piecewise(0, time < 1.001, 1)
      n = 0
      m = 0
      E: at (X > 2), t0=false: n = n + 1
      F: at (n > 0.5), t0=false: m = m + 1
    end
    """
    simulation = Simulation(end=5, changes=[Change(1, {"X": 2.0})], times=times)
    copasi, roadrunner = both(sbml(model), simulation, ["time", "X", "n", "m"])
    assert copasi.values[-1, 2:].tolist() == [1.0, 1.0]
    if times is None:
        assert_equal_rows(copasi, roadrunner, changes=[1.0])
    else:
        assert_equal_results(copasi, roadrunner)


@pytest.mark.parametrize("times", [[0, 1, 1.001, 1.002, 5], None])
@pytest.mark.parametrize("t0", ["true", "false"])
def test_a_trigger_at_its_threshold_at_the_start(
    t0: str, times: list[float] | None
) -> None:
    """A trigger exactly at its threshold in the initial state fires when it turns."""
    copasi, roadrunner = both(
        at_threshold("X > 2", x0=2.0, t0=t0),
        Simulation(end=5, times=times),
        ["time", "X", "n", "T1"],
    )
    assert copasi["n"][-1] == 1.0
    if times is None:
        assert_equal_rows(copasi, roadrunner)
    else:
        assert_equal_results(copasi, roadrunner)


@pytest.mark.parametrize("times", [[0, 2, 3, 5], None])
def test_an_event_of_the_time_at_a_change_fires_right_after_it(
    times: list[float] | None,
) -> None:
    """`time > 2` with a change at 2 fires right after the change, as roadrunner."""
    simulation = Simulation(end=5, changes=[Change(2, {"X": 5.0})], times=times)
    copasi, roadrunner = both(
        at_threshold("time > 2", rate="1"), simulation, ["time", "X", "n", "T1"]
    )
    assert copasi["n"][-1] == 1.0
    # at most about one resolution of the time late, see `_threshold_shift`
    assert 2.0 < copasi["T1"][-1] <= 2.0 + 2.0 * time_tolerance(0.0, 5.0)
    if times is None:
        assert_equal_rows(copasi, roadrunner, changes=[2.0])
    else:
        assert_equal_results(copasi, roadrunner)


@pytest.mark.parametrize("times", [[0, 2, 5], None])
def test_a_trigger_copasi_fired_at_its_threshold_does_not_fire_again(
    times: list[float] | None,
) -> None:
    """A comparison COPASI found turned at its threshold keeps its value when others are shifted.

    `X > 2` is at its threshold in the initial state and rises, COPASI fires
    `E` at the initial values; `Y > 1` is at its threshold on a plateau,
    which shifts every comparison at its threshold: `X > 2` keeps its value
    after the crossing, `E` does not fire again, `F` fires when `Y` rises.
    """
    model = """
    model turned
      X = 2
      X' = 1
      Y = 1
      Y' = piecewise(0, time < 1, 1)
      n = 0
      m = 0
      E: at (X > 2), t0=true: n = n + 1
      F: at (Y > 1), t0=true: m = m + 1
    end
    """
    copasi, roadrunner = both(
        sbml(model), Simulation(end=5, times=times), ["time", "X", "Y", "n", "m"]
    )
    assert copasi.values[-1, 3:].tolist() == [1.0, 1.0]
    if times is None:
        assert_equal_rows(copasi, roadrunner)
    else:
        assert_equal_results(copasi, roadrunner)


#: models whose event sets a comparison of another trigger exactly to its
#: threshold in an integration of COPASI, `T1` the time `G` fires first
LANDING = {
    "set-to-the-threshold": """
    model internal
      X = 0
      X' = 1
      n = 0
      T1 = -1
      E: at (time > 1), t0=true: X = 2
      G: at (X > 2), t0=true: n = n + 1, T1 = piecewise(time, n < 0.5, T1)
    end
    """,
    "reset-to-zero": """
    model reset
      X = 5
      X' = 1
      n = 0
      T1 = -1
      E: at (time > 1), t0=true: X = 0
      G: at (X > 0), t0=true: n = n + 1, T1 = piecewise(time, n < 0.5, T1)
    end
    """,
    "reset-on-a-plateau": """
    model plateau
      X = 5
      X' = piecewise(0, time < 2, 1)
      n = 0
      T1 = -1
      E: at (time > 1), t0=true: X = 0
      G: at (X > 0), t0=true: n = n + 1, T1 = piecewise(time, n < 0.5, T1)
    end
    """,
    "dose-to-the-threshold": """
    model dose
      X = 0
      X' = -0.1*X
      n = 0
      T1 = -1
      D: at (time > 1), t0=true: X = X + 10
      G: at (X < 10), t0=true: n = n + 1, T1 = piecewise(time, n < 0.5, T1)
    end
    """,
    "two-resets": """
    model resets
      X = 5
      X' = 1
      n = 0
      T1 = -1
      E: at (time > 1), t0=true: X = 0
      F: at (time > 2.5), t0=true: X = 0
      G: at (X > 0), t0=true: n = n + 1, T1 = piecewise(time, n < 0.5, T1)
    end
    """,
}


@pytest.mark.parametrize("times", [[0, 1, 1.5, 3, 5], [0, 4, 5], None])
@pytest.mark.parametrize("name", LANDING)
def test_an_event_of_the_model_which_sets_a_threshold(
    name: str, times: list[float] | None
) -> None:
    """An event COPASI fires which sets another trigger exactly to its threshold.

    `G` fires when its comparison turns, right after the event or after a
    plateau, as in roadrunner: the run is integrated again in pieces, see
    `executor_copasi._TimeCourse._integrate`. A change at 0.5 is a write
    between two runs.
    """
    simulation = Simulation(end=5, changes=[Change(0.5, {"n": 0.0})], times=times)
    copasi, roadrunner = both(sbml(LANDING[name]), simulation, ["time", "X", "n", "T1"])
    assert copasi["n"][-1] == roadrunner["n"][-1] >= 1.0
    # at most one resolution of the time after roadrunner
    late = copasi["T1"][-1] - roadrunner["T1"][-1]
    assert late <= time_tolerance(0.0, 5.0)
    if times is None:
        assert_equal_rows(copasi, roadrunner, changes=[0.5])
    else:
        assert_equal_results(copasi, roadrunner)


def test_a_model_whose_events_set_no_threshold_runs_in_one_piece() -> None:
    """A run without a comparison at its threshold after an event is the run in one piece, bitwise."""
    model = CopasiSBMLModel(source=sbml(M1), settings=SETTINGS)
    assert model.lands
    plan = plan_of(sbml(M1), Simulation(end=20, times=[0, 5, 10, 20]))
    once = executor_copasi._TimeCourse._once
    calls: list[tuple[float, float]] = []

    def counted(
        course: executor_copasi._TimeCourse,
        start: float,
        end: float,
        past: bool,
        grid: np.ndarray | None,
        halvings: int = 0,
    ) -> np.ndarray:
        calls.append((start, end))
        return once(course, start, end, past, grid, halvings)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(executor_copasi._TimeCourse, "_once", counted)
        pieces = executor_copasi.execute(plan, model, ["time", "X", "n"])
    one = CopasiSBMLModel(source=sbml(M1), settings=SETTINGS)
    one.lands = False
    np.testing.assert_array_equal(
        pieces.values, executor_copasi.execute(plan, one, ["time", "X", "n"]).values
    )
    assert len(calls) > 2


@pytest.mark.parametrize("times", [[0, 1, 2, 5], None])
@pytest.mark.parametrize(
    ("threshold", "rate", "x0"), [("2", "1e-9", 0.0), ("0", "1e-12", 5.0)]
)
def test_a_slow_comparison_at_its_threshold_fires_in_time(
    threshold: str, rate: str, x0: float, times: list[float] | None
) -> None:
    """A comparison whose operands move slowly after a change sets them to its threshold fires in time, at the defaults.

    The shift is the change of the comparison over the resolution of the
    time, at least two units in the last place of the threshold, see
    `CopasiSBMLModel._threshold_shift`.
    """
    source = at_threshold(f"X > {threshold}", rate=rate, x0=x0)
    simulation = Simulation(
        end=5, changes=[Change(1, {"X": float(threshold)})], times=times
    )
    copasi_model = CopasiSBMLModel(source=source)
    roadrunner_model = RoadrunnerSBMLModel(source=source)
    plan = compile_simulation(
        simulation, roadrunner_model.symbols, roadrunner_model.uinfo
    )
    selections = ["time", "X", "n", "T1"]
    copasi = executor_copasi.execute(plan, copasi_model, selections)
    roadrunner = execute(plan, roadrunner_model, selections)
    assert copasi["n"][-1] == roadrunner["n"][-1] == 1.0
    assert copasi["T1"][-1] == pytest.approx(roadrunner["T1"][-1], abs=1e-6)


def test_the_states_of_the_roots_are_found_for_one_root() -> None:
    """The position of the states of the roots is told by their values, also for a single root.

    The shift of `time >= 2` probed far is the value of its shift parameter
    as well, which comes earlier in the values of COPASI.
    """
    model = CopasiSBMLModel(source=sbml(AT_SIX.replace("6", "2")), settings=SETTINGS)
    at = model._states_at
    assert at is not None
    model.initialize([])
    model.set_time(1.0)
    model.mc.resetEventsFound()
    assert model.mc.getValues().get(at) == 0.0
    model.set_time(3.0)
    model.mc.resetEventsFound()
    assert model.mc.getValues().get(at) == 1.0


#: a sawtooth whose reset `E` sets `G` to its threshold in every period
SAWTOOTH = """
model sawtooth
  X = 0
  X' = 1.1
  n = 0
  m = 0
  E: at (X > 1), t0=true: X = 0, n = n + 1
  G: at (X > 0), t0=true: m = m + 1
end
"""


@pytest.mark.parametrize("times", [[0, 0.5, 1, 1.5, 2.5], None])
@pytest.mark.parametrize(
    ("x0", "reset"),
    [(0.0, "E: at (X > 1), t0=true: X = 0"), (2.0, "E: at (X > 3), t0=true: X = 2")],
)
def test_a_trigger_at_its_threshold_at_the_start_of_a_model_which_resets_it(
    x0: float, reset: str, times: list[float] | None
) -> None:
    """The event COPASI fires at the initial values is kept in a model whose events reset the trigger.

    COPASI executes it at the start of the first integration, which the
    walk of the run in pieces repeats, so it is executed first, see
    `executor_copasi._TimeCourse._integrate`.
    """
    threshold = f"{x0:g}"
    model = f"""
    model reset
      X = {x0}
      X' = 1.1
      m = 0
      G: at (X > {threshold}), t0=true: m = m + 1
      {reset}
    end
    """
    copasi, roadrunner = both(
        sbml(model), Simulation(end=2.5, times=times), ["time", "X", "m"]
    )
    assert copasi["m"][-1] == roadrunner["m"][-1] == 3.0
    if times is None:
        assert_equal_rows(copasi, roadrunner)
    else:
        assert_equal_results(copasi, roadrunner)


@pytest.mark.parametrize("end", [5.0, 10.0])
@pytest.mark.parametrize("times", [True, False])
def test_a_change_at_the_time_of_an_event_of_the_model_which_resets(
    end: float, times: bool
) -> None:
    """A change at the time of an event of the model, whose run LSODA fails with a critical time behind its output time, is integrated in halves.

    See `executor_copasi._TimeCourse._once`.
    """
    model = """
    model reset
      X = 5
      X' = 1
      n = 0
      E: at (time > 1), t0=true: X = 0
      G: at (X > 0.5), t0=true: n = n + 1
    end
    """
    simulation = Simulation(
        end=end,
        changes=[Change(1, {"X": 2.0})],
        times=[0, 1, 2.5, end] if times else None,
    )
    copasi, roadrunner = both(sbml(model), simulation, ["time", "X", "n"])
    assert copasi.values[-1, 1:].tolist() == pytest.approx([end - 1.0, 1.0])
    if times:
        assert_equal_results(copasi, roadrunner)
    else:
        assert_equal_rows(copasi, roadrunner, changes=[1.0])


@pytest.mark.parametrize("times", [True, False])
def test_two_thousand_resets_of_a_threshold(times: bool) -> None:
    """A run with 2000 events which set a trigger to its threshold, walked in pieces without recursion."""
    end = 2000 / 1.1 + 0.5
    simulation = Simulation(end=end, times=[0.0, end / 2, end] if times else None)
    copasi = executor_copasi.execute(
        plan_of(sbml(SAWTOOTH), simulation),
        CopasiSBMLModel(source=sbml(SAWTOOTH), settings=SETTINGS),
        ["time", "n", "m"],
    )
    assert copasi.values[-1, 1:].tolist() == [2000.0, 2001.0]


def test_the_walk_of_a_run_is_linear_in_its_events() -> None:
    """The integrations of the walk of a run grow with the number of its events, not with its square."""
    once = executor_copasi._TimeCourse._integrate_once
    calls = [0]

    def counted(
        course: executor_copasi._TimeCourse,
        start: float,
        end: float,
        past: bool,
        grid: np.ndarray | None,
    ) -> np.ndarray:
        calls[0] += 1
        return once(course, start, end, past, grid)

    counts = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(executor_copasi._TimeCourse, "_integrate_once", counted)
        for periods in (50, 100, 200):
            calls[0] = 0
            simulation = Simulation(end=periods / 1.1 + 0.5)
            copasi = executor_copasi.execute(
                plan_of(sbml(SAWTOOTH), simulation),
                CopasiSBMLModel(source=sbml(SAWTOOTH), settings=SETTINGS),
                ["time", "m"],
            )
            assert copasi["m"][-1] == periods + 1
            counts.append(calls[0])
    # about two integrations per event
    assert counts[1] <= 2 * counts[0] + 4
    assert counts[2] <= 2 * counts[1] + 4
    assert counts[2] <= 3 * 200


#: an event of the time which moves its own trigger, so its model lands, see
#: `CopasiSBMLModel.lands`
MOVED = """
model moved
  tn = TN
  n = 0
  total = 0
  total' = -0.1 * total
  E: at (time >= tn), t0=false: total = total + 10, n = n + 1, tn = tn + 5
end
"""


@pytest.mark.parametrize("times", [True, False], ids=["times", "steps"])
@pytest.mark.parametrize("start", [-1.0, -3.3, -0.5, -1e-3, 0.0])
@pytest.mark.parametrize("tn", [0.0, 0.5])
def test_a_walk_from_a_negative_start_reaches_its_output_times(
    tn: float, start: float, times: bool
) -> None:
    """The walk of a run in pieces reaches an output time right after an event of the model with the steps of LSODA.

    LSODA exceeds its internal step limit on an output time it is set to
    one resolution after an event, see `executor_copasi._TimeCourse._piece`.
    roadrunner evaluates the triggers at the time 0 also from a negative
    start, a known difference for the event at 0, whose state after it
    COPASI has from the time 0.
    """
    source = sbml(MOVED.replace("TN", repr(tn)))
    grid = [start, *(t for t in (0.0, 0.5, 1.0, 2.0) if t > start)]
    simulation = Simulation(start=start, end=2, times=grid if times else None)
    copasi, roadrunner = both(source, simulation, ["time", "n", "total"])
    assert copasi["n"][-1] == 1.0
    if tn == 0.0 and start < 0.0:
        after = copasi.time > 0.0
        np.testing.assert_allclose(
            copasi["total"][after], 10.0 * np.exp(-0.1 * copasi.time[after]), rtol=1e-6
        )
    elif times:
        assert_equal_results(copasi, roadrunner)
    else:
        assert_equal_rows(copasi, roadrunner, events=[tn] if tn > start else [])
