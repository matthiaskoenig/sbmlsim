"""The executor of COPASI runs a plan with the semantics of the executor of roadrunner.

Every test runs a plan on both engines and compares them on the output
times: all rows of an output of times, and for the steps of the integrator,
which differ between the engines, the rows at the changes, at the events
and at the end.
"""

import contextlib
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
from sbmlsim.model.model_copasi import CopasiSBMLModel  # noqa: E402
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
from sbmlsim.simulator.plan import Plan, compile_simulation  # noqa: E402
from tests.simulator.models import sbml, sbml_pk  # noqa: E402

#: the tolerances of the integrators of both engines
SETTINGS = {"relative_tolerance": 1e-10, "absolute_tolerance": 1e-12}

#: the probe model of the executor of roadrunner
SEL = ["time", "[A]", "[B]", "X", "C", "k1"]

#: `X` grows by `k` and is set back to 0 at 8, `n` counts the events (the
#: model `m1` of the spike of the engine)
M1 = """
model m1
  compartment c = 1
  species X in c
  x0 = 1
  X = 2*x0
  k = 1
  X' = k
  n = 0
  E: at (X >= 8), t0=false: X = 0, n = n + 1
end
"""

#: an event of the time, which doubles `C` at 10
TIME_EVENT = """
model ev_time
  compartment C = 4
  species S in C = 3
  S' = 1
  E: at (time >= 10), t0=true: C = 2 * C
end
"""

#: an event of the time at 6, which sets `X` back to 0
AT_SIX = """
model at_six
  X = 0
  X' = 1
  n = 0
  E: at (time >= 6), t0=false: X = 0, n = n + 1
end
"""

#: an event of the time at the start, whose trigger is false before it
AT_THE_START = """
model at_start
  X = 0
  X' = 1
  n = 0
  E: at (time >= 0), t0=false: n = n + 1
end
"""

#: an event which counts the crossings of `X` above 1
COUNTER = """
model counter
  var X = 0
  var Y = 0
  E1: at (X > 1): Y = Y + 1
end
"""

#: a delayed event of the model, which COPASI schedules itself
DELAYED = """
model delayed
  X = 0
  X' = 1
  Y = 0
  E: at 1 after (X > 0.5), t0=true: Y = Y + 1
end
"""


def plan_of(source: str | Path, simulation: Simulation) -> Plan:
    """Compile a simulation against a model."""
    model = RoadrunnerSBMLModel(source=source)
    return compile_simulation(simulation, model.symbols, model.uinfo)


def both(
    source: str | Path, simulation: Simulation, selections: Sequence[str]
) -> tuple[TimecourseResult, TimecourseResult]:
    """Run a simulation with COPASI and with roadrunner.

    The simulation is compiled once and executed with both engines at the
    tolerances `SETTINGS`.

    Returns:
        The results of COPASI and of roadrunner.
    """
    roadrunner_model = RoadrunnerSBMLModel(source=source, settings=SETTINGS)
    copasi_model = CopasiSBMLModel(source=source, settings=SETTINGS)
    plan = compile_simulation(
        simulation, roadrunner_model.symbols, roadrunner_model.uinfo
    )
    copasi = executor_copasi.execute(plan, copasi_model, selections)
    roadrunner = execute(plan, roadrunner_model, selections)
    return copasi, roadrunner


def assert_equal_results(
    copasi: TimecourseResult, roadrunner: TimecourseResult
) -> None:
    """Compare two results of an output of times on all rows."""
    assert copasi.columns == roadrunner.columns
    np.testing.assert_array_equal(copasi.time, roadrunner.time)
    np.testing.assert_allclose(copasi.values, roadrunner.values, rtol=1e-6, atol=1e-9)


def rows_at(result: TimecourseResult, time: float) -> np.ndarray:
    """Get the rows of a result at a time."""
    return result.values[result.time == time]


def pairs_at(result: TimecourseResult, times: Sequence[float]) -> np.ndarray:
    """Get the two rows of a change or an event at times, and the last row.

    The pair of a time is its first row within `1e-6` and the row after it,
    the states before and after the change or the event.
    """
    rows = []
    for time in times:
        first = np.flatnonzero(np.abs(result.time - time) <= 1e-6)[0]
        rows.append(result.values[first : first + 2])
    rows.append(result.values[-1:])
    return np.concatenate(rows)


def assert_equal_rows(
    copasi: TimecourseResult,
    roadrunner: TimecourseResult,
    changes: Sequence[float] = (),
    events: Sequence[float] = (),
) -> None:
    """Compare the steps of the integrator at the changes, the events and the end.

    roadrunner gives the rows of an event of the model times within the
    resolution of the time after the event, see `sbmlsim.model.event_rows`,
    whose states differ by the rates times that offset.
    """
    assert copasi.columns == roadrunner.columns
    for times, atol in ((changes, 1e-9), (events, 1e-6)):
        c, r = pairs_at(copasi, times), pairs_at(roadrunner, times)
        np.testing.assert_allclose(c[:, 0], r[:, 0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(c[:, 1:], r[:, 1:], rtol=1e-6, atol=atol)


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


def test_a_delayed_event_fires_across_a_change() -> None:
    """COPASI schedules a delayed event of the model, also across a change."""
    simulation = Simulation(
        end=3, changes=[Change(1, {"X": 0.0})], times=[0, 1, 1.4, 1.6, 3]
    )
    copasi, roadrunner = both(sbml(DELAYED), simulation, ["time", "X", "Y"])
    assert copasi["Y"].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
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
        method.getParameter("Absolute Tolerance").setDblValue(1e-12)
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


def test_an_event_at_the_time_of_a_change_within_the_root_tolerance() -> None:
    """An event of the time at a change, within the root tolerance of the stop.

    LSODA, as CVODE, finds a root within about a hundred units in the last
    place of the time and the step of the integrator: the event fires at the
    stop before the change in both engines.
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
