"""The executor of AMICI runs a plan with the semantics of the executor of roadrunner.

Every test runs a plan on both engines and compares them on the output
times, see `tests.simulator.engine_comparison`; the steps of the integrator
of AMICI are a refined grid with the events, which interpolate as the steps
of roadrunner, see `sbmlsim.simulator.executor_amici`.
"""

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import libsbml
import numpy as np
import pytest

pytest.importorskip("amici")

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.model.model_amici import (
    SHIFT_PREFIX,
    AmiciSBMLModel,
    amici_sbml,
    shift_ids,
)
from sbmlsim.resources import (
    DEMO_SBML,
    HCTZ_SBML,
    MIDAZOLAM_SBML,
    REPRESSILATOR_SBML,
)
from sbmlsim.result import TimecourseResult
from sbmlsim.result.timecourse import interpolate
from sbmlsim.simulation import Change, Simulation, SteadyState
from sbmlsim.simulator import engines, executor_amici
from sbmlsim.simulator.executor import SteadyStateError, execute
from sbmlsim.simulator.plan import compile_simulation, time_tolerance
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
  G: at (X > 3), t0=true: n = n + 10
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


def assert_interpolates(
    result: TimecourseResult,
    source: str | Path,
    simulation: dict[str, Any],
    grid: np.ndarray,
    tolerance: float,
    changes: Sequence[float] = (),
    columns: Sequence[str] | None = None,
) -> None:
    """Interpolate the steps of the integrator onto a grid and compare with the exact output of roadrunner at the grid.

    Every column within `tolerance` of its largest magnitude, as the
    consumers of the steps interpolate them (`time=` of a run, the bands,
    `ScanResult.interpolate`); a column of a math which jumps, e.g. of a
    piecewise, is left out, no linear interpolation has its jump.

    Args:
        result: the steps of the integrator of AMICI.
        source: the model.
        simulation: the arguments of the `Simulation` of the result, which
            roadrunner runs with the grid as its output times.
        grid: the times.
        tolerance: the largest error relative to the largest magnitude.
        changes: the times of the changes after the start.
        columns: the columns compared, all for `None`.
    """
    model = RoadrunnerSBMLModel(source=source, settings=SETTINGS)
    # a change is two rows of the exact output, its time is left out
    grid = grid[~np.isin(grid, changes)]
    plan = compile_simulation(
        Simulation(times=grid.tolist(), **simulation), model.symbols, model.uinfo
    )
    exact = execute(plan, model, list(result.columns))
    assert np.array_equal(exact.time, grid)
    interpolated = interpolate(
        result.time,
        result.values,
        grid,
        time_tolerance(plan.start, plan.end),
        np.asarray(changes, dtype=float),
    )
    for k, column in enumerate(result.columns):
        if columns is not None and column not in columns:
            continue
        # the largest magnitude of the column, the one of the state after a
        # change included, which the grid leaves out
        scale = np.max(np.abs(result.values[:, k]))
        if scale == 0.0:
            np.testing.assert_array_equal(interpolated[:, k], 0.0, err_msg=column)
            continue
        error = np.max(np.abs(interpolated[:, k] - exact.values[:, k])) / scale
        assert error <= tolerance, f"{column}: {error:.2e} of its largest value"


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


@pytest.mark.parametrize("dose", ["PODOSE_hctz", "IVDOSE_hctz"])
def test_the_steps_of_a_packaged_model(dose: str) -> None:
    """The substitute of the steps: a refined grid per segment, two rows at a change, which interpolates as the steps of roadrunner.

    The grid of 100 intervals alone interpolates the concentration after an
    intravenous dose to `7.7e-1` of its peak, the steps of roadrunner to
    `3e-5`; the refined rows interpolate within `1e-5`.
    """
    selections = ["time", "[Cve_hctz]", "Cve_hctz", dose]
    arguments: dict[str, Any] = {
        "end": 24 * 60,
        "changes": [Change(60, {dose: 25.0})],
    }
    simulation = Simulation(**arguments)
    amici, roadrunner = both(HCTZ_SBML, simulation, selections)
    assert_equal_rows(amici, roadrunner, changes=[60.0])
    # the grid of the first segment, refined where the curves bend
    grid = amici.time[: executor_amici.STEPS + 1]
    assert np.all(np.isin(np.linspace(0, 60, executor_amici.STEPS + 1), grid))
    assert amici.time.size > 2 * (executor_amici.STEPS + 1)
    assert amici.time.size < 3 * roadrunner.time.size
    assert_interpolates(
        amici,
        HCTZ_SBML,
        arguments,
        np.linspace(0, 24 * 60, 2401),
        tolerance=1e-5,
        changes=[60.0],
    )


#: an oral dose with a lag, absorbed by a rate a piecewise of the time
#: switches on at `tlag`; `ka = 20` is absorbed within minutes
ORAL_LAG = """
model oral_lag
  compartment Vc = 10
  species A_gut = 0
  species C in Vc = 0
  ka = KA
  CL = 2
  tlag = 0.5
  ka_eff := piecewise(ka, time > tlag, 0)
  absorption: A_gut -> C; ka_eff * A_gut
  elimination: C -> ; CL * C
end
"""


@pytest.mark.parametrize("times", [[0, 0.25, 0.5, 0.75, 1, 2, 6, 24], None])
@pytest.mark.parametrize("ka", [1.2, 20.0])
def test_an_oral_dose_with_a_lag(ka: float, times: list[float] | None) -> None:
    """A fast absorption after a lag: the row at `tlag` has the rate as written, the steps interpolate within `1e-5`.

    An output row exactly at the time the piecewise switches has the rate
    and the flux of the SBML, `ka_eff = 0` at `tlag` for `time > tlag`, as
    roadrunner, not the ones AMICI evaluates after the root.
    """
    source = sbml(ORAL_LAG.replace("KA", repr(ka)))
    arguments: dict[str, Any] = {"end": 24, "changes": [Change(0, {"A_gut": 100.0})]}
    simulation = Simulation(times=times, **arguments)
    selections = ["time", "[C]", "A_gut", "ka_eff", "absorption"]
    amici, roadrunner = both(source, simulation, selections)
    assert_agree(amici, roadrunner, times)
    if times is not None:
        at_lag = rows_at(amici, 0.5)
        assert at_lag.shape[0] == 1
        assert at_lag[0, 3] == 0.0 and at_lag[0, 4] == 0.0
        np.testing.assert_array_equal(at_lag[0, 3:], rows_at(roadrunner, 0.5)[0, 3:])
    else:
        assert_interpolates(
            amici,
            source,
            arguments,
            np.linspace(0, 24, 2401),
            tolerance=1e-5,
            columns=["[C]", "A_gut"],
        )


#: an infusion of the length `tend` by a piecewise of the time in an
#: assignment rule, the rate of the flux into `C`
INFUSION_RATE = """
model infusion_rate
  compartment Vc = 10
  species C in Vc = 0
  Ri = 5
  tend = 1
  CL = 2
  Rate := piecewise(Ri, time <= tend, 0)
  Jin: -> C; Rate
  elimination: C -> ; CL * C
end
"""


@pytest.mark.parametrize("times", [[0, 0.5, 1, 1.5, 2, 4], None])
def test_the_row_at_the_end_of_an_infusion_has_the_rate_as_written(
    times: list[float] | None,
) -> None:
    """The row at `tend` has `Rate = Ri` and `Jin = Ri`, the values of `time <= tend` as written, as roadrunner.

    A change at `tend` ends a segment exactly there in both output modes.
    """
    simulation = Simulation(end=4, times=times, changes=[Change(1, {"CL": 3.0})])
    selections = ["time", "[C]", "Rate", "Jin", "CL"]
    amici, roadrunner = both(sbml(INFUSION_RATE), simulation, selections)
    before, after = rows_at(amici, 1.0)
    assert before[2] == 5.0 and before[3] == 5.0
    assert after[2] == 5.0 and after[3] == 5.0
    assert_agree(amici, roadrunner, times, changes=[1.0])
    np.testing.assert_allclose(rows_at(amici, 1.0), rows_at(roadrunner, 1.0), rtol=1e-6)


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


@pytest.mark.parametrize("value", [2.0, 3.0])
@pytest.mark.parametrize("times", [[0, 1, 3, 5], None])
def test_a_strict_trigger_of_a_state_set_to_its_threshold(
    value: float, times: list[float] | None
) -> None:
    """A change which sets a state to the threshold of a strict trigger fires it after it.

    `X = 2` is the threshold of `F`, `X = 3` the one of `G`, which the
    change triggers `F` with; both fire once.
    """
    simulation = Simulation(end=5, changes=[Change(1, {"X": value})], times=times)
    amici, roadrunner = both(sbml(STRICT_STATE), simulation, ["time", "X", "n"])
    assert amici["n"][-1] == 11.0
    assert_agree(amici, roadrunner, times, changes=[1.0])


def test_the_strict_comparisons_of_the_triggers_are_shifted() -> None:
    """Every strict comparison of a trigger is compiled shifted by a parameter of AMICI."""
    model = AmiciSBMLModel(source=sbml(STRICT_STATE))
    formula = libsbml.formulaToL3String
    comparisons = [
        (
            c.event,
            formula(libsbml.readMathMLFromString(c.left)),
            formula(libsbml.readMathMLFromString(c.right)),
        )
        for c in model.comparisons
    ]
    assert comparisons == [(0, "X", "2"), (1, "X", "3")]
    assert shift_ids(model.comparisons) == [
        f"{SHIFT_PREFIX}0_0",
        f"{SHIFT_PREFIX}1_0",
    ]
    doc = libsbml.readSBMLFromString(
        amici_sbml(model.sbml, model.symbols, model.label).sbml
    )
    triggers = [
        formula(e.getTrigger().getMath()) for e in doc.getModel().getListOfEvents()
    ]
    assert triggers[:2] == [
        f"X > (2 + {SHIFT_PREFIX}0_0)",
        f"X > (3 + {SHIFT_PREFIX}1_0)",
    ]
    free = [str(p) for p in model.amici_model.get_free_parameter_ids()]
    assert set(shift_ids(model.comparisons)) <= set(free)


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


#: a model without states, of parameters and an assignment rule
STATELESS = """
model stateless
  a = 2; b = 3
  y := a^2 * b^(-1)
end
"""


@pytest.mark.parametrize("times", [[0, 0.5, 1], None])
def test_a_model_without_states(times: list[float] | None) -> None:
    """AMICI has no states of the model, its rules follow the changes."""
    simulation = Simulation(end=1, changes=[Change(0.5, {"a": 1.0})], times=times)
    amici, roadrunner = both(sbml(STATELESS), simulation, ["time", "a", "b", "y"])
    assert amici["y"][-1] == pytest.approx(1 / 3)
    assert_agree(amici, roadrunner, times, changes=[0.5])


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


@pytest.mark.parametrize("times", [[0, 1, 1.4, 1.6, 5], None])
def test_a_strict_trigger_which_stays_at_its_threshold_fires_when_it_rises(
    times: list[float] | None,
) -> None:
    """A trigger at its threshold which stays there fires when it becomes true.

    `X` stays 2 until 1.5 and rises after it: the shifted comparison makes
    AMICI find the crossing at 1.5, where roadrunner fires the event.
    """
    model = FLAT.replace("RATE", "piecewise(0, time < 1.5, 1)")
    simulation = Simulation(end=5, changes=[Change(1, {"X": 2.0})], times=times)
    amici, roadrunner = both(sbml(model), simulation, ["time", "X", "n"])
    assert amici["n"][-1] == roadrunner["n"][-1] == 1.0
    if times is not None:
        assert amici["n"].tolist() == [0, 0, 0, 0, 1, 1]
    if times is None:
        # roadrunner fires a hair before 1.5, after the tiny steps of CVODE
        # at the switch of the piecewise
        rows = amici.values[np.abs(amici.time - 1.5) <= 1e-6]
        np.testing.assert_allclose(rows[:, 1:], [[2, 0], [2, 1]], atol=1e-6)
    assert_agree(amici, roadrunner, times, changes=[1.0])


@pytest.mark.parametrize(
    ("trigger", "fired"),
    [
        ("time > 10", 1.0),
        ("time >= 10", 1.0),
        ("time > 5", 1.0),
        ("time > 1", 2.0),
        ("time > 1.0000000001", 2.0),
    ],
)
def test_a_strict_trigger_of_the_time_at_a_horizon_of_the_steady_state(
    trigger: str, fired: float
) -> None:
    """A trigger of the time fires once in the presimulation, also at the end of a horizon.

    The horizons end at 1, 10, 100, ... after the start of the presimulation,
    the time is continuous across them; after the presimulation the time is
    set back to 0 and `time > 1` fires again after the end 1. AMICI fires a
    strict trigger at its root at the end of a horizon, the next horizon does
    not fire it again (`_Course.horizon`); roadrunner fires it once as well
    (#444, `sbmlsim.simulator.executor._EventAtTheHorizon`).
    """
    model = f"model horizon\n  X = 1\n  X' = -X\n  n = 0\n  E: at ({trigger}), t0=true: n = n + 1\nend"
    simulation = Simulation(end=1, times=[0, 1], presimulation=SteadyState())
    amici, roadrunner = both(sbml(model), simulation, ["time", "X", "n"])
    assert amici["n"][-1] == roadrunner["n"][-1] == fired
    np.testing.assert_allclose(amici["X"], roadrunner["X"], rtol=1e-6, atol=1e-9)


#: a state which a change sets to the threshold of a strict trigger and which
#: stays there until 1.001, the event records its first time
LATER = """
model later
  X = 0
  X' = piecewise(0, time < 1.001, 1)
  n = 0
  T1 = -1
  E: at (X > 2), t0=true: n = n + 1, T1 = piecewise(time, n < 0.5, T1), X = 0
end
"""


@pytest.mark.parametrize("times", [[0, 1, 1.0005, 1.002, 5], None])
def test_a_strict_trigger_at_its_threshold_which_becomes_true_later(
    times: list[float] | None,
) -> None:
    """The event fires at the time its trigger becomes true, within the resolution.

    The trigger stays at its threshold for 1e-3 after the change at 1, longer
    than the resolution of the time: the time it becomes true is located by
    bisection and the integration fires the event there, as roadrunner.
    """
    simulation = Simulation(end=5, changes=[Change(1, {"X": 2.0})], times=times)
    amici, roadrunner = both(sbml(LATER), simulation, ["time", "X", "n", "T1"])
    resolution = 5e-9
    assert amici["T1"][-1] == pytest.approx(1.001, abs=resolution)
    assert amici["T1"][-1] == pytest.approx(roadrunner["T1"][-1], abs=resolution)
    assert amici["n"][-1] == roadrunner["n"][-1]
    if times is None:
        pairs = np.flatnonzero(np.diff(amici.time) == 0.0)
        assert amici.time[pairs[1]] == pytest.approx(1.001, abs=resolution)
        np.testing.assert_allclose(
            amici.values[pairs[1] : pairs[1] + 2, 1:3],
            [[2.0, 0.0], [0.0, 1.0]],
            atol=1e-8,
        )
        np.testing.assert_allclose(amici.values[-1], roadrunner.values[-1], atol=1e-6)
    else:
        np.testing.assert_allclose(
            amici.values, roadrunner.values, rtol=1e-6, atol=1e-8
        )


@pytest.mark.parametrize("times", [[0, 5], None])
def test_a_strict_trigger_at_a_change_at_the_end(times: list[float] | None) -> None:
    """An event right after a change at the end has the state after it at the end."""
    model = "model last\n  X = 0\n  X' = 1\n  n = 0\n"
    model += "  E: at (time > 5), t0=true: n = n + 1, X = X + 100\nend"
    simulation = Simulation(end=5, changes=[Change(5, {"X": 1.0})], times=times)
    amici, roadrunner = both(sbml(model), simulation, ["time", "X", "n"])
    assert amici.values[-1].tolist() == [5.0, 101.0, 1.0]
    assert_agree(amici, roadrunner, times, changes=[5.0])


#: an event which sets its own strict trigger to its threshold, where it stays
#: until 20 (`Y` rises from 2 after 20): roadrunner fires it again after 20
HOLD = """
model hold
  Y = 3
  Y' = piecewise(0, time < 20, piecewise(1, Y < 3, 0))
  n = 0
  E: at ((time > 10) && (Y > 2)), t0=true: n = n + 1, Y = piecewise(2, n < 0.5, 10)
end
"""


@pytest.mark.parametrize("times", [[0, 10, 15, 25, 30], None])
@pytest.mark.parametrize("changes", [[], [Change(10, {"n": 0.0})]])
def test_an_event_which_leaves_its_strict_trigger_at_its_threshold_fires_again(
    times: list[float] | None, changes: list[Change]
) -> None:
    """An event which sets its trigger back to its threshold fires again when it crosses it.

    The event fires at 10, in an integration of AMICI or at the start of a
    segment after a change, and leaves `Y` at 2, the threshold of `Y > 2`,
    which AMICI evaluates as true: the shifted comparison makes it find the
    crossing at 20, where roadrunner fires the event again.
    """
    simulation = Simulation(end=30, times=times, changes=changes)
    amici, roadrunner = both(sbml(HOLD), simulation, ["time", "Y", "n"])
    assert amici["n"][-1] == roadrunner["n"][-1] == 2.0
    assert amici["Y"][-1] == 10.0
    if times is not None:
        assert amici["n"].tolist() == (
            [0, 0, 0, 1, 2, 2] if changes else [0, 0, 1, 2, 2]
        )
    else:
        # roadrunner fires the second event a hair before 20, after the tiny
        # steps of CVODE at the switch of the piecewise
        rows = amici.values[np.abs(amici.time - 20.0) <= 1e-6]
        np.testing.assert_allclose(rows[:, 1:], [[2, 1], [10, 2]], atol=1e-6)
    assert_agree(
        amici, roadrunner, times, changes=[10.0] if changes else [], events=[10.0]
    )


def test_an_event_at_a_horizon_of_the_steady_state_which_sets_its_trigger_back() -> (
    None
):
    """An event at the end of a horizon which leaves its trigger at its threshold fires again when it crosses it.

    `E` fires at 10, the end of a horizon of the presimulation, and sets `Y`
    to the threshold of `Y > 2`, which rises above it at 20: both engines
    fire it twice.
    """
    model = HOLD.replace("Y = 3", "X = 1\n  X' = -X\n  Y = 3")
    simulation = Simulation(end=1, times=[0, 1], presimulation=SteadyState())
    amici, roadrunner = both(sbml(model), simulation, ["time", "X", "Y", "n"])
    assert amici["n"][-1] == roadrunner["n"][-1] == 2.0
    assert_equal_results(amici, roadrunner)


#: an event which sets a state to the threshold of a strict trigger of
#: another event, which it crosses right after
CROSSING = """
model crossing
  Y = 0
  Y' = 1
  n = 0
  E: at (time > 1), t0=true: Y = 2
  F: at (Y > 2), t0=true: n = n + 1
end
"""


@pytest.mark.parametrize("times", [[0, 0.5, 2, 3], None])
def test_an_event_which_sets_a_strict_trigger_to_its_threshold(
    times: list[float] | None,
) -> None:
    """The trigger crosses right after the event of AMICI, which fires its event, as roadrunner.

    Both engines fire the second event right after the first: the steps of
    AMICI have three rows at 1, the states before, between and after them.
    """
    amici, roadrunner = both(
        sbml(CROSSING), Simulation(end=3, times=times), ["time", "Y", "n"]
    )
    assert amici["n"][-1] == 1.0
    if times is None:
        rows = amici.values[np.abs(amici.time - 1.0) <= 1e-9]
        np.testing.assert_allclose(rows, [[1, 1, 0], [1, 2, 0], [1, 2, 1]], atol=1e-8)
        np.testing.assert_allclose(amici.values[-1], roadrunner.values[-1], atol=1e-6)
    else:
        assert_equal_results(amici, roadrunner)


#: a dose at a strict trigger of the time, as a model of pharmacokinetics
DOSE = """
model dose
  A = 0
  A' = -0.3 * A
  n = 0
  D: at (time OP 12), t0=true: A = A + 10, n = n + 1
end
"""


@pytest.mark.parametrize("times", [[0, 6, 12, 24], None])
@pytest.mark.parametrize("op", [">", ">="])
def test_a_trigger_of_the_time_at_an_output_time(
    times: list[float] | None, op: str
) -> None:
    """An output time at the root of a trigger of the time.

    The output time 12 is a stop of the integration where AMICI finds the
    root: a strict trigger (`time > 12`) fires after it, so the row at 12 has
    the state before the event, as in roadrunner; one which is not strict
    fires at 12, the row has the state after it. The steps have the event as
    two rows at 12.
    """
    simulation = Simulation(end=24, times=times)
    amici, roadrunner = both(
        sbml(DOSE.replace("OP", op)), simulation, ["time", "A", "n"]
    )
    assert amici["n"][-1] == 1.0
    if times is not None:
        assert rows_at(amici, 12.0)[0, 2] == (0.0 if op == ">" else 1.0)
    else:
        rows = amici.values[np.abs(amici.time - 12.0) <= 1e-9]
        assert rows.shape[0] == 2
        np.testing.assert_allclose(rows[:, 1:], [[0.0, 0.0], [10.0, 1.0]], atol=1e-9)
    assert_agree(amici, roadrunner, times, events=[12.0])


@pytest.mark.parametrize("times", [[0, 0.5, 2], None])
def test_two_events_closer_than_the_resolution_fire_one_after_the_other(
    times: list[float] | None,
) -> None:
    """Two roots a little apart are two events, each with the state after the one before.

    Both events assign `n`; fired one after the other `n` is 11, fired as
    simultaneous events with the values at the trigger it would be 10. AMICI
    1.2 computes the trigger time of `F` as 1, the executor fires it after
    `E` with the state after it.
    """
    model = "model close\n  X = 0\n  X' = 1\n  n = 0\n"
    model += "  E: at (time >= 1), t0=true: n = n + 1\n"
    model += "  F: at (time >= 1 + 1e-8), t0=true: n = n + 10\nend"
    amici, roadrunner = both(
        sbml(model), Simulation(end=2, times=times), ["time", "X", "n"]
    )
    assert amici["n"][-1] == roadrunner["n"][-1] == 11.0
    assert_agree(amici, roadrunner, times, events=[1.0])


REPEATS = 3


def _integrations(source: str, simulation: Simulation) -> tuple[int, float]:
    """Run a simulation with AMICI, counting its integrations and the time it takes.

    The simulation runs once without being timed and then `REPEATS` times, the
    time is the one of the fastest run and the integrations those of one run.
    """
    model = AmiciSBMLModel(source=source, settings=SETTINGS)
    plan = plan_of(source, simulation)
    integrate = executor_amici._Course._integrate
    calls = 0

    def counted(self: executor_amici._Course, start: float, times: np.ndarray) -> Any:
        nonlocal calls
        calls += 1
        return integrate(self, start, times)

    executor_amici._Course._integrate = counted
    seconds = []
    try:
        executor_amici.execute(plan, model, ["time", "X", "n"])
        for _ in range(REPEATS):
            calls = 0
            started = time.perf_counter()
            result = executor_amici.execute(plan, model, ["time", "X", "n"])
            seconds.append(time.perf_counter() - started)
    finally:
        executor_amici._Course._integrate = integrate
    assert abs(result["n"][-1] - simulation.end * 1000.0) <= 1.0
    # the fastest of the runs, which a busy machine delays the least
    return calls, min(seconds)


@pytest.mark.parametrize("times", [[0, 1, 2], None])
def test_the_cost_is_linear_in_the_number_of_events(times: list[float] | None) -> None:
    """A model with periodic events integrates once per event, and the time grows linearly.

    `X` is set back to 0 at 0.001: 1000 events per unit of the time. The
    integrations of a run are counted, so the test does not depend on the
    speed of the machine (about three per event, and one per piece between
    two events which checks the refinement of the steps); the time of 200
    events is at most a few times the time of 100.
    """
    model = "model sawtooth\n  X = 0\n  X' = 1\n  n = 0\n"
    model += "  E: at (X >= 0.001), t0=true: X = 0, n = n + 1\nend"
    source = sbml(model)
    hundred, t_hundred = _integrations(
        source, Simulation(end=0.1, times=None if times is None else [0, 0.05, 0.1])
    )
    two_hundred, t_two_hundred = _integrations(
        source, Simulation(end=0.2, times=None if times is None else [0, 0.1, 0.2])
    )
    assert hundred <= 4 * 100 + 10
    assert two_hundred <= 2 * hundred + 10
    assert t_two_hundred <= 4.0 * t_hundred + 0.05


@pytest.mark.parametrize("times", [list(range(0, 43, 3)), None])
def test_many_doses_at_strict_triggers_of_the_time(times: list[float] | None) -> None:
    """Twenty doses at `time > 1, 3, ..., 39`, every one a root at the end of a chunk.

    The grid of the steps has a time within the rounding of a dose (21 at
    the time 20.999999999999996), for which AMICI gives a row of NaN as the
    last output time of an integration; the executor integrates the row
    again, see `_Course._integrate`.
    """
    doses = "\n  ".join(
        f"D{k}: at (time > {2 * k + 1}), t0=true: A = A + 10, n = n + 1"
        for k in range(20)
    )
    model = f"model doses\n  A = 0\n  A' = -0.3 * A\n  n = 0\n  {doses}\nend"
    amici, roadrunner = both(
        sbml(model), Simulation(end=42, times=times), ["time", "A", "n"]
    )
    assert amici["n"][-1] == roadrunner["n"][-1] == 20.0
    assert_agree(amici, roadrunner, times, events=[1.0, 21.0, 39.0])


#: an infusion which a piecewise of the time ends at `tend`, with a dose at
#: `tend`, after which the integration starts again at the switch
INFUSION = """
model infusion
  A = 0
  tend = 10
  A' = piecewise(2, CONDITION, 0) - 0.1 * A
  n = 0
  D: at (time >= tend), t0=true: A = A + 5, n = n + 1
end
"""

#: a rate which a piecewise of the time switches at 10, with an event or a
#: change there
SWITCH_OF_THE_TIME = """
model switch_of_the_time
  A = 0
  A' = piecewise(1, CONDITION, 0)
  p = 0
  EVENT
end
"""

#: a rate which a piecewise of the time switches at the start 0, the lag 0
SWITCH_AT_THE_START = """
model switch_at_the_start
  A = 0
  tlag = 0
  A' = piecewise(1, CONDITION, 0) + 0.5
end
"""

#: a rate which a piecewise of the difference of a state and a parameter
#: switches, the parameter set to the state by a change
SWITCH_OF_A_DIFFERENCE = """
model switch_of_a_difference
  A = 1
  A' = 1
  B = 100
  X = 0
  X' = piecewise(1, CONDITION, 0)
end
"""

#: a rate which a piecewise of a state switches, the state set to the
#: threshold of the comparison by a change
SWITCH_OF_A_STATE = """
model switch_of_a_state
  A = 0
  B = 0
  B' = 1
  A' = RATE
end
"""


def assert_equal_after_event(
    amici: TimecourseResult, roadrunner: TimecourseResult, marker: str
) -> None:
    """Compare the first rows after an event of the model, the ones at which a marker is 1.

    roadrunner steps within the resolution of the time before an event at a
    switch of a piecewise, which `assert_equal_rows` would pair with the row
    of the event.
    """

    def first(result: TimecourseResult) -> np.ndarray:
        return result.values[result[marker] == 1.0][0]

    np.testing.assert_allclose(first(amici), first(roadrunner), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("times", [[0, 10, 20, 30], None])
@pytest.mark.parametrize(
    "condition", ["time <= tend", "time - tend <= 0", "(time - tend) * 1e10 <= 0"]
)
def test_an_infusion_ended_by_a_piecewise_at_the_time_of_a_dose(
    condition: str, times: list[float] | None
) -> None:
    """`piecewise(2, time <= tend, 0)` is 0 after `tend`, from which the integration starts again after the dose.

    AMICI evaluates the comparison at its threshold at the start of an
    integration as written, `time <= tend` true, and keeps it; the executor
    shifts it, so AMICI finds the crossing right after the start, see the
    module. The shift of `time - tend <= 0` is sized by the terms `time` and
    `tend`, not by the operands 0, so the explicit root of AMICI at
    `tend + shift` is after `tend`; the one of the product by its change
    over the resolution of the time, so the root is not absorbed by `1e10`.
    """
    simulation = Simulation(end=30, times=times)
    model = INFUSION.replace("CONDITION", condition)
    amici, roadrunner = both(sbml(model), simulation, ["time", "A", "n"])
    assert amici["n"][-1] == 1.0
    np.testing.assert_allclose(amici["A"][-1], roadrunner["A"][-1], rtol=1e-6)
    assert amici["A"][-1] < 3.0
    assert_agree(amici, roadrunner, times)
    assert_equal_after_event(amici, roadrunner, "n")


@pytest.mark.parametrize("times", [[0, 5, 10, 15, 20], None])
@pytest.mark.parametrize("at", ["event", "change"])
@pytest.mark.parametrize(
    "condition",
    [
        "time <= 10",
        "time < 10",
        "time >= 10",
        "time > 10",
        "time - 10 > 0",
        "time - 10 <= 0",
    ],
)
def test_a_piecewise_of_the_time_switches_at_the_time_of_an_event_or_a_change(
    condition: str, at: str, times: list[float] | None
) -> None:
    """The rate switches at 10 as roadrunner evaluates it, also from a start at 10.

    An event or a change at 10 starts the integration again at the time of
    the switch, where every comparison of the time is at its threshold,
    `time - 10` compared with 0 as well.
    """
    event = "E: at (time >= 10), t0=true: p = 1" if at == "event" else ""
    model = SWITCH_OF_THE_TIME.replace("CONDITION", condition).replace("EVENT", event)
    changes = [Change(10, {"p": 1.0})] if at == "change" else []
    simulation = Simulation(end=20, times=times, changes=changes)
    amici, roadrunner = both(sbml(model), simulation, ["time", "A", "p"])
    np.testing.assert_allclose(amici["A"][-1], 10.0, rtol=1e-6)
    assert amici["p"][-1] == 1.0
    assert_agree(amici, roadrunner, times, changes=[10.0] if at == "change" else [])
    if at == "event":
        assert_equal_after_event(amici, roadrunner, "p")


@pytest.mark.parametrize("times", [[0, 5, 12, 20], None])
@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        ("piecewise(1, B <= 10, 0)", 5.0),
        ("piecewise(1, B < 10, 0)", 5.0),
        ("piecewise(1, B >= 10, 0)", 15.0),
        ("piecewise(1, B > 10, 0)", 15.0),
        ("piecewise(0, B == 10, 1)", 20.0),
        ("piecewise(1, B != 10, 0)", 20.0),
    ],
    ids=["leq", "lt", "geq", "gt", "eq", "neq"],
)
def test_a_piecewise_of_a_state_set_to_its_threshold_by_a_change(
    rate: str, expected: float, times: list[float] | None
) -> None:
    """A comparison of a state at its threshold after a change keeps its value until the state moves.

    `B` is set to 10 at 5 and rises from there: `B <= 10` and `B == 10` are
    true at 5 and false right after it, `B > 10` and `B != 10` false and
    true, as roadrunner evaluates them at every step.
    """
    model = SWITCH_OF_A_STATE.replace("RATE", rate)
    simulation = Simulation(end=20, times=times, changes=[Change(5, {"B": 10.0})])
    amici, roadrunner = both(sbml(model), simulation, ["time", "A", "B"])
    np.testing.assert_allclose(amici["A"][-1], expected, rtol=1e-6)
    assert_agree(amici, roadrunner, times, changes=[5.0])


@pytest.mark.parametrize("output", ["times", "steps"])
@pytest.mark.parametrize(
    ("condition", "end", "expected"),
    [
        ("time > tlag", 20.0, 30.0),
        ("time >= tlag", 20.0, 30.0),
        ("time <= tlag", 20.0, 10.0),
        ("time < tlag", 20.0, 10.0),
        ("time / 86400 > tlag", 172800.0, 259200.0),
        ("time * 1e-6 > tlag", 20.0, 30.0),
    ],
    ids=["gt", "geq", "leq", "lt", "days", "micro"],
)
def test_a_piecewise_of_the_time_at_the_start(
    condition: str, end: float, expected: float, output: str
) -> None:
    """A comparison of the time at its threshold at the start 0 switches right after it.

    A lag of 0 is a common default of a model: the shift of the comparison
    is at least its change over the resolution of the time, so the explicit
    root of AMICI is after the start 0 (a shift of the units in the last
    place of 0 is behind it for CVODES) and a comparison of the scaled time
    turns at most the resolution of the time late.
    """
    times = [0, end / 4, end] if output == "times" else None
    model = SWITCH_AT_THE_START.replace("CONDITION", condition)
    amici, roadrunner = both(
        sbml(model), Simulation(end=end, times=times), ["time", "A"]
    )
    np.testing.assert_allclose(amici["A"][-1], expected, rtol=1e-6)
    assert_agree(amici, roadrunner, times)


@pytest.mark.parametrize("times", [[0, 5, 20], None])
def test_a_piecewise_of_a_state_against_the_scaled_time(
    times: list[float] | None,
) -> None:
    """`C > 1e-12 * time` with `C' = 2e-12` from 0 turns right after the start.

    The shift of the comparison is its change over the resolution of the
    time, in the units of `C`, not the resolution of the time itself, which
    `C` would never reach.
    """
    model = "model scaled\n  C = 0\n  C' = 2e-12\n  A = 0\n"
    model += "  A' = piecewise(1, C > 1e-12 * time, 0)\nend"
    amici, roadrunner = both(
        sbml(model), Simulation(end=20, times=times), ["time", "A"]
    )
    np.testing.assert_allclose(amici["A"][-1], 20.0, rtol=1e-6)
    assert_agree(amici, roadrunner, times)


@pytest.mark.parametrize("times", [[0, 5, 10, 20], None])
@pytest.mark.parametrize(
    ("condition", "expected"),
    [("A - B > 0", 15.0), ("A - B <= 0", 5.0)],
    ids=["gt", "leq"],
)
def test_a_piecewise_of_a_difference_at_its_threshold_after_a_change(
    condition: str, expected: float, times: list[float] | None
) -> None:
    """A comparison of a difference of states at its threshold after a change switches as the states move.

    `B` is 100 and set to `A` at 5, so `A - B` is exactly 0 there and rises
    after it; the shift is sized by the terms `A` and `B`.
    """
    model = SWITCH_OF_A_DIFFERENCE.replace("CONDITION", condition)
    simulation = Simulation(end=20, times=times, changes=[Change(5, {"B": "A"})])
    amici, roadrunner = both(sbml(model), simulation, ["time", "A", "B", "X"])
    np.testing.assert_allclose(amici["X"][-1], expected, rtol=1e-6)
    assert_agree(amici, roadrunner, times, changes=[5.0])


#: two events at one time whose priorities are formulas of a parameter, the
#: higher one fires first
FORMULA_PRIORITIES = """
model formula_priorities
  X = 0
  X' = 1
  p = P
  n = 0
  E1: at (X >= 2), t0=false, fromTrigger=false, priority=p: n = 2 * n + 1
  E2: at (X >= 2), t0=false, fromTrigger=false, priority=2 * p: n = 2 * n + 10
end
"""


@pytest.mark.parametrize("times", [[0, 1, 2, 3], None])
@pytest.mark.parametrize(("p", "n"), [(1.0, 21.0), (-1.0, 12.0)])
def test_events_with_priorities_which_are_formulas(
    p: float, n: float, times: list[float] | None
) -> None:
    """The executor orders simultaneous events by their priorities, which AMICI does not import as formulas."""
    amici, roadrunner = both(
        sbml(FORMULA_PRIORITIES.replace("P", repr(p))),
        Simulation(end=3, times=times),
        ["time", "X", "n"],
    )
    assert amici["n"][-1] == roadrunner["n"][-1] == n
    assert_agree(amici, roadrunner, times, events=[2.0])


#: an event whose trigger is not persistent, cancelled by the event of a
#: higher priority at the same time, which sets its trigger back to false
NOT_PERSISTENT = """
model not_persistent
  X = 0
  X' = 1
  n = 0
  m = 0
  E: at (X >= 1), t0=false, persistent=PERSISTENT, priority=1: n = n + 1
  F: at (X >= 1), t0=false, priority=2: X = 0, m = m + 1
end
"""


@pytest.mark.parametrize("times", [[0, 0.5, 1, 1.5, 2, 2.5], None])
@pytest.mark.parametrize(("persistent", "n"), [("false", 0.0), ("true", 2.0)])
def test_an_event_whose_trigger_is_not_persistent_is_cancelled(
    persistent: str, n: float, times: list[float] | None
) -> None:
    """The executor cancels a pending event whose trigger became false, which AMICI does not import."""
    amici, roadrunner = both(
        sbml(NOT_PERSISTENT.replace("PERSISTENT", persistent)),
        Simulation(end=2.5, times=times),
        ["time", "X", "n", "m"],
    )
    assert amici["m"][-1] == roadrunner["m"][-1] == 2.0
    assert amici["n"][-1] == roadrunner["n"][-1] == n
    assert_agree(amici, roadrunner, times, events=[1.0, 2.0])
