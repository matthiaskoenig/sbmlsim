"""The comparison of the executor of an engine with the executor of roadrunner.

The tests of the executors of COPASI and of AMICI run a plan on the engine
and on roadrunner and compare them on the output times: all rows of an
output of times, and for the steps of the integrator, which differ between
the engines, the rows at the changes, at the events and at the end.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sbmlsim.model import RoadrunnerSBMLModel, SBMLModel
from sbmlsim.result import TimecourseResult
from sbmlsim.simulation import Simulation
from sbmlsim.simulator.executor import execute
from sbmlsim.simulator.plan import Plan, compile_simulation

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


#: an event which a change makes true, of an assignment rule which no rate
#: reads (the review of the executor)
STALE = """
model stale
  X = 0
  X' = 1
  Z := 2*X
  n = 0
  E: at (Z < 1), t0=T0: n = n + 1
end
"""

#: an event which fires at 2, whose trigger a change sets back
REARM = """
model rearm
  X = 0
  X' = 1
  n = 0
  E: at (X > 2), t0=false: n = n + 1
end
"""

#: an event of the time which fires in the presimulation and in the simulation
PRESIMULATED = """
model presim
  X = 1
  X' = -X
  n = 0
  E: at (time >= 2), t0=true: n = n + 1
end
"""


def plan_of(source: str | Path, simulation: Simulation) -> Plan:
    """Compile a simulation against a model."""
    model = RoadrunnerSBMLModel(source=source)
    return compile_simulation(simulation, model.symbols, model.uinfo)


def both(
    engine: Callable[..., SBMLModel],
    execute_engine: Callable[[Plan, Any, Sequence[str]], TimecourseResult],
    source: str | Path,
    simulation: Simulation,
    selections: Sequence[str],
) -> tuple[TimecourseResult, TimecourseResult]:
    """Run a simulation with an engine and with roadrunner.

    The simulation is compiled once and executed with both engines at the
    tolerances `SETTINGS`.

    Args:
        engine: the class of the model of the engine.
        execute_engine: the executor of the engine.
        source: the model.
        simulation: the simulation.
        selections: the columns of the results.

    Returns:
        The results of the engine and of roadrunner.
    """
    roadrunner_model = RoadrunnerSBMLModel(source=source, settings=SETTINGS)
    engine_model = engine(source=source, settings=SETTINGS)
    plan = compile_simulation(
        simulation, roadrunner_model.symbols, roadrunner_model.uinfo
    )
    result = execute_engine(plan, engine_model, selections)
    roadrunner = execute(plan, roadrunner_model, selections)
    return result, roadrunner


def assert_equal_results(
    engine: TimecourseResult, roadrunner: TimecourseResult
) -> None:
    """Compare two results of an output of times on all rows."""
    assert engine.columns == roadrunner.columns
    np.testing.assert_array_equal(engine.time, roadrunner.time)
    np.testing.assert_allclose(engine.values, roadrunner.values, rtol=1e-6, atol=1e-9)


def rows_at(result: TimecourseResult, time: float) -> np.ndarray:
    """Get the rows of a result at a time."""
    return result.values[result.time == time]


def pairs_at(
    result: TimecourseResult, times: Sequence[float], tolerance: float
) -> np.ndarray:
    """Get the two rows of a change or an event at times, and the last row.

    The pair of a time is its first row within `tolerance` and the row after
    it, the states before and after the change or the event.
    """
    rows = []
    for time in times:
        first = np.flatnonzero(np.abs(result.time - time) <= tolerance)[0]
        rows.append(result.values[first : first + 2])
    rows.append(result.values[-1:])
    return np.concatenate(rows)


def assert_equal_rows(
    engine: TimecourseResult,
    roadrunner: TimecourseResult,
    changes: Sequence[float] = (),
    events: Sequence[float] = (),
) -> None:
    """Compare the steps of the integrator at the changes, the events and the end.

    The rows of a change are at its time in both engines. roadrunner gives
    the rows of an event of the model times within the resolution of the
    time after the event, see `sbmlsim.model.event_rows`, whose states
    differ by the rates times that offset.
    """
    assert engine.columns == roadrunner.columns
    c, r = pairs_at(engine, changes, 0.0), pairs_at(roadrunner, changes, 0.0)
    np.testing.assert_array_equal(c[:, 0], r[:, 0])
    np.testing.assert_array_equal(c[:-1, 0], np.repeat(changes, 2))
    np.testing.assert_allclose(c[:, 1:], r[:, 1:], rtol=1e-6, atol=1e-9)
    c, r = pairs_at(engine, events, 1e-6), pairs_at(roadrunner, events, 1e-6)
    np.testing.assert_allclose(c[:, 0], r[:, 0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(c[:, 1:], r[:, 1:], rtol=1e-6, atol=1e-6)
