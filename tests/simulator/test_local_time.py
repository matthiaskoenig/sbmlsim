"""A model which does not read the time is integrated in local time."""

import ctypes
import sys
from dataclasses import replace

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.executor import execute
from sbmlsim.simulator.plan import compile_simulation
from tests.simulator.models import sbml

SEL = ["time", "[A]", "[B]", "X", "C", "k1"]

EVENT = """
model events
  compartment C = 1;
  species A in C; species B in C;
  A = 1; B = 0; k = 0.5
  J: A -> B; k*A
  E: at A < 0.2: A = 1
end
"""

DELAYED = """
model delayed
  compartment C = 1;
  species A in C;
  A = 1; n = 0; k = 0.1
  J: A -> ; k*A
  E: at 5 after (A < 0.5), t0=true, persistent=true: n = n + 1
end
"""

TIME_RULE = """
model clock
  compartment C = 1;
  species A in C;
  A = 1; k := 1 + time
  J: A -> ; 0*A
end
"""


def _run(model: RoadrunnerSBMLModel, sim: Simulation, local: bool, sel: list[str]):
    """Run a plan in local or in absolute time."""
    model.symbols = replace(model.symbols, time_dependent=not local)
    plan = compile_simulation(sim, model.symbols, model.uinfo)
    return execute(plan, model, sel)


@pytest.mark.parametrize(
    "sim",
    [
        Simulation(end=30, changes=[Change([10, 20], {"[A]": 2.0})], steps=30),
        Simulation(start=-20, end=10, changes=[Change(-10, {"X": 0.0})], steps=30),
        Simulation(end=30, changes=[Change(10, {"[A]": "[A] + 1"})]),
    ],
)
def test_local_time_equals_absolute_time(sim: Simulation) -> None:
    """The result of the local time is the one of the absolute time."""
    local = _run(RoadrunnerSBMLModel(source=sbml()), sim, True, SEL)
    absolute = _run(RoadrunnerSBMLModel(source=sbml()), sim, False, SEL)
    if sim.steps is not None:
        np.testing.assert_array_equal(local["time"], absolute["time"])
        np.testing.assert_allclose(local.values, absolute.values, rtol=1e-6, atol=1e-9)
    else:
        assert local["time"][0] == pytest.approx(absolute["time"][0])
        assert local["time"][-1] == pytest.approx(absolute["time"][-1])
        assert local["[A]"][-1] == pytest.approx(absolute["[A]"][-1], rel=1e-6)


def test_the_output_times_are_the_absolute_times() -> None:
    """Exact output times come back as they were asked for."""
    times = [0.0, 9.999, 10.0, 10.5, 25.0]
    sim = Simulation(end=25, changes=[Change(10, {"[A]": 2.0})], times=times)
    res = _run(RoadrunnerSBMLModel(source=sbml()), sim, True, SEL)
    assert list(res["time"]) == times
    assert res["[A]"][2] == pytest.approx(2.0)


def test_a_change_formula_reads_the_absolute_time() -> None:
    """`time` in the formula of a change is the time of the change."""
    sim = Simulation(end=20, changes=[Change(10, {"[A]": "time"})], times=[10, 20])
    res = _run(RoadrunnerSBMLModel(source=sbml()), sim, True, SEL)
    assert res["[A]"][0] == pytest.approx(10.0)


def test_an_event_fires_in_local_time() -> None:
    """An event which does not read the time fires as in absolute time."""
    sim = Simulation(end=40, changes=[Change(20, {"[A]": 0.1})], steps=400)
    sel = ["time", "[A]", "[B]"]
    local = _run(RoadrunnerSBMLModel(source=sbml(EVENT)), sim, True, sel)
    absolute = _run(RoadrunnerSBMLModel(source=sbml(EVENT)), sim, False, sel)
    np.testing.assert_allclose(local["[A]"], absolute["[A]"], rtol=1e-5, atol=1e-8)
    # the change to 0.1 made the trigger true: A is reset to 1 at the change
    assert local["[A]"][200] == pytest.approx(1.0)


def test_a_pending_delayed_event_fires_at_its_time() -> None:
    """An event which is pending at a change fires its delay after the trigger."""
    # A < 0.5 at t = ln(2) / 0.1 = 6.93, the event is pending at the change at 10
    sim = Simulation(end=40, changes=[Change(10, {"k": 0.1})], steps=40)
    res = Simulator(n_workers=1).run(RoadrunnerSBMLModel(source=sbml(DELAYED)), sim)
    fired = np.asarray(res["n"], dtype=float) > 0
    assert np.asarray(res["time"], dtype=float)[np.argmax(fired)] == pytest.approx(12.0)


def test_a_model_which_reads_the_time_keeps_the_absolute_time() -> None:
    """`k := 1 + time` is the absolute time after a change."""
    model = RoadrunnerSBMLModel(source=sbml(TIME_RULE))
    assert model.symbols.time_dependent
    plan = compile_simulation(
        Simulation(end=30, changes=[Change(10, {"[A]": 2.0})], times=[0, 15, 30]),
        model.symbols,
        model.uinfo,
    )
    res = execute(plan, model, ["time", "k"])
    np.testing.assert_allclose(res["k"], [1.0, 16.0, 31.0])


def _late_dose_output(capfd, local: bool) -> str:
    """Get the output of C of a dose into an empty state at a late time.

    In absolute time the first step after the dose at 9600 hr is below the
    resolution of the time ("t + h = t"). CVODE writes the warning with the
    buffered streams of C, which are flushed first.
    """
    from examples.hctz_fitting import MODEL_PATH

    sim = Simulation(
        time_unit="hr",
        end=10000,
        steps=100,
        changes=[Change(9600, {"PODOSE_hctz": Q(25, "mg")})],
    )
    model = RoadrunnerSBMLModel(source=MODEL_PATH)
    assert not model.symbols.time_dependent
    plan = compile_simulation(sim, model.symbols, model.uinfo)
    capfd.readouterr()
    if local:
        Simulator(n_workers=1).run(model, sim)
    else:
        model.symbols = replace(model.symbols, time_dependent=True)
        execute(plan, model, ["time"])
    ctypes.CDLL(None).fflush(None)
    captured = capfd.readouterr()
    return captured.out + captured.err


@pytest.mark.skipif(
    sys.platform == "win32", reason="the streams of C are flushed with POSIX ctypes"
)
def test_no_warning_after_a_late_dose(capfd) -> None:
    """A late dose warns in absolute time, not in the local time of the simulator."""
    assert "t + h = t" in _late_dose_output(capfd, local=False)
    assert "t + h = t" not in _late_dose_output(capfd, local=True)
