"""The rows of an event of the model get the time of the event.

roadrunner reports an event in the steps of the integrator at `root - roottol`
and `root + roottol`, where `roottol` is fixed by the first integration of the
process, see `sbmlsim.model.roottol`.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import roadrunner

from sbmlsim.model.roottol import (
    EPSILON,
    PINNED,
    RootTolerance,
    at_event_times,
    root_tolerance,
)
from tests.simulator.models import sbml
from tests.simulator.test_executor import SAWTOOTH

#: `roottol` after a first integration over `0..10000`
LARGE = RootTolerance(value=100 * 32 * EPSILON * 1e4, precision=2e-19)


def test_the_two_rows_of_a_root_get_the_time_of_the_event() -> None:
    root, value = 1.0 + 3.1e-13, LARGE.value
    time = np.array([0.0, 0.5, root - value, root + value, 1.5])
    snapped = at_event_times(time, LARGE)
    assert snapped[2] == snapped[3]
    np.testing.assert_allclose(snapped, [0.0, 0.5, root, root, 1.5], rtol=0, atol=1e-15)
    # a copy
    assert time[2] == root - value


def test_the_two_rows_of_an_event_at_the_end_of_a_step_get_its_time() -> None:
    """roadrunner reports the state before it at the end of the step less epsilon."""
    for end in (0.75, 1.0, 3.0):
        time = np.array([0.0, end - EPSILON, end + LARGE.value, end + 1.0])
        snapped = at_event_times(time, LARGE)
        assert snapped[1] == snapped[2]
        np.testing.assert_allclose(snapped[1], end, rtol=0, atol=1e-15)


def test_a_step_after_an_event_and_before_its_row_is_sorted() -> None:
    """roadrunner reports the second step of CVODE after an event even if it is
    before the row of the state after the event."""
    root, value = 2.0, LARGE.value
    time = np.array([0.0, root - value, root + value, root + value / 3, 3.0])
    snapped = at_event_times(time, LARGE)
    assert np.all(np.diff(snapped) >= 0.0)
    np.testing.assert_allclose(snapped[1:3], root, rtol=0, atol=1e-15)


def test_other_rows_keep_their_times() -> None:
    value = LARGE.value
    time = np.array([0.0, 1e-9, 1e-9 + 3 * value, 0.5, 0.5 + 1.5 * value, 1.0])
    np.testing.assert_array_equal(at_event_times(time, LARGE), time)


def test_a_row_belongs_to_one_event() -> None:
    """Three rows in a row are one event and a row; the rows are taken from the first."""
    value = LARGE.value
    time = np.array([1.0 - value, 1.0 + value, 1.0 + 3 * value, 2.0])
    snapped = at_event_times(time, LARGE)
    np.testing.assert_allclose(snapped[:3], [1.0, 1.0, 1.0 + 3 * value], atol=1e-15)


def test_rows_of_one_time_stay_with_a_root_tolerance_of_zero() -> None:
    zero = RootTolerance(value=0.0, precision=2e-19)
    time = np.array([0.0, 1.0, 1.0, 1.5, 1.5 + 1e-9])
    np.testing.assert_array_equal(at_event_times(time, zero), time)
    assert at_event_times(np.array([1.0]), zero).tolist() == [1.0]


def test_the_probe_measures_the_root_tolerance_of_the_process() -> None:
    """The half gap of the rows of an event of any model is the measured one."""
    tolerance = root_tolerance()
    r = roadrunner.RoadRunner(sbml(SAWTOOTH))
    r.timeCourseSelections = ["time", "X"]
    r.getIntegrator().setValue("variable_step_size", True)
    rows = np.array(r.simulate(0, 2.5))
    after = int(np.flatnonzero((rows[:, 0] > 0.5) & (rows[:, 1] == 0.0))[0])
    gap = rows[after, 0] - rows[after - 1, 0]
    assert abs(gap / 2 - tolerance.value) <= 4 * (np.spacing(1.0) + tolerance.precision)


@pytest.mark.parametrize(
    ("first", "expected"),
    [("", 0.0), ("10000", 100 * 32 * EPSILON * 1e4)],
    ids=["probe first", "after a first integration over 0..10000"],
)
def test_the_probe_fixes_the_root_tolerance_if_it_integrates_first(
    first: str, expected: float
) -> None:
    """In a fresh process the probe fixes `roottol` to `PINNED`, measured as 0.

    After a first integration of roadrunner over `0..10000` the probe
    measures the `roottol` of that integration and warns once, with the
    remedy.
    """
    code = (
        "import logging, sys, roadrunner\n"
        "from sbmlsim.model.roottol import root_tolerance\n"
        "from tests.simulator.models import sbml\n"
        "logging.basicConfig()\n"
        "if sys.argv[1]:\n"
        '    r = roadrunner.RoadRunner(sbml("model m\\n  S = 1\\n  S\' = -S\\nend"))\n'
        "    r.getIntegrator().setValue('variable_step_size', True)\n"
        "    r.simulate(0, float(sys.argv[1]))\n"
        "print(repr(root_tolerance().value))\n"
        "root_tolerance()\n"
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", code, first],
        capture_output=True,
        text=True,
        check=False,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    value = float(result.stdout.strip().splitlines()[-1])
    assert value == pytest.approx(expected, rel=1e-9, abs=0.0)
    assert value <= PINNED if not first else value > PINNED
    warnings = result.stderr.count("roadrunner integrated in this process before")
    assert warnings == (1 if first else 0), result.stderr
    if first:
        assert "load a model before integrating with roadrunner" in result.stderr


#: the gaps of the rows of the event of the probe of a roadrunner without the
#: `static` root tolerance of libroadrunner 2.10: no rows of the event, both
#: rows at the time of the event, or a root tolerance per integration
DEGRADED = {
    "no rows": (None, None),
    "rows at the event": (0.0, 0.0),
    "per integration": (2 * 3200 * EPSILON * 2.0**10, 2 * 3200 * EPSILON * 2.0**5),
}


@pytest.mark.parametrize("gaps", [*DEGRADED, "failure"])
def test_the_probe_degrades_on_another_roadrunner(
    gaps: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A probe which cannot measure or pin `roottol` gives 0, without a warning.

    The rows of an event stay at the times roadrunner reports them, here at
    the time of the event, which the probe of the process pinned.
    """
    from sbmlsim.model import roottol
    from sbmlsim.simulation import Simulation
    from sbmlsim.simulator import Simulator

    def probe() -> tuple[float | None, float | None]:
        if gaps == "failure":
            raise RuntimeError("roadrunner cannot simulate the probe")
        return DEGRADED[gaps]

    monkeypatch.setattr(roottol, "_probe", probe)
    root_tolerance.cache_clear()
    try:
        with caplog.at_level("DEBUG", logger="sbmlsim.model.roottol"):
            tolerance = root_tolerance()
        assert tolerance.value == 0.0
        assert not [r for r in caplog.records if r.levelname == "WARNING"]
        if gaps != "rows at the event":
            assert "left at the times roadrunner reports them" in caplog.text
        simulator = Simulator(n_workers=1)
        result = simulator.simulate(simulator.load(sbml(SAWTOOTH)), Simulation(end=2.5))
        time, x = result.time, np.asarray(result["X"])
        after = int(np.flatnonzero((time > 0.5) & (x == 0.0))[0])
        assert time[after - 1] == time[after] == pytest.approx(1.0, abs=1e-12)
        assert x[after - 1] == pytest.approx(1.0, abs=1e-9)
        # an event at the end has its two rows, an output time at the end the
        # state after it
        dose = simulator.load(sbml(DOSE_AT_THE_END))
        steps = simulator.simulate(dose, Simulation(end=2))
        assert steps.time[-2:].tolist() == [2.0, 2.0]
        np.testing.assert_allclose(np.asarray(steps["A"])[-2:], [0.0, 10.0])
        times = simulator.simulate(dose, Simulation(end=2, times=[0, 1, 2]))
        assert times["A"][-1] == pytest.approx(10.0)
    finally:
        root_tolerance.cache_clear()


#: a dose by a trigger of the time at the end 2 of a simulation
DOSE_AT_THE_END = """
model dose
  var A = 0
  A' = -0.1 * A
  E: at (time >= 2.0): A = A + 10
end
"""


@pytest.mark.parametrize("apart", [0, 1, 4])
def test_the_rows_of_an_event_at_the_end_are_one_time_within_their_rounding(
    apart: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two rows of an event at the end are paired within the rounding of a time.

    roadrunner reports them a few units in the last place apart where the
    root tolerance is not known, `at_event_times` leaves them so; a step of
    CVODE is at least a hundred times the machine epsilon times the time.
    """
    from sbmlsim.simulator import executor

    event = 2.0 + 1e-13
    after = event
    for _ in range(apart):
        after = float(np.nextafter(after, np.inf))
    block = np.array(
        [[2.0 - 4 * EPSILON, 0.0], [event, 0.0], [after, 10.0], [2.0 + 2e-9, 10.0]]
    )
    monkeypatch.setattr(executor, "_steps", lambda *args: (block.copy(), True))
    monkeypatch.setattr(
        executor, "root_tolerance", lambda: RootTolerance(value=0.0, precision=0.0)
    )
    state = executor._event_at_the_end(
        None,  # ty: ignore[invalid-argument-type]
        2.0 - 4 * EPSILON,
        2.0,
        2e-9,
    )
    assert state is not None
    assert state.tolist() == [after, 10.0]
    # a step of CVODE after the stop is no event
    block = np.array(
        [
            [2.0 - 4 * EPSILON, 0.0],
            [2.0 + 100 * EPSILON * 2.0, 1e-13],
            [2.0 + 2e-9, 2e-9],
        ]
    )
    state = executor._event_at_the_end(
        None,  # ty: ignore[invalid-argument-type]
        2.0 - 4 * EPSILON,
        2.0,
        2e-9,
    )
    assert state is None
