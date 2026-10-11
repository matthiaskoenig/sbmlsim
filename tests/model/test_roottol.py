"""The rows of an event of the model get the time of the event.

roadrunner reports an event in the steps of the integrator at `root - roottol`
and `root + roottol`, where `roottol` is fixed by the first integration of the
process, see `sbmlsim.model.roottol`.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import roadrunner

from sbmlsim.model.roottol import (
    EPSILON,
    PIN_SPAN,
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
    """In a fresh process the probe fixes `roottol` to 6.2e-31, measured as 0.

    After a first integration of roadrunner over `0..10000` the probe
    measures the `roottol` of that integration.
    """
    code = (
        "import sys, roadrunner\n"
        "from sbmlsim.model.roottol import root_tolerance\n"
        "from tests.simulator.models import sbml\n"
        "if sys.argv[1]:\n"
        '    r = roadrunner.RoadRunner(sbml("model m\\n  S = 1\\n  S\' = -S\\nend"))\n'
        "    r.getIntegrator().setValue('variable_step_size', True)\n"
        "    r.simulate(0, float(sys.argv[1]))\n"
        "print(repr(root_tolerance().value))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, first],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr
    value = float(result.stdout.strip().splitlines()[-1])
    assert value == pytest.approx(expected, rel=1e-9, abs=0.0)
    assert 100 * 32 * EPSILON * PIN_SPAN < 1e-30
