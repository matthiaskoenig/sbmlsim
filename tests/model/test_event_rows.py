"""The rows of an event of the model get the time of the event.

roadrunner reports an event in the steps of the integrator as two rows a
little before and after its time; sbmlsim tells them by the markers of the
events, see `sbmlsim.model.event_rows`.
"""

import libsbml
import numpy as np
import pytest
import roadrunner

from sbmlsim.model.event_rows import (
    EVENT_COUNTER,
    EVENT_TRIGGER_PREFIX,
    at_event_times,
    event_pairs,
    with_event_markers,
)
from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from tests.simulator.models import sbml
from tests.simulator.test_executor import SAWTOOTH

#: an event with a delay whose values are the ones at its trigger, and one
#: whose values are the ones at its execution
DELAYS = """
model delays
  var X = 0
  var Y = 0
  k = 1
  X' = k
  E: at (X >= 1): X = 0
  D: at 0.3 after (time >= 0.5): Y = Y + 100
  L: at 0.2 after (time >= 0.25), fromTrigger=false: Y = Y + 1
end
"""


def test_every_event_counts_the_events() -> None:
    text, markers = with_event_markers(sbml(DELAYS))
    assert markers == (EVENT_COUNTER, f"{EVENT_TRIGGER_PREFIX}D")
    model: libsbml.Model = libsbml.readSBMLFromString(text).getModel()
    variables = {
        model.getEvent(k).getId(): [
            a.getVariable() for a in model.getEvent(k).getListOfEventAssignments()
        ]
        for k in range(model.getNumEvents())
    }
    # the event with the values of its trigger marks itself with its time
    assert variables == {
        "E": ["X", EVENT_COUNTER],
        "D": ["Y", f"{EVENT_TRIGGER_PREFIX}D"],
        "L": ["Y", EVENT_COUNTER],
    }
    assert model.getParameter(EVENT_COUNTER).getValue() == 0.0
    assert model.getParameter(f"{EVENT_TRIGGER_PREFIX}D").getValue() == -np.inf


def test_a_marker_must_not_be_an_entity_of_the_model() -> None:
    text, _ = with_event_markers(sbml(SAWTOOTH))
    with pytest.raises(ValueError, match=EVENT_COUNTER):
        with_event_markers(text)


def test_the_markers_are_no_parameters_of_the_model() -> None:
    model = RoadrunnerSBMLModel(source=sbml(DELAYS))
    assert model.event_markers == (EVENT_COUNTER, f"{EVENT_TRIGGER_PREFIX}D")
    assert not any(sid.startswith("sbmlsim__") for sid in model.selections or [])
    sids = RoadrunnerSBMLModel.parameter_df(model.r_loaded)["sid"].tolist()
    assert not any(sid.startswith("sbmlsim__") for sid in sids)


def test_a_model_without_events_has_no_markers() -> None:
    model = RoadrunnerSBMLModel(
        source=sbml("model decay\n  var A = 1\n  A' = -A\nend\n")
    )
    assert model.event_markers == ()


def _rows(text: str, end: float, *selections: str) -> np.ndarray:
    """Integrate a model with its markers with the steps of the integrator."""
    marked, markers = with_event_markers(sbml(text))
    r = roadrunner.RoadRunner(marked)
    r.getIntegrator().setValue("variable_step_size", True)
    r.timeCourseSelections = ["time", *selections, *markers]
    return np.asarray(r.simulate(0, end), dtype=float)


def test_the_two_rows_of_every_event_are_paired() -> None:
    rows = _rows(SAWTOOTH, 2.5, "X")
    pairs = event_pairs(rows, 1)
    assert len(pairs) == 2
    for k, time in zip(pairs, (1.0, 2.0), strict=True):
        # the state before and after the reset, a little before and after it
        assert rows[k, 1] == pytest.approx(1.0) and rows[k + 1, 1] == 0.0
        assert rows[k, 0] < time < rows[k + 1, 0]
    times = at_event_times(rows, 1)
    for k, time in zip(pairs, (1.0, 2.0), strict=True):
        assert times[k] == times[k + 1] == pytest.approx(time, abs=1e-12)
    assert np.all(np.diff(times) >= 0.0)
    # a copy
    assert rows[pairs[0], 0] != times[pairs[0]]


def test_the_rows_of_a_delayed_event_are_paired_and_its_trigger_is_one_row() -> None:
    """A delayed event is two rows at its time, its trigger changes nothing.

    roadrunner executes a delayed event in the steps of the integrator at its
    time (roadrunner #1382), `L` at 0.45 and `D` at 0.8, and the reset at 1,
    each as the state before it and the state after it (roadrunner #1446);
    its rows are paired at the time of the event. The triggers at 0.25 and
    0.5 change nothing and are one row each, the row at their time.
    """
    rows = _rows(DELAYS, 1.5, "X", "Y")
    times = at_event_times(rows, 2)
    pairs = event_pairs(rows, 2)
    np.testing.assert_array_equal(times[pairs], times[pairs + 1])
    np.testing.assert_allclose(np.unique(times[pairs]), [0.45, 0.8, 1.0], atol=1e-12)
    for trigger in (0.25, 0.5):
        at_trigger = np.flatnonzero(np.abs(rows[:, 0] - trigger) < 1e-9)
        assert at_trigger.size == 1, trigger
        assert at_trigger[0] not in pairs and at_trigger[0] - 1 not in pairs
    # every change of Y is the row after an event
    jumps = np.flatnonzero(np.diff(rows[:, 2]) != 0)
    assert jumps.size > 0 and set(jumps) <= set(pairs)


def test_other_rows_keep_their_times() -> None:
    rows = np.array(
        [[0.0, 0.0, 0.0], [1e-9, 1e-9, 0.0], [0.5, 0.5, 0.0], [0.5 + 1e-8, 0.6, 0.0]]
    )
    np.testing.assert_array_equal(at_event_times(rows, 1), rows[:, 0])
    # without markers there are no events
    assert event_pairs(rows, 0).size == 0


def test_the_first_row_is_no_row_of_an_event() -> None:
    """roadrunner applies an event at the start of an integration without rows."""
    rows = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 1.0], [1.0, 1.0, 1.0]])
    assert event_pairs(rows, 1).size == 0


def test_a_row_belongs_to_one_event() -> None:
    """Three rows whose markers change in a row are an event and a row."""
    rows = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0 - 1e-12, 1.0, 0.0],
            [1.0 + 1e-12, 0.0, 1.0],
            [1.0 + 3e-12, 0.0, 2.0],
            [2.0, 1.0, 2.0],
        ]
    )
    assert event_pairs(rows, 1).tolist() == [1]


def test_two_equal_rows_are_no_event() -> None:
    """Two rows of equal values are two steps, also within the resolution of the time.

    They were the trigger of an event with a delay until the fix of
    roadrunner #1446, which reports it as one row.
    """
    rows = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5 + 1e-12, 1.0, 0.0],
            [0.5 + 2e-12, 2.0, 1.0],
            [1.0, 2.0, 1.0],
        ]
    )
    # the marker changes at the event alone
    assert event_pairs(rows, 1).tolist() == [2]
    rows[3:, 1:] = [1.0, 0.0]
    assert event_pairs(rows, 1).size == 0
    np.testing.assert_array_equal(at_event_times(rows, 1), rows[:, 0])
