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
    pairs = event_pairs(rows, 1, 2.5e-9)
    assert len(pairs) == 2
    for k, time in zip(pairs, (1.0, 2.0), strict=True):
        # the state before and after the reset, a little before and after it
        assert rows[k, 1] == pytest.approx(1.0) and rows[k + 1, 1] == 0.0
        assert rows[k, 0] < time < rows[k + 1, 0]
    times = at_event_times(rows, 1, 2.5e-9)
    for k, time in zip(pairs, (1.0, 2.0), strict=True):
        assert times[k] == times[k + 1] == pytest.approx(time, abs=1e-12)
    assert np.all(np.diff(times) >= 0.0)
    # a copy
    assert rows[pairs[0], 0] != times[pairs[0]]


def test_the_rows_of_a_delayed_event_and_of_its_trigger_are_paired() -> None:
    """The trigger of an event with a delay changes nothing, its rows are paired.

    roadrunner executes a delayed event in the steps of the integrator at the
    end of the step of CVODE after its time, here at the trigger at 0.5 and at
    the reset at 1; its rows are paired wherever they are.
    """
    rows = _rows(DELAYS, 1.5, "X", "Y")
    times = at_event_times(rows, 2, 1.5e-9)
    pairs = event_pairs(rows, 2, 1.5e-9)
    np.testing.assert_array_equal(times[pairs], times[pairs + 1])
    np.testing.assert_allclose(np.unique(times[pairs]), [0.25, 0.5, 1.0], atol=1e-12)
    # every change of Y is the row after an event
    jumps = np.flatnonzero(np.diff(rows[:, 2]) != 0)
    assert jumps.size > 0 and set(jumps) <= set(pairs)


def test_other_rows_keep_their_times() -> None:
    rows = np.array(
        [[0.0, 0.0, 0.0], [1e-9, 1e-9, 0.0], [0.5, 0.5, 0.0], [0.5 + 1e-8, 0.6, 0.0]]
    )
    np.testing.assert_array_equal(at_event_times(rows, 1, 1e-9), rows[:, 0])
    # without markers there are no events
    assert event_pairs(rows, 0, 1e-9).size == 0


def test_the_first_row_is_no_row_of_an_event() -> None:
    """roadrunner applies an event at the start of an integration without rows."""
    rows = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 1.0], [1.0, 1.0, 1.0]])
    assert event_pairs(rows, 1, 1e-9).size == 0


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
    assert event_pairs(rows, 1, 1e-9).tolist() == [1]


def test_the_rows_of_an_event_win_over_equal_rows_before_it() -> None:
    """Two equal rows before the row before an event are no trigger of the event."""
    rows = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5 + 1e-12, 1.0, 0.0],
            [0.5 + 2e-12, 2.0, 1.0],
            [1.0, 2.0, 1.0],
        ]
    )
    assert event_pairs(rows, 1, 1e-9).tolist() == [2]
    # without the event the equal rows are the trigger of a delayed event
    rows[3:, 1:] = [1.0, 0.0]
    assert event_pairs(rows, 1, 1e-9).tolist() == [1]
