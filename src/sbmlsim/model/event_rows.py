"""The rows of the events of the model in the steps of the integrator.

libroadrunner reports an event of the model in the steps of the integrator
(`variable_step_size`) as two rows, the state before the event and the state
after it, at two times a little apart: `d` before and after the root of the
trigger, and for an event which roadrunner finds at the end of a step of
CVODE rather than as a root, the machine epsilon before the end of the step
and `d` after it. `d` is the root tolerance of CVODE at the event,
`100 * 32 * eps * (|t| + |h|)` of the time `t` of the event and the last step
`h` of CVODE, at most half the time since the row before
(`CVODEIntegrator::getRootTolerance` of libroadrunner 2.11; up to 2.10 the
first integration of the process fixed it). It depends on a step of CVODE
which roadrunner does not report, and the first step of CVODE after an event
can be shorter than it, so the times do not tell the rows of an event from
two steps.

sbmlsim marks the events instead: `with_event_markers` adds a parameter which
every event of the model counts, `EVENT_COUNTER`, which nothing of the model
reads, and the executor selects it, and the other markers, as the last
columns of its integrations, see `sbmlsim.simulator.executor`. Two adjacent
rows between which a marker changes are the rows of an event, see
`event_pairs`, and `at_event_times` gives both the time half way between
them, the root of the trigger, so an event is a jump at one time, as a
change is. An event with a delay whose values are the ones at the time of
its trigger would count from the count at that time and miss an event which
executes during its delay: it marks itself with the time of its trigger in a
parameter of its own instead, which differs between two of its executions.

The behavior of roadrunner which sbmlsim depends on is pinned by the tests a
new release of roadrunner has to pass: `tests/model/test_event_rows.py`,
`tests/simulator/test_event_rows.py` (the rows of events and their
independence of the history of the process and of the number of workers),
`tests/simulator/test_executor.py` (the events at the end of a segment and
of a simulation, the folded rows, the rounding of the output times by
roadrunner and the runs again past the end) and
`tests/fit/test_data_at_change.py` (the data at an event).
"""

from __future__ import annotations

import libsbml
import numpy as np

#: the machine epsilon, which roadrunner subtracts from the end of a step of
#: CVODE at which it finds an event
EPSILON = float(np.finfo(float).eps)

#: the prefix of the ids of the markers, which are no parameters of the model
MARKER_PREFIX = "sbmlsim__"

#: the parameter which counts the events of the model, see the module
EVENT_COUNTER = f"{MARKER_PREFIX}events"

#: the prefix of the parameter which marks an event with a delay whose values
#: are the ones at the time of its trigger with that time, see the module
EVENT_TRIGGER_PREFIX = f"{MARKER_PREFIX}trigger_"


def with_event_markers(sbml: str) -> tuple[str, tuple[str, ...]]:
    """Add the markers of the events to a model, see the module.

    Every event assigns `EVENT_COUNTER` its value plus one, except an event
    with a delay whose values are the ones at the time of its trigger, which
    assigns `<EVENT_TRIGGER_PREFIX><event>` the time of its trigger; a marker
    starts at `-inf`, which no time of a trigger is.

    Args:
        sbml: the SBML of a model with events.

    Returns:
        The SBML with the markers and their ids, the counter first.

    Raises:
        ValueError: if the model already has an entity of the id of a marker.
    """
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    model: libsbml.Model = doc.getModel()

    def marker(sid: str, value: float) -> str:
        if model.getElementBySId(sid) is not None:
            raise ValueError(
                f"The model already has an entity '{sid}', the id of a marker of "
                f"its events."
            )
        parameter: libsbml.Parameter = model.createParameter()
        parameter.setId(sid)
        parameter.setValue(value)
        parameter.setConstant(False)
        return sid

    markers: list[str] = []
    counter = marker(EVENT_COUNTER, 0.0)
    markers.append(counter)
    for k in range(model.getNumEvents()):
        event: libsbml.Event = model.getEvent(k)
        delayed = event.isSetDelay() and event.getDelay().getMath() is not None
        if delayed and event.getUseValuesFromTriggerTime():
            sid = marker(f"{EVENT_TRIGGER_PREFIX}{event.getId() or k}", -np.inf)
            markers.append(sid)
            math = libsbml.ASTNode(libsbml.AST_NAME_TIME)
            math.setName("time")
        else:
            sid = counter
            math = libsbml.parseL3Formula(f"{counter} + 1")
        assignment: libsbml.EventAssignment = event.createEventAssignment()
        assignment.setVariable(sid)
        assignment.setMath(math)
    return libsbml.writeSBMLToString(doc), tuple(markers)


def event_pairs(rows: np.ndarray, markers: int, tolerance: float) -> np.ndarray:
    """Get the rows of the events of an integration with the steps of the integrator.

    Two adjacent rows are an event if a marker changes between them, see the
    module. Two rows of the same values closer than the resolution of the
    time, neither of which is a row of such an event, are the trigger of an
    event with a delay, which changes nothing. The first row, the state at
    the start, is no row of an event: roadrunner applies an event at the
    start of an integration without rows. A row belongs to one event, of
    adjacent rows in a row the first, third, ... are events.

    Args:
        rows: the rows, the time first and the markers last.
        markers: the number of markers.
        tolerance: the resolution of the time.

    Returns:
        The index of the row before every event, the row after it is the next,
        sorted.
    """
    if rows.shape[0] < 3 or markers == 0:
        return np.empty(0, dtype=int)
    values = rows[:, 1:]
    changed = np.any(values[1:, -markers:] != values[:-1, -markers:], axis=1)
    changed[0] = False
    events = _alternate(np.flatnonzero(changed))
    rows_of_events = np.zeros(rows.shape[0], dtype=bool)
    rows_of_events[events] = True
    rows_of_events[events + 1] = True
    trigger = (
        np.all(values[1:] == values[:-1], axis=1)
        & (rows[1:, 0] - rows[:-1, 0] <= tolerance)
        & ~rows_of_events[:-1]
        & ~rows_of_events[1:]
    )
    trigger[0] = False
    return np.sort(np.concatenate([events, _alternate(np.flatnonzero(trigger))]))


def _alternate(candidates: np.ndarray) -> np.ndarray:
    """Keep the first, third, ... of every run of adjacent candidates."""
    if candidates.size == 0:
        return candidates
    index = np.arange(candidates.size)
    first = np.maximum.accumulate(
        np.where(np.diff(candidates, prepend=-2) > 1, index, 0)
    )
    return candidates[(index - first) % 2 == 0]


def at_event_times(rows: np.ndarray, markers: int, tolerance: float) -> np.ndarray:
    """Give the two rows of every event of the model the time of the event.

    Both rows of an event, see `event_pairs`, get the time half way between
    them, the root of the trigger, which roadrunner reports the rows the same
    time before and after; an event at the end of a step of CVODE is a few
    thousand machine epsilons of the time before that. roadrunner's last row,
    interpolated to the end, is no row of an event and is left out by the
    caller, see `sbmlsim.simulator.executor._at_event_times`.

    Args:
        rows: the rows, the time first and the markers last.
        markers: the number of markers.
        tolerance: the resolution of the time.

    Returns:
        The times, a copy.
    """
    t = np.array(rows[:, 0], dtype=float)
    pairs = event_pairs(rows, markers, tolerance)
    if pairs.size == 0:
        return t
    event = t[pairs] + (t[pairs + 1] - t[pairs]) / 2.0
    t[pairs] = event
    t[pairs + 1] = event
    return t
