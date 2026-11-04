"""The segments of a plan on an engine which continues one integration from another.

The executors of COPASI and of AMICI integrate the segments of a plan
between its changes in absolute time, every one from the state the one
before it ended in, and share how the segments are cut, which changes and
events of the model are applied at their bounds and which rows the output
has, see `simulate_segments`; an engine integrates through its `Course`.
The executor of roadrunner has its own segments, see
`sbmlsim.simulator.executor`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from sbmlsim.model.model_sbml import SBMLModel
from sbmlsim.simulator.changes import ModelValues, apply_changes, fire_events, triggers
from sbmlsim.simulator.executor import integrable, segment_stop
from sbmlsim.simulator.plan import OutputMode, Plan, SteadyStatePlan, time_tolerance


class Course(Protocol):
    """The integrations of a model of an engine which continue one another."""

    def restart(self, time: float) -> np.ndarray | None:
        """Prepare an integration from the current state at a time, after its changes.

        Returns:
            The row of the state at the time before the events which the
            engine fires right after it, see
            `sbmlsim.simulator.executor_amici`, `None` without such an event:
            the output has it as the row at the time, the integration starts
            from the state after them.
        """
        ...

    def steps(self, start: float, end: float) -> np.ndarray:
        """Integrate from `start` to `end` with the output of the steps of the integrator.

        Returns:
            The values, a row per step, the first at the start and the last
            at the end, an event of the model as two rows at its time.
        """
        ...

    def times(self, start: float, times: np.ndarray, end: float) -> np.ndarray:
        """Integrate from `start` with output at times, the last one the stop.

        Args:
            start: the time of the current state.
            times: the output times, sorted, distinct, after the start.
            end: the end of the segment, which the integration must not
                reach if the last time is before it.

        Returns:
            The values, a row at the start and a row per time, at the times.
        """
        ...

    def event_at_the_end(
        self, start: float, end: float, tolerance: float
    ) -> np.ndarray | None:
        """Get the state after the events of the model at the end of a simulation.

        An event within the resolution of the time after the end is at the
        end, as a time of a grid sees it, see
        `sbmlsim.result.timecourse.grid_weights`.

        Args:
            start: the start of the integration, just before the end, where
                the last segment stopped, or the end after a change at the
                end.
            end: the end of the simulation.
            tolerance: the resolution of the time.

        Returns:
            The state after the last event within the resolution after the
            end, `None` without one.
        """
        ...

    def steady_state(self, plan: SteadyStatePlan, start: float) -> float:
        """Integrate the model from a time until its rates of change vanish.

        Returns:
            The time the steady state was reached at.
        """
        ...


def simulate_segments(
    plan: Plan,
    model: SBMLModel,
    values: ModelValues,
    columns: list[str],
    course: Course,
) -> np.ndarray:
    """Integrate the segments of a plan.

    The interval is split at the times of the changes. At the start of a
    segment the changes of its time are applied, every value evaluated with
    the state at that time before any target is set, see
    `sbmlsim.simulator.changes.apply_changes`; after a change after the start
    (or at the start after a presimulation) the events whose triggers the
    change made true fire, which an engine does not see, see
    `sbmlsim.simulator.changes.fire_events`. A change at the start without a
    presimulation fires no event, as in roadrunner, whose triggers are the
    ones of the initial state. A change after the start, the one at the end
    included, has two rows at its time, the state before the change, which
    ends the segment before it, and the state after it, which starts the
    next one. A segment of a model with events stops just before its end, see
    `sbmlsim.simulator.executor.segment_stop`, so an event of the model at
    the time of a change fires after the change; a segment too short for the
    integrator is not integrated, its state is the one at its start, see
    `sbmlsim.simulator.executor.integrable`. An event of the model within the
    resolution of the time after the end is at the end, see
    `Course.event_at_the_end`: the steps of the integrator have it as two
    rows, an output time at the end the state after it. The steady state
    after the end starts after it.

    Args:
        plan: the plan.
        model: the model, initialized, after the presimulation.
        values: the values of the model.
        columns: the selections, the time first.
        course: the integrations of the model.

    Returns:
        The values, a row per output time and a column per selection, two
        rows at the time of a change after the start which is an output
        time.
    """
    events = {event.time: event for event in plan.events}
    points = sorted({plan.start, plan.end, *events})
    model_events = plan.symbols.events
    times = np.asarray(plan.times, dtype=float)
    tolerance = time_tolerance(plan.start, plan.end)

    blocks: list[np.ndarray] = []
    # whether the last segment is not integrated, and the time the model is at
    # after it: where its integration stopped, the start of a segment which is
    # not integrated
    held, stopped = False, plan.start
    for k in range(len(points) - 1):
        a, b = points[k], points[k + 1]
        # the segment ends at a change, whose state before it is its last row
        before_change = b in events
        if a in events:
            if model_events and (a > plan.start or plan.steady_state is not None):
                # the triggers at the time the model is at before the change,
                # the end of the steady state for a change at the start
                before = model.value("time") if a == plan.start else stopped
                triggered = triggers(model_events, values, before, model)
                apply_changes(events[a], values, plan)
                fire_events(model_events, triggered, values, a, model)
            else:
                # the triggers at the start are the ones of the initial state
                apply_changes(events[a], values, plan)
        head = course.restart(a) if model_events else None

        # an event of the model at the time of the next change fires after the
        # change, an event at the end has the state before it as a row
        end_stop = segment_stop(a, b) if model_events else b
        held = not integrable(a, end_stop)
        if held:
            if plan.output is OutputMode.INTEGRATOR:
                block = held_rows(model, columns, np.array([a, b]))
            else:
                wanted = times[
                    (times >= a) & ((times < b) if before_change else (times <= b))
                ]
                if before_change and np.any(times == b):
                    # the state before the change at `b`
                    wanted = np.append(wanted, b)
                block = held_rows(model, columns, wanted)
        elif plan.output is OutputMode.INTEGRATOR:
            block = course.steps(a, end_stop)
            # the state at the end of the simulation, or the state before the
            # change at `b`, whose state after it starts the next segment
            block[-1, 0] = b
        else:
            wanted = times[
                (times >= a) & ((times < b) if before_change else (times <= b))
            ]
            # an output time between the stop and the end is the stop
            capped = np.minimum(wanted, end_stop)
            grid = np.unique(np.concatenate([capped, [end_stop]]))
            grid = grid[grid > a]
            result = course.times(a, grid, b)
            if head is not None:
                result[0] = head
            block = result[np.isin(np.concatenate([[a], grid]), capped)]
            if block.shape[0] != wanted.size:
                raise RuntimeError(
                    f"The output times of the segment [{a}, {b}] are not distinct "
                    f"before its stop."
                )
            block[:, 0] = wanted
            if before_change and np.any(times == b):
                # the state before the change at `b`, the next segment starts
                # with the state after it
                state = result[-1].copy()
                state[0] = b
                block = np.vstack([block, state])
        if head is not None and plan.output is OutputMode.INTEGRATOR:
            # the state before the events right after the start of the segment
            block = np.vstack([head, block])
        stopped = a if held else model.value("time")
        blocks.append(block)

    end = plan.end
    changed = end in events
    at_end = bool(blocks) and blocks[-1].shape[0] > 0 and blocks[-1][-1, 0] == end
    after = False
    if changed or (held and model_events):
        # a change at the end is applied after the integration, the output at
        # the end has the state before it and the state after it and the
        # events it triggers; the events of the time in a last segment which
        # is not integrated fire at its end, as after a change
        fired = 0
        if model_events:
            triggered = triggers(model_events, values, stopped, model)
            if changed:
                apply_changes(events[end], values, plan)
            fired = fire_events(model_events, triggered, values, end, model)
        elif changed:
            apply_changes(events[end], values, plan)
        if at_end and (changed or fired):
            state = state_row(model, columns, end)
            if changed or plan.output is OutputMode.INTEGRATOR:
                blocks[-1] = np.vstack([blocks[-1], state])
            else:
                blocks[-1][-1, :] = state
            after = True

    if model_events and at_end:
        # the events at the end, from the stop before it, or from the end
        # after a change there or a last segment which is not integrated
        state = course.event_at_the_end(
            end if changed or held else stopped, end, tolerance
        )
        if state is not None:
            state[0] = end
            if after or plan.output is OutputMode.TIMES:
                blocks[-1][-1, :] = state
            else:
                blocks[-1] = np.vstack([blocks[-1], state])

    if plan.steady_state_output is not None:
        # the steady state after the end, PEtab's measurement at `inf`, from
        # the time of the model after the events at the end
        course.steady_state(plan.steady_state_output, model.value("time"))
        blocks.append(state_row(model, columns, np.inf)[np.newaxis, :])

    if not blocks:
        return np.empty((0, len(columns)))
    return np.concatenate(blocks, axis=0)


def held_rows(
    model: SBMLModel, columns: Sequence[str], times: np.ndarray
) -> np.ndarray:
    """Get the rows of a segment which is not integrated, the state at its start."""
    state = state_row(model, columns, 0.0)
    rows = np.tile(state, (times.size, 1))
    rows[:, 0] = times
    return rows


def state_row(model: SBMLModel, columns: Sequence[str], time: float) -> np.ndarray:
    """Get the current values of the selections, `time` the given time."""
    return np.array([time if c == "time" else model.value(c) for c in columns])
