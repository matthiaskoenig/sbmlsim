"""The executor runs a plan on roadrunner.

The semantics are the ones of PEtab v2:

1. The model is initialized with the pre-initialization values, see
   `RoadrunnerSBMLModel.initialize`: the initial assignments follow a changed
   parameter, and nothing of an earlier simulation is left.
2. A presimulation integrates until the rates of change vanish.
3. The interval is split at the times of the events. At the start of a
   segment the events of its time are applied: every value is evaluated
   first, with the state at that time, then every target is set; a
   compartment keeps the concentration of the concentration species in it,
   see `sbmlsim.simulator.changes.apply_changes`.
4. After a change at a time after the start, the events whose triggers the
   change made true fire, which roadrunner does not see, see
   `sbmlsim.simulator.changes.fire_events`.
5. A segment is integrated with the output of the plan and ends in its
   state at its end, which the next change is applied to; for the steps of
   the integrator see `_steps`, which has an event of the model as two rows
   at the time of the event, the states before and after it, which the
   markers of the events tell, see `sbmlsim.model.event_rows`. A
   change after the start, the one at the end included, has two rows at its
   time in the output, the state before the change, which ends the segment
   before it, and the state after it, which starts the next one: the steps
   of the integrator at every change, an output of times at its times which
   are changes. A segment of a model with events stops just before its end,
   a unit in the last place of the larger magnitude of its start and end
   and at least four machine epsilons, see `segment_stop` and
   `_simulate_times`, so an event of the model at the time of a change fires
   after the change and an event at the end after the state before it. A
   segment too short for CVODE, a few units in the last place in absolute
   time, is not integrated: its state is the one at its start, and the
   triggers before the next change are evaluated at the time the model is
   at, its start, so an event of the time in it fires once, after it, see
   `integrable`. A delayed event which roadrunner keeps pending closer after
   the start of an integration than CVODE integrates holds the state at the
   start likewise, see `_integrate`.
6. An event of the model within the resolution of the time after the end is
   at the end, see `_event_at_the_end`: the steps of the integrator have it
   as two rows, the states before and after it, an output time at the end
   the state after it, as a time of a grid at an event, and it fires after
   a change at the end. The steady state after the end starts after it. An
   event within the resolution before the end is in the chain at the end,
   whose rows are the state after it, as at a change. An event whose
   trigger becomes true within the rounding of the state one unit in the
   last place before the end, e.g. `X >= 1` of `X' = 1` from 0 at the end 1,
   fires before it, and the end has the state after it.
7. A model which does not read the time (`ModelSymbols.time_dependent`) is
   integrated in local time: every segment starts at the time 0 of
   roadrunner and its output is shifted back, so the first step of CVODE
   after a change is never below the resolution of the time. A model which
   reads the time is integrated in absolute time.

The executor uses no pint, no deepcopy, no xarray and no pandas: it is what
an evaluation of the objective of a fit runs.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager, nullcontext

import numpy as np
import roadrunner

from sbmlsim.model.event_rows import EPSILON, at_event_times, event_pairs
from sbmlsim.model.model_roadrunner import (
    RoadrunnerSBMLModel,
    apply_pending_step_event,
)
from sbmlsim.model.symbols import ModelSymbols
from sbmlsim.result.timecourse import TimecourseResult
from sbmlsim.simulator.changes import apply_changes, fire_events, triggers
from sbmlsim.simulator.plan import (
    OutputMode,
    Plan,
    SteadyStatePlan,
    preinit_targets,
    time_tolerance,
)

logger = logging.getLogger(__name__)

#: the setting of the integrator which switches between the output of its
#: steps and exact output times
VARIABLE_STEP_SIZE = "variable_step_size"

#: the least time between the stop of an integration before the end of a
#: segment and the end: roadrunner applies an event which CVODE finds less
#: than twice the machine epsilon after the start of a call of the integrator
#: without its rows (`CVODEIntegrator::integrate` of libroadrunner 2.11), and
#: an event at the end is found from the stop, see `_event_at_the_end`
STOP = 4.0 * EPSILON

#: the error of CVODE for an output time closer to the time of the integrator
#: than it integrates, see `integrable`
TOO_CLOSE = "CV_TOO_CLOSE"


class SteadyStateError(RuntimeError):
    """A presimulation did not reach a steady state."""


class _EventAtTheHorizon(Exception):
    """roadrunner fired an event of the model before its time at the end of a horizon.

    CVODE locates the root of a trigger in its steps within its root
    tolerance, also a little before the time the trigger becomes true, e.g.
    `time > 10` at `10 - 7.5e-12`, and roadrunner fires the event there. If
    that is within the resolution of the time before the end of a horizon of
    a steady state, see `steady_state`, roadrunner starts the next horizon
    with the triggers evaluated at its start, where such a trigger is still
    false, and fires the event a second time when it becomes true. `execute`
    runs the plan again, which integrates the horizon with exact output at
    its end, where CVODE does not fire the event before its time: it fires
    once, in the next horizon, as with COPASI and AMICI. A plan runs again
    at most once per horizon.

    Attributes:
        exact: the horizons of the steady state with exact output at their
            end, which the horizon is added to.
        horizon: the index of the horizon, from 0.
    """

    def __init__(self, exact: set[int], horizon: int) -> None:
        super().__init__(f"An event fired at the end of the horizon {horizon}.")
        self.exact = exact
        self.horizon = horizon


class _PastTheEnd(Exception):
    """roadrunner integrated a segment with the steps of the integrator past its end.

    An event of the model in the last step of roadrunner after the end of the
    segment fired or is pending, see `_steps`, and the state after the
    segment is wrong. `_execute` runs the plan again, which integrates the
    segment with exact output at the times of its steps, see `_exact_times`.
    If a later segment of that run is past its end as well, the plan runs a
    third time with both segments with exact output, and keeps the state at
    the start of every segment of the steps after them in memory
    (`_saved_state`): a later segment past its end starts again from its
    state at the start with exact output at the times of its steps. A plan
    runs at most three times however many of its segments roadrunner
    integrates past their end, besides the runs of `_EventAtTheHorizon`, and
    only a third run keeps states, one save per later segment and one load
    per later segment past its end.

    Attributes:
        segment: the index of the segment.
        output: its output times in the time of roadrunner, the times it is
            integrated to and the times of its rows, see `_exact_times`.
    """

    def __init__(self, segment: int, output: tuple[np.ndarray, np.ndarray]) -> None:
        super().__init__(f"The segment {segment} was integrated past its end.")
        self.segment = segment
        self.output = output


def execute(
    plan: Plan, model: RoadrunnerSBMLModel, selections: Sequence[str]
) -> TimecourseResult:
    """Run a plan on a loaded model.

    The output has two rows at the time of a change after the start, the
    state before the change and the state after it, and the steps of the
    integrator have two at an event of the model at the end, see the module.

    Args:
        plan: the plan, compiled against the model.
        model: the loaded model.
        selections: the columns of the result, `time` is added as the first
            column if it is missing.

    Returns:
        The result, with the time shifted by `plan.time_shift`.

    Raises:
        SteadyStateError: if the presimulation does not reach a steady state.
        RuntimeError: if roadrunner fails to integrate.
    """
    columns = output_columns(selections)
    r = model.r_loaded
    local = not model.symbols.time_dependent
    if (
        not local
        and plan.start != 0.0
        and r.model.getNumEvents() > 0
        and not model.warned_start
    ):
        # roadrunner evaluates the triggers of the events when the model is
        # initialized, at the time 0, and keeps their state across a reset
        logger.warning(
            "The model '%s' has events and is simulated from the time %s: "
            "roadrunner evaluates the triggers of the events at the time 0, an "
            "event whose trigger depends on the time may fire at the start.",
            model.sid or r.model.getModelName(),
            plan.start,
        )
        model.warned_start = True
    # the markers of the events of the model are the last columns of the
    # integrations, see `sbmlsim.model.event_rows`
    selected = [*columns, *model.event_markers]
    variable_step_size = bool(r.getIntegrator().getValue(VARIABLE_STEP_SIZE))
    # the output times of the segments of the steps of the integrator with
    # exact output, see `_PastTheEnd`
    exact: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    # the horizons of the steady states before the simulation and after its
    # end with exact output at their end, see `_EventAtTheHorizon`
    presimulation: set[int] = set()
    after_the_end: set[int] = set()
    run = 1
    # the time the simulation integrates to, see `discard_pending_events`
    horizon = (
        np.inf
        if plan.steady_state is not None or plan.steady_state_output is not None
        else plan.end + time_tolerance(plan.start, plan.end)
    )
    while True:
        model.discard_pending_events(horizon)
        model.initialize(preinit_targets(plan))
        if list(r.timeCourseSelections) != selected:
            r.timeCourseSelections = selected
        try:
            if plan.steady_state is not None:
                steady_state(model, plan.steady_state, exact=presimulation)
            values = _simulate(plan, model, r, selected, local, exact, after_the_end)
        except _EventAtTheHorizon as err:
            err.exact.add(err.horizon)
            run += 1
            logger.debug(
                "roadrunner fired an event of the model '%s' before the end of the "
                "horizon %s of a steady state, the plan runs again with exact "
                "output at its end (run %s).",
                model.sid or r.model.getModelName(),
                err.horizon,
                run,
            )
            continue
        except _PastTheEnd as err:
            # a segment with exact output is never past its end
            exact[err.segment] = err.output
            run += 1
            logger.debug(
                "roadrunner integrated the segment %s of the model '%s' past its "
                "end, the plan runs again with exact output in it (run %s).",
                err.segment,
                model.sid or r.model.getModelName(),
                run,
            )
            continue
        finally:
            if model.event_markers:
                # see `apply_pending_step_event`
                apply_pending_step_event(r)
            r.getIntegrator().setValue(VARIABLE_STEP_SIZE, variable_step_size)
            model.simulated()
        break

    values = output_time(values[:, : len(columns)], columns, plan, model.symbols)
    return TimecourseResult(columns=tuple(columns), values=values)


def output_columns(selections: Sequence[str]) -> list[str]:
    """Get the columns of the output of a plan, the time of the simulation first.

    A model may have an entity named `time` as well, which a later column
    `time` is (case 01820); else a later column `time` is the time of the
    simulation again, see `output_time`. Every executor shares it.

    Args:
        selections: the selections of the output.

    Returns:
        The columns, `time` added as the first one if it is not.
    """
    columns = list(selections)
    if not columns or columns[0] != "time":
        columns = ["time", *columns]
    return columns


def output_time(
    values: np.ndarray, columns: Sequence[str], plan: Plan, symbols: ModelSymbols
) -> np.ndarray:
    """Give the columns of the time of an output the time of the simulation.

    The first column is shifted by `plan.time_shift`. A later column `time`
    is a copy of it, the time of the row: an engine reads it in its own time,
    which is the local time of a segment or misses the shift, and the state
    after a change is no row of the engine. A later column `time` of a model
    with an entity named `time` is the value of that entity. Every executor
    shares it.

    Args:
        values: the output, a column per entry of `columns`; changed in place.
        columns: the columns, see `output_columns`.
        plan: the plan.
        symbols: the symbols of the model.

    Returns:
        The output.
    """
    if plan.time_shift != 0.0:
        values[:, 0] += plan.time_shift
    entity = "time" in symbols.parameters | symbols.compartments | symbols.species
    if not entity:
        for k, column in enumerate(columns[1:], start=1):
            if column == "time":
                values[:, k] = values[:, 0]
    return values


def _simulate(
    plan: Plan,
    model: RoadrunnerSBMLModel,
    r: roadrunner.RoadRunner,
    columns: list[str],
    local: bool,
    exact: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    after_the_end: set[int] | None = None,
) -> np.ndarray:
    """Integrate the segments of a plan, see the module.

    Args:
        plan: the plan.
        model: the loaded model.
        r: its roadrunner instance.
        columns: the selections, the markers of the events of the model last.
        local: integrate every segment from the time 0 of roadrunner (local
            time) instead of its absolute time.
        exact: the output times in the time of roadrunner of the segments of
            the steps of the integrator with exact output, by the index of
            the segment, see `_PastTheEnd`.
        after_the_end: the horizons of the steady state after the end with
            exact output at their end, see `_EventAtTheHorizon`.

    Returns:
        The values, a row per output time and a column per selection, two
        rows at the time of a change after the start which is an output
        time.

    Raises:
        _PastTheEnd: if roadrunner integrated a segment of the steps of the
            integrator past its end before any segment with exact output.
        _EventAtTheHorizon: if roadrunner fired an event of the model before
            the end of a horizon of the steady state after the end.
    """
    events = {event.time: event for event in plan.events}
    points = sorted({plan.start, plan.end, *events})
    model_events = plan.symbols.events
    markers = len(model.event_markers)
    times = np.asarray(plan.times, dtype=float)
    tolerance = time_tolerance(plan.start, plan.end)
    exact = exact or {}

    blocks: list[np.ndarray] = []
    # whether the last segment is not integrated, and the time of roadrunner
    # the model is at after it: where its integration ended, the start of a
    # segment which is not integrated, the start before the first segment
    held, stopped = False, plan.start
    for k in range(len(points) - 1):
        a, b = points[k], points[k + 1]
        # the segment ends at a change, whose state before it is its last row:
        # every segment but the last, since the changes are within the
        # simulation, and the last one at a change at the end
        before_change = b in events
        if a in events:
            if model_events and (a > plan.start or plan.steady_state is not None):
                # the triggers at the time the model is at before the change:
                # where the segment before it stopped, also a segment which is
                # not integrated, so an event of the time fires once, and the
                # end of the steady state for the change at the start
                before = float(r.model.getTime()) if a == plan.start else stopped
                triggered = triggers(model_events, r, a if local else before, model)
                apply_changes(events[a], r, plan)
                fire_events(model_events, triggered, r, a, model)
            else:
                # at the start after a reset roadrunner evaluates the triggers
                # itself
                apply_changes(events[a], r, plan)

        # the segment in the time of roadrunner, from 0 in local time
        offset = a if local else 0.0
        start, end = a - offset, b - offset
        # an event of the model at the time of the next change fires after
        # the change (PEtab v2, reinitialization), and an event at the end has
        # the state before it as a row: the integration stops just before the
        # end of the segment, where roadrunner does not fire it, see
        # `fire_events` and `_event_at_the_end`
        end_stop = segment_stop(start, end) if model_events else end
        # a segment too short for CVODE keeps the state at its start
        held = not integrable(start, end_stop)
        if held:
            if plan.output is OutputMode.INTEGRATOR:
                block = _held(r, np.array([a, b]))
            else:
                wanted = times[
                    (times >= a) & ((times < b) if before_change else (times <= b))
                ]
                if before_change and np.any(times == b):
                    # the state before the change at `b`
                    wanted = np.append(wanted, b)
                block = _held(r, wanted)
        elif plan.output is OutputMode.INTEGRATOR:
            output = exact.get(k)
            if output is None:
                # after the second segment past its end the state at the start
                # of every later segment is kept, see `_PastTheEnd`
                keep = len(exact) > 1 and k > max(exact)
                with _saved_state(r) if keep else nullcontext(None) as load:
                    block, at_end = _steps(r, start, end_stop, tolerance, markers)
                    if not at_end:
                        # the steps are sorted once their events have their
                        # times; a row of roadrunner which `at_event_times`
                        # cannot pair can be before an earlier step
                        output = _exact_times(
                            start, np.sort(block[1:-1, 0]), end_stop, tolerance
                        )
                        if load is None:
                            raise _PastTheEnd(k, output)
                        load()
            if output is not None:
                times_k, labels = output
                block = _simulate_times(r, times_k, end, bool(model_events))
                block[:, 0] = labels
            block[:, 0] += offset
            # the end of the segment, not its end in the time of roadrunner
            # shifted back: the state at the end of the simulation, or the
            # state before the change at `b`, whose state after it starts the
            # next segment
            block[-1, 0] = b
        else:
            wanted = times[
                (times >= a) & ((times < b) if before_change else (times <= b))
            ]
            # a time which rounds to the end in local time is before the change
            shifted = np.minimum(wanted - offset, end_stop)
            grid = np.unique(np.concatenate([[start], shifted, [end_stop]]))
            result = _simulate_times(r, grid, end, bool(model_events))
            block = result[np.isin(grid, shifted)]
            if block.shape[0] != wanted.size:
                raise RuntimeError(
                    f"The output times of the segment [{a}, {b}] are not distinct "
                    f"in local time."
                )
            # the output times are the ones asked for, not the shifted ones
            block[:, 0] = wanted
            if before_change and np.any(times == b):
                # the state before the change at `b`, the next segment starts
                # with the state after it
                state = result[-1].copy()
                state[0] = b
                block = np.vstack([block, state])
        # the time the model is at: roadrunner may integrate an exact output
        # time to one unit in the last place after it and fire an event there,
        # see `_simulate_times`; a segment which is not integrated is at its start
        stopped = start if held else float(r.model.getTime())
        blocks.append(block)

    # the last segment in the time of roadrunner
    offset = points[-2] if local else 0.0
    end = plan.end - offset
    changed = plan.end in events
    at_end = bool(blocks) and blocks[-1].shape[0] > 0 and blocks[-1][-1, 0] == plan.end
    after = False
    if changed or (held and model_events):
        # a change at the end is applied after the integration, the output at
        # the end has the state before it and the state after it and the
        # events it triggers; the events of the time in a last segment which
        # is not integrated fire at its end, as after a change
        fired = 0
        if model_events and plan.end > plan.start:
            triggered = triggers(model_events, r, end if local else stopped, model)
            if changed:
                apply_changes(events[plan.end], r, plan)
            fired = fire_events(model_events, triggered, r, plan.end, model)
        elif changed:
            apply_changes(events[plan.end], r, plan)
        if at_end and (changed or fired):
            state = _state(r, columns, plan.end)
            if plan.end > plan.start and (
                changed or plan.output is OutputMode.INTEGRATOR
            ):
                blocks[-1] = np.vstack([blocks[-1], state])
            else:
                blocks[-1][-1, :] = state
            after = True

    if model_events and at_end:
        # the events at the end, from the stop before it, or from the end after
        # a change there or a last segment which is not integrated
        state = _event_at_the_end(
            r, end if changed or held else stopped, end, tolerance, markers
        )
        if state is not None:
            state[0] = plan.end
            if after or plan.output is OutputMode.TIMES:
                # the state after the change and the events at the end, the
                # state after the events at an output time at the end
                blocks[-1][-1, :] = state
            else:
                blocks[-1] = np.vstack([blocks[-1], state])

    if plan.steady_state_output is not None:
        # the steady state after the end, PEtab's measurement at `inf`, from
        # the time of the model, after the events at the end: an event of the
        # time it fired does not fire again
        start = 0.0 if local else float(r.model.getTime())
        steady_state(
            model,
            plan.steady_state_output,
            start=start,
            exact=set() if after_the_end is None else after_the_end,
        )
        row = _state(r, columns, np.inf)
        blocks.append(row[np.newaxis, :])

    if not blocks:
        return np.empty((0, len(columns)))
    return np.concatenate(blocks, axis=0)


def _steps(
    r: roadrunner.RoadRunner, start: float, end: float, tolerance: float, markers: int
) -> tuple[np.ndarray, bool]:
    """Integrate a segment with the steps of the integrator.

    roadrunner integrates with the steps of CVODE up to the step after the
    end and interpolates its last row to the end with the dense output of
    CVODE, the model is left in the state at the end (libroadrunner 2.11).
    roadrunner also fires an event of the model it finds in that step after
    the end, or keeps it pending and fires it in its next integration, also
    of the next simulation: `_to_the_end` fires such an event, and the
    segment is integrated past its end and gets exact output, see
    `_PastTheEnd`.

    roadrunner reports an event of the model inside the segment as two rows,
    the states before and after it, a little before and after its time; both
    rows get the time of the event, see `sbmlsim.model.event_rows`, also for
    an event just before or after the end, see `_events_at_the_end`.

    Args:
        r: the roadrunner instance, in the state at the start.
        start: the start of the segment in the time of roadrunner.
        end: its end in the time of roadrunner.
        tolerance: the resolution of the time: a step of a model with events
            which is closer to the end is an event which roadrunner fired at
            the end.
        markers: the number of markers of the events of the model, the last
            columns, see `sbmlsim.model.event_rows`; 0 without events.

    Returns:
        The values in the time of roadrunner, a row per step, the first at
        the start and the last at the end, and whether the segment ended at
        its end; if not, the last row is the one of roadrunner.
    """
    events = markers > 0
    block = _variable_steps(r, start, end)
    if not events and block[-1, 0] == end:
        return block, True
    if events:
        block, inside = _events_at_the_end(r, block, end, tolerance, markers)
        if not inside:
            return block, False
    if not _to_the_end(r, end, events):
        return block, False
    steps = block[1:-1, 0]
    if events and steps.size and steps[-1] > end - tolerance:
        return block, False
    state = np.array(r.getSelectedValues(), dtype=float)
    state[0] = end
    if block[-1, 0] < end - tolerance:
        # roadrunner stopped at the most rows of its output, the integrator
        # went on to the end
        logger.warning(
            "The steps of the integrator stopped at the time %s before the end %s "
            "of a segment: roadrunner keeps at most `max_output_rows` rows of them, "
            "the state at the end follows the last one.",
            block[-1, 0],
            end,
        )
        return np.vstack([block, state]), True
    block[-1] = state
    return block, True


def segment_stop(start: float, end: float) -> float:
    """Get the time just before the end of a segment where its integration stops.

    It is one unit in the last place of the times of the segment before its
    end, of the larger of the magnitudes of its start and its end in the
    time of roadrunner, which CVODE resolves: its rounding of the time is the
    machine epsilon times that magnitude. It is at least `STOP`, so the stop
    before an end at 0 is no subnormal number, at which CVODE fails, e.g. for
    a trigger of the time at 0, and roadrunner reports an event at the end
    with its rows when it integrates from the stop.

    Args:
        start: the start of the segment in the time of roadrunner.
        end: its end in the time of roadrunner.

    Returns:
        The stop, a normal number before the end.
    """
    return end - max(float(np.spacing(max(abs(start), abs(end)))), STOP)


def integrable(start: float, stop: float) -> bool:
    """Whether CVODE integrates from the start of a segment to its stop.

    CVODE rejects an output time closer to the start than twice its rounding
    of the time, the machine epsilon times the larger magnitude
    (`CV_TOO_CLOSE`), and roadrunner a stop at the start: such a segment, of
    a few units in the last place in absolute time, e.g. between two changes
    a few units in the last place apart, is not integrated. Its state is the
    one at its start, which differs from the integrated one by the rate times
    its duration, and the events of the time in it fire after it, see
    `_simulate`. LSODA of COPASI has the same condition (`TOUT too close to
    T`), see `sbmlsim.simulator.executor_copasi`.

    Args:
        start: the start of the segment in the time of roadrunner.
        stop: where its integration stops, see `segment_stop`.

    Returns:
        Whether the segment is integrated, the condition of CVODE.
    """
    distance = stop - start
    return distance > 0.0 and distance >= 2.0 * (EPSILON * max(abs(start), abs(stop)))


def _held(r: roadrunner.RoadRunner, times: np.ndarray) -> np.ndarray:
    """Get the rows of a segment which is not integrated, see `integrable`.

    Args:
        r: the roadrunner instance, in the state at the start of the segment.
        times: the times of the rows.

    Returns:
        The values, a row per time, each the state at the start.
    """
    state = np.asarray(r.getSelectedValues(), dtype=float)
    values = np.tile(state, (times.size, 1))
    values[:, 0] = times
    return values


def _event_at_the_end(
    r: roadrunner.RoadRunner, start: float, end: float, tolerance: float, markers: int
) -> np.ndarray | None:
    """Get the state after the events of the model at the end of a simulation.

    CVODE locates an event a little after its time, an event at the end of a
    simulation after the end, where an integration up to the end does not
    find it: a state trigger at the end, and a trigger of the time at the end
    in the steps of the integrator. An event within the resolution of the
    time after the end is at the end, as a time of a grid sees it, see
    `sbmlsim.result.timecourse.grid_weights`. The model is integrated with
    the steps of the integrator from `start` to the resolution after the end,
    which gives the two rows of every event the time of the event, see
    `_steps`.

    Args:
        r: the roadrunner instance in the state at `start`.
        start: the start of the integration in the time of roadrunner, one
            unit in the last place before the end, where the last segment
            stopped, or the end after a change at the end.
        end: the end of the simulation in the time of roadrunner.
        tolerance: the resolution of the time.
        markers: the number of markers of the events, see `_steps`.

    Returns:
        The state after the last event within the resolution after the end,
        at the time of the event, `None` without one.
    """
    block, _ = _steps(r, start, end + tolerance, tolerance, markers)
    # roadrunner's last row is no row of an event
    rows = block[:-1]
    pairs = event_pairs(rows, markers, tolerance)
    pairs = pairs[rows[pairs + 1, 0] <= end + tolerance]
    if pairs.size == 0:
        return None
    return block[pairs[-1] + 1].copy()


def _events_at_the_end(
    r: roadrunner.RoadRunner,
    block: np.ndarray,
    end: float,
    tolerance: float,
    markers: int,
) -> tuple[np.ndarray, bool]:
    """Give the events of the steps their times, also the ones near the end.

    The two rows of an event inside the steps get its time, see
    `at_event_times`. roadrunner reports the state after an event a little
    after its time, see `sbmlsim.model.event_rows`; for an event less than
    that before the end, or after it, that row is after the end and
    roadrunner folds it into its last row, and the row before the event is
    left alone, see `_folded_event`. The model is then in the state after it:

    - an event before the end: the row before it gets its time, and the steps
      continue from the event, whose first row is the state after it, so a
      further event before the end gets its rows as well; an event within
      the resolution of the end gets the state of the model as its row after
      it, and the segment is past its end as it is when roadrunner reports
      both rows, see `_steps`;
    - an event at or after the end: roadrunner applied it before the change
      at the end, the segment is past its end, and the row before the event,
      at roadrunner's offset rather than at a step, is left out of its steps.

    Args:
        r: the roadrunner instance after the steps of the integrator.
        block: the steps.
        end: the end in the time of roadrunner.
        tolerance: the resolution of the time.
        markers: the number of markers of the events, see `_steps`.

    Returns:
        The steps with the times of their events, and whether no event after
        the end was applied.
    """
    _at_event_times(block, end, tolerance, markers)
    done: list[np.ndarray] = []
    last = -np.inf
    while (event := _folded_event(r, block, end, tolerance, markers)) is not None:
        if event >= end:
            return np.concatenate([*done, np.delete(block, -2, axis=0)]), False
        if event <= last:
            break
        last = event
        block[-2, 0] = event
        if end - event <= tolerance:
            after = np.array(r.getSelectedValues(), dtype=float)
            after[0] = event
            block = np.insert(block, block.shape[0] - 1, after, axis=0)
            break
        done.append(block[:-1])
        block = _variable_steps(r, event, end)
        _at_event_times(block, end, tolerance, markers)
    if done:
        block = np.concatenate([*done, block])
    return block, True


def _at_event_times(
    block: np.ndarray, end: float, tolerance: float, markers: int
) -> None:
    """Give the two rows of every event inside the steps the time of the event.

    The last row, which roadrunner interpolated to the end, is no row of an
    event: it is left out, also where roadrunner folded the row after an
    event into it, see `_folded_event`. A last row before the end is a step,
    roadrunner stopped at the most rows of its output.

    Args:
        block: the steps, their times are changed.
        end: the end in the time of roadrunner.
        tolerance: the resolution of the time.
        markers: the number of markers of the events, see `_steps`.
    """
    rows = block.shape[0] - 1 if block[-1, 0] >= end - tolerance else block.shape[0]
    block[:rows, 0] = at_event_times(block[:rows], markers, tolerance)


def _folded_event(
    r: roadrunner.RoadRunner,
    block: np.ndarray,
    end: float,
    tolerance: float,
    markers: int,
) -> float | None:
    """Get the time of an event whose row after it roadrunner folded into the end.

    The last row is the one roadrunner interpolated to the end and the row
    before the last is the state before an event which roadrunner applied
    after it: the markers of the model differ from the ones of that row, see
    `sbmlsim.model.event_rows`. The model is at the time of the event in the
    state after it, or at the end if CVODE interpolates the end from the
    event, a few units in the last place from it.

    Args:
        r: the roadrunner instance after the steps of the integrator, before
            `_to_the_end`.
        block: the steps, their events inside at their times.
        end: the end in the time of roadrunner.
        tolerance: the resolution of the time: the last row is the one
            roadrunner interpolated to the end if it is closer to the end.
        markers: the number of markers of the events, see `_steps`.

    Returns:
        The time of the event, before, at or after the end, or `None`.
    """
    if block.shape[0] < 3 or block[-1, 0] < end - tolerance:
        return None
    time = float(r.model.getTime())
    if time < float(block[-2, 0]):
        return None
    state = np.asarray(r.getSelectedValues(), dtype=float)
    if np.array_equal(state[-markers:], block[-2, -markers:]):
        return None
    return time


def _variable_steps(r: roadrunner.RoadRunner, start: float, end: float) -> np.ndarray:
    """Integrate with the steps of the integrator from `start` to `end`.

    roadrunner warns that the end was not reached if the last time is more
    than `end / 10000` before it, which is every negative end; the end is
    checked by `_steps`, so the warning of a negative end is not shown.
    """
    r.getIntegrator().setValue(VARIABLE_STEP_SIZE, True)
    if end >= 0.0:
        return _integrate(r, [start, end], steps=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Simulation requested end time point",
            category=UserWarning,
        )
        return _integrate(r, [start, end], steps=True)


def _integrate(r: roadrunner.RoadRunner, times: list[float], steps: bool) -> np.ndarray:
    """Integrate from the first time, past an event pending too close after it.

    roadrunner integrates to the time of an event of the model it keeps
    pending, a delayed event, before the next output time. CVODE rejects that
    time if it is closer to the start than it integrates (`CV_TOO_CLOSE`, the
    condition of `integrable`), e.g. an event at the end pending at the stop
    before it, see `_event_at_the_end`, or a unit in the last place after a
    change. As for a segment which is not integrated, the state is the one at
    the start up to the first time CVODE integrates to from it, from which
    roadrunner integrates again and fires the event in its first step, late
    as every delayed event of roadrunner (#1382). An event pending too close
    before a later output time is not handled: roadrunner fails after it
    integrated to the event.

    Args:
        r: the roadrunner instance in the state at the start.
        times: the start and the end for the steps of the integrator, else the
            output times, the start first.
        steps: whether the output is the steps of the integrator.

    Returns:
        The values, the first row at the start; the steps of the integrator
        past such an event have a row at the first time integrated from, the
        state at the start, before the steps.
    """

    def run(times: list[float]) -> np.ndarray:
        if steps:
            return np.array(r.simulate(times[0], times[-1]), dtype=float)
        return np.array(r.simulate(times=times), dtype=float)

    start = times[0]
    try:
        return run(times)
    except RuntimeError as err:
        # the integration failed before it left the start
        if (
            TOO_CLOSE not in str(err)
            or len(times) < 2
            or float(r.model.getTime()) != start
            or not integrable(start, times[1])
        ):
            raise
    # the model is in the state at the start, the event pending
    first = _held(r, np.array([start]))
    held = _first_integrable(start)
    logger.debug(
        "An event of the model is pending closer after the time %s than CVODE "
        "integrates, the state is held to the time %s.",
        start,
        held,
    )
    values = run([held, *times[1:]])
    if steps:
        return np.vstack([first, values])
    return np.vstack([first, values[1:]])


def _first_integrable(start: float) -> float:
    """Get the first time after `start` which CVODE integrates to from it.

    Args:
        start: the start of an integration in the time of roadrunner.

    Returns:
        The least time `t` of `integrable(start, t)`, a few units in the last
        place after the start.
    """
    stop = float(np.nextafter(start, np.inf))
    while not integrable(start, stop):
        stop = float(np.nextafter(stop, np.inf))
    return stop


def _to_the_end(r: roadrunner.RoadRunner, end: float, events: bool) -> bool:
    """Set the model to its state at the end of an integration with its steps.

    roadrunner leaves the model at the end of an integration with its steps
    (libroadrunner 2.11), unless it stopped at the most rows of its output.
    Else, and for a model with events, the integrator is called with exact
    output at the end, which CVODE interpolates from its last step without
    integrating: an event of the model which roadrunner keeps pending fires
    first, after the end.

    Args:
        r: the roadrunner instance after an integration with the steps of
            the integrator to `end`.
        end: the end in the time of roadrunner.
        events: whether the model has events.

    Returns:
        Whether the model is in the state at the end; not if a pending event
        fired, or if roadrunner fired an event after the end and the
        integrator cannot go back.

    Raises:
        RuntimeError: if the integrator fails for a model without events.
    """
    if not events and float(r.model.getTime()) == end:
        return True
    integrator: roadrunner.Integrator = r.getIntegrator()
    integrator.setValue(VARIABLE_STEP_SIZE, False)
    try:
        return float(integrator.integrate(end, 0.0)) == end
    except RuntimeError as err:
        if not events:
            raise
        logger.debug(
            "The integrator does not reach the end %s after an event of the model "
            "after it: %s",
            end,
            err,
        )
        return False


def _exact_times(
    start: float, steps: np.ndarray, end: float, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Get the output times of a segment with exact output, see `_PastTheEnd`.

    The times are the start, the steps before the end and the end. A step
    closer to the end than the resolution of the time is an event which
    roadrunner fired at the end and is left out. Steps closer to each other
    than the resolution are an event of the model, the states before and
    after it, which `_steps` gives at the time of the event, with the first
    steps of CVODE after it: the segment is integrated to a resolution
    before the first and after the last of them, where CVODE has not found
    the event yet and has found it, and the rows have the times of the first
    and the last, as the steps of the integrator have them, so data at the
    event sees the state after it. Two events less than two resolutions
    apart share the time between them, half way: both rows are the state
    between the events. An event closer to the start than the resolution has
    no row before it, the start is the state before it.

    Args:
        start: the start of the segment in the time of roadrunner.
        steps: the steps of the integrator inside the segment, sorted.
        end: its end in the time of roadrunner.
        tolerance: the resolution of the time.

    Returns:
        The times the segment is integrated to, sorted, a time twice between
        two events, and the times of the rows, sorted.
    """
    steps = np.asarray(steps, dtype=float)
    steps = steps[steps <= end - tolerance]
    times, labels = [start], [start]
    k = 0
    while k < steps.size:
        last = k
        while last + 1 < steps.size and steps[last + 1] - steps[last] < tolerance:
            last += 1
        if last == k:
            times.append(steps[k])
            labels.append(steps[k])
        else:
            # at most half way to the steps next to the event, which are at
            # least a resolution away
            before = steps[k] - tolerance
            if k > 0:
                before = max(before, (steps[k - 1] + steps[k]) / 2)
            after = steps[last] + tolerance
            if last + 1 < steps.size:
                after = min(after, (steps[last] + steps[last + 1]) / 2)
            for time, label in ((before, steps[k]), (after, steps[last])):
                if start < time < end:
                    times.append(time)
                    labels.append(label)
        k = last + 1
    times.append(end)
    labels.append(end)
    return np.array(times), np.array(labels)


def _simulate_times(
    r: roadrunner.RoadRunner, grid: np.ndarray, end: float, events: bool
) -> np.ndarray:
    """Integrate a segment with exact output at the times of a grid.

    roadrunner integrates every output time from the one before it, to
    `t + (next - t)`, which can round up by one unit in the last place: for a
    last time one unit before the end, before a change at the end, and a time
    before it below half of it, roadrunner integrated to the end and an event
    of the model at the time of the change fired before the change. Such a
    last interval gets a time between, from which the sum is exact; its row
    is not returned.

    Args:
        r: the roadrunner instance, in the state at the start.
        grid: the output times in the time of roadrunner, sorted, the first
            the start of the segment.
        end: the end of the segment in the time of roadrunner, which the
            integration must not reach if the last time is before it.
        events: whether the model has events.

    Returns:
        The values, a row per time of the grid.
    """
    r.getIntegrator().setValue(VARIABLE_STEP_SIZE, False)
    times = [float(t) for t in grid]
    between: float | None = None
    if events and len(times) > 1:
        between = intermediate_time(times[-2], times[-1], end)
        if between is not None:
            times.insert(len(times) - 1, between)
    values = _integrate(r, times, steps=False)
    if between is not None:
        values = np.delete(values, values.shape[0] - 2, axis=0)
    return values


def intermediate_time(previous: float, last: float, end: float) -> float | None:
    """Get a time between two output times from which the last one is exact.

    An integrator integrates an output time from the one before it, to
    `previous + (last - previous)`, which can round up by one unit in the
    last place: a last time before the end of a segment, its stop, can round
    to the end, where an event of the model at the time of a change fires
    before the change. From a time between them of the same sign within a
    factor of two of the last time the difference is exact, and so is the
    sum. Both executors share it, see `_simulate_times` and
    `sbmlsim.simulator.executor_copasi`.

    Args:
        previous: the output time before the last one.
        last: the last output time.
        end: the end of the segment, which the integration must not reach.

    Returns:
        The time between, `None` if the last time is not before the end or
        the sum does not reach the end.

    Raises:
        RuntimeError: if no time between keeps the integration before the end.
    """
    if not last < end or previous + (last - previous) < end:
        return None
    between = last / 2.0 if last > 0.0 else 2.0 * last
    if not previous < between < last or between + (last - between) >= end:
        raise RuntimeError(
            f"The integration from {previous} to {last} reaches the end {end} of "
            f"the segment."
        )
    return between


@contextmanager
def _saved_state(r: roadrunner.RoadRunner) -> Generator[Callable[[], None]]:
    """Keep the state of a roadrunner instance in memory for a block.

    Up to libroadrunner 2.10 `RoadRunner.saveStateS` kept a copy of every
    state it saved for the life of the process; the state of a block is
    released with it.

    Args:
        r: the roadrunner instance.

    Yields:
        A function which loads the state into the instance.
    """
    state = r.saveStateS()
    yield lambda: r.loadStateS(state)


def _state(r: roadrunner.RoadRunner, columns: list[str], time: float) -> np.ndarray:
    """Get the current values of the selections, `time` first, see `output_time`."""
    return np.array([time, *(r.getValue(c) for c in columns[1:])])


def steady_state(
    model: RoadrunnerSBMLModel,
    plan: SteadyStatePlan,
    start: float = 0.0,
    exact: set[int] | None = None,
) -> float:
    """Integrate a model until its rates of change vanish.

    The model is integrated with the steps of the integrator over horizons
    which grow by a factor of ten, from `start`, until
    `|dx/dt| <= absolute_tolerance + relative_tolerance * |x|` for every
    state, events active. Every horizon ends in the state at its end, see
    `_to_the_end`; after an event which roadrunner kept pending and which
    fires after the end, the next horizon starts at the time of the model
    after it, so its trigger is not false again at the start of the next
    horizon and the event fires once. A horizon at whose end roadrunner
    fired an event before its time is integrated with exact output at its
    end, see `_EventAtTheHorizon`.

    Args:
        model: the model in the state the integration starts from, the
            markers of its events the last selections, see `execute`.
        plan: the steady state.
        start: the time the integration starts at.
        exact: the horizons with exact output at their end, by their index.

    Returns:
        The time the steady state was reached at.

    Raises:
        SteadyStateError: if the steady state is not reached by `max_time`.
        _EventAtTheHorizon: if roadrunner fired an event of the model before
            its time at the end of a horizon which is not in `exact`.
    """
    r = model.r_loaded
    events = bool(model.symbols.events)
    markers = len(model.event_markers)
    exact = set() if exact is None else exact
    states = model.state_ids()
    time = start
    horizon = 1.0
    index = 0
    while True:
        end = start + min(horizon, plan.max_time)
        if index in exact:
            _simulate_times(r, np.array([time, end]), end, events)
        else:
            # the rows of the steps are only read at the end: roadrunner stops
            # at its maximum number of rows before the end of a horizon with
            # many events and warns, `_to_the_end` integrates the rest of it
            # (case 01627)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Simulation requested end time point",
                    category=UserWarning,
                )
                block = _variable_steps(r, time, end)
            if markers and _event_at_the_horizon(model, block, end, markers):
                raise _EventAtTheHorizon(exact, index)
        # the state at the end of the horizon, an event of the model which
        # roadrunner keeps pending fires in the presimulation; the next
        # horizon starts at the time of the model after it, from the end its
        # trigger would be false again and the event fire a second time
        time = (
            end if _to_the_end(r, end, events) else max(end, float(r.model.getTime()))
        )
        # the rates of every state of the integrator, the targets of rate
        # rules and the species; roadrunner's `getRatesOfChangeNamedArray`
        # has the values of every state but no name of a species reference
        # of a rate rule, so the states are read by their ids (#307)
        rates = np.abs(np.asarray(r.getRatesOfChange(), dtype=float).ravel())
        state = np.abs(np.array([r.getValue(sid) for sid in states], dtype=float))
        if np.all(rates <= plan.absolute_tolerance + plan.relative_tolerance * state):
            return time
        if time - start >= plan.max_time:
            worst = float(np.max(rates)) if rates.size else 0.0
            raise SteadyStateError(
                f"The model '{model.sid or r.model.getModelName()}' did not reach a "
                f"steady state up to the time {plan.max_time}, the largest rate of "
                f"change is {worst:.3g}."
            )
        horizon *= 10.0
        index += 1


def _event_at_the_horizon(
    model: RoadrunnerSBMLModel, block: np.ndarray, end: float, markers: int
) -> bool:
    """Tell whether roadrunner fired an event before its time at the end of a horizon.

    An event fired within the resolution of the time before the end, which
    the markers of the events tell, see `sbmlsim.model.event_rows`, and the
    trigger of an event of the model, in the state after it, is false at the
    end and true at the resolution of the time after it: roadrunner
    evaluates the triggers again at the start of the next horizon, at the
    end, and would fire such an event a second time right after it, see
    `_EventAtTheHorizon`. An event whose assignments make its own trigger
    false, e.g. one which fires every 0.01 and records the time it fired, is
    no such event. A trigger which is no math the simulator evaluates is not
    evaluated.

    Args:
        model: the model after the steps of the horizon.
        block: the steps of the horizon, the markers the last columns.
        end: the end of the horizon in the time of roadrunner.
        markers: the number of markers.

    Returns:
        Whether the horizon is integrated again with exact output at its end.
    """
    tolerance = time_tolerance(float(block[0, 0]), end)
    before = np.flatnonzero(block[:, 0] < end - tolerance)
    if block[-1, 0] < end - tolerance or before.size == 0:
        return False
    if np.array_equal(block[before[-1], -markers:], block[-1, -markers:]):
        return False
    r = model.r_loaded
    events = model.symbols.events
    at_the_end = triggers(events, r, end, model)
    after_it = triggers(events, r, end + tolerance, model)
    return any(
        at is False and later is True
        for at, later in zip(at_the_end, after_it, strict=True)
    )
