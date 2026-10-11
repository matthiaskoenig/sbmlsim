"""The executor runs a plan on roadrunner.

The semantics are the ones of PEtab v2:

1. The model is initialized with the pre-initialization values, see
   `RoadrunnerSBMLModel.initialize`: the initial assignments follow a changed
   parameter, and nothing of an earlier simulation is left.
2. A presimulation integrates until the rates of change vanish.
3. The interval is split at the times of the events. At the start of a
   segment the events of its time are applied: every value is evaluated
   first, with the state at that time, then every target is set; a
   compartment keeps the concentration of the concentration species in it.
4. After a change at a time after the start, the events whose triggers the
   change made true fire, which roadrunner does not see.
5. A segment is integrated with the output of the plan and ends in its
   state at its end, which the next change is applied to; for the steps of
   the integrator see `_steps`. A time of a change appears once in the
   output, with the state after the change. The state before a change after
   the start is the end of the segment before it, which
   `execute_with_changes` gives next to the output with the state after it,
   see
   `sbmlsim.fit.options.FitSettings.data_at_change`.
6. A model which does not read the time (`ModelSymbols.time_dependent`) is
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
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import roadrunner

from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from sbmlsim.model.symbols import EventSymbols, TargetKind
from sbmlsim.result.timecourse import TimecourseResult
from sbmlsim.simulator.formula import compile_formula
from sbmlsim.simulator.plan import (
    OutputMode,
    Plan,
    PlanEvent,
    SteadyStatePlan,
    preinit_targets,
    time_tolerance,
)

logger = logging.getLogger(__name__)

#: the setting of the integrator which switches between the output of its
#: steps and exact output times
VARIABLE_STEP_SIZE = "variable_step_size"


class SteadyStateError(RuntimeError):
    """A presimulation did not reach a steady state."""


@dataclass
class _ChangeStates:
    """The states before and after every change after the start, in their order."""

    before: list[np.ndarray] = field(default_factory=list)
    after: list[np.ndarray] = field(default_factory=list)


class _PastTheEnd(Exception):
    """roadrunner integrated a segment with the steps of the integrator past its end.

    An event of the model in the last step of roadrunner after the end of the
    segment fired or is pending, see `_steps`, and the state after the
    segment is wrong. `_execute` runs the plan again, which integrates the
    segment with exact output at the times of its steps (`_exact_times`) and
    keeps the state at the start of every segment of the steps after it
    (`RoadRunner.saveStateS`): a later segment past its end starts again from
    its state at the start (`RoadRunner.loadStateS`) with exact output at the
    times of its steps. A plan runs at most twice however many of its
    segments roadrunner integrates past their end; the second run costs a
    saved state per segment after the first one past its end and a loaded
    state per later one, only for a model with events.

    Attributes:
        segment: the index of the segment.
        times: its output times in the time of roadrunner.
    """

    def __init__(self, segment: int, times: np.ndarray) -> None:
        super().__init__(f"The segment {segment} was integrated past its end.")
        self.segment = segment
        self.times = times


def execute(
    plan: Plan, model: RoadrunnerSBMLModel, selections: Sequence[str]
) -> TimecourseResult:
    """Run a plan on a loaded model.

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
    result, _ = _execute(plan, model, selections, keep_states=False)
    return result


def execute_with_changes(
    plan: Plan, model: RoadrunnerSBMLModel, selections: Sequence[str]
) -> tuple[TimecourseResult, TimecourseResult, TimecourseResult]:
    """Run a plan and keep the states before and after every change after the start.

    The output is the one of `execute`, the state after a change at its time
    if it has the time. The state before it is the end of the integration of
    the segment before the change, which the output drops, and the state
    after it the start of the segment after it, which an output without the
    time drops; a fit reads its data at the time of a change and the jump of
    a curve there, see `sbmlsim.fit.options.FitSettings.data_at_change`.

    Args:
        plan: the plan, compiled against the model.
        model: the loaded model.
        selections: the columns of the result, `time` is added as the first
            column if it is missing.

    Returns:
        The result of `execute`, the states before the changes and the states
        after them: a row per time of `Plan.change_times`, with the columns of
        the result and the time shifted by `plan.time_shift`.

    Raises:
        SteadyStateError: if the presimulation does not reach a steady state.
        RuntimeError: if roadrunner fails to integrate.
    """
    result, states = _execute(plan, model, selections, keep_states=True)
    if states is None:
        raise RuntimeError("The states at the changes were not kept.")
    before, after = states
    return result, before, after


def _execute(
    plan: Plan,
    model: RoadrunnerSBMLModel,
    selections: Sequence[str],
    keep_states: bool,
) -> tuple[TimecourseResult, tuple[TimecourseResult, TimecourseResult] | None]:
    """Run a plan, see `execute` and `execute_with_changes`."""
    # a model may have an entity named `time` as well, which is a second
    # column `time` after the time of the simulation (case 01820)
    columns = list(selections)
    if not columns or columns[0] != "time":
        columns = ["time", *columns]
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
    variable_step_size = bool(r.getIntegrator().getValue(VARIABLE_STEP_SIZE))
    # the output times of the segments of the steps of the integrator with
    # exact output, see `_PastTheEnd`
    exact: dict[int, np.ndarray] = {}
    while True:
        model.initialize(preinit_targets(plan))
        if list(r.timeCourseSelections) != columns:
            r.timeCourseSelections = columns
        states = _ChangeStates() if keep_states else None
        try:
            if plan.steady_state is not None:
                steady_state(model, plan.steady_state)
            values = _simulate(plan, model, r, columns, local, states, exact)
        except _PastTheEnd as err:
            # a segment with exact output is never past its end
            exact[err.segment] = err.times
            continue
        finally:
            r.getIntegrator().setValue(VARIABLE_STEP_SIZE, variable_step_size)
            model.simulated()
        break

    if plan.time_shift != 0.0:
        values[:, 0] += plan.time_shift
    result = TimecourseResult(columns=tuple(columns), values=values)
    if states is None:
        return result, None
    kept: list[TimecourseResult] = []
    for rows in (states.before, states.after):
        array = np.array(rows, dtype=float).reshape(len(rows), len(columns))
        if plan.time_shift != 0.0:
            array[:, 0] += plan.time_shift
        kept.append(TimecourseResult(columns=tuple(columns), values=array))
    return result, (kept[0], kept[1])


def _simulate(
    plan: Plan,
    model: RoadrunnerSBMLModel,
    r: roadrunner.RoadRunner,
    columns: list[str],
    local: bool,
    states: _ChangeStates | None = None,
    exact: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Integrate the segments of a plan, see the module.

    Args:
        plan: the plan.
        model: the loaded model.
        r: its roadrunner instance.
        columns: the selections.
        local: integrate every segment from the time 0 of roadrunner (local
            time) instead of its absolute time.
        states: the states before and after every change after the start
            are appended to it, in the order of the changes; `None` to keep
            none.
        exact: the output times in the time of roadrunner of the segments of
            the steps of the integrator with exact output, by the index of
            the segment, see `_PastTheEnd`.

    Returns:
        The values, a row per output time and a column per selection.

    Raises:
        _PastTheEnd: if roadrunner integrated a segment of the steps of the
            integrator past its end before any segment with exact output.
    """
    events = {event.time: event for event in plan.events}
    points = sorted({plan.start, plan.end, *events})
    model_events = plan.symbols.events
    times = np.asarray(plan.times, dtype=float)
    tolerance = time_tolerance(plan.start, plan.end)
    exact = exact or {}

    blocks: list[np.ndarray] = []
    for k in range(len(points) - 1):
        a, b = points[k], points[k + 1]
        last = k == len(points) - 2
        if a in events:
            if model_events and (a > plan.start or plan.steady_state is not None):
                # the triggers at the end of the integration before the change,
                # the one of the steady state for the change at the start
                triggers = _triggers(
                    model_events, r, a if local else float(r.model.getTime())
                )
                _apply(events[a], r, plan)
                _fire_events(model_events, triggers, r, a, model)
            else:
                # at the start after a reset roadrunner evaluates the triggers
                # itself
                _apply(events[a], r, plan)

        # the segment in the time of roadrunner, from 0 in local time
        offset = a if local else 0.0
        start, end = a - offset, b - offset
        # an event of the model at the time of the next change fires after
        # the change (PEtab v2, reinitialization): the integration stops just
        # before it, where roadrunner does not fire it, see `_fire_events`
        end_stop = (
            float(np.nextafter(end, -np.inf)) if model_events and not last else end
        )
        if plan.output is OutputMode.INTEGRATOR:
            output = exact.get(k)
            # the state at the start of a segment after one with exact output,
            # see `_PastTheEnd`
            saved = (
                r.saveStateS() if output is None and exact and k > min(exact) else None
            )
            if output is None:
                block, at_end = _steps(
                    r, start, end_stop, tolerance, bool(model_events)
                )
                if not at_end:
                    output = _exact_times(start, block[1:-1, 0], end_stop, tolerance)
                    if saved is None:
                        raise _PastTheEnd(k, output)
                    r.loadStateS(saved)
            if output is not None:
                r.getIntegrator().setValue(VARIABLE_STEP_SIZE, False)
                block = np.array(r.simulate(times=output.tolist()), dtype=float)
            block[:, 0] += offset
            if last:
                # the end of the simulation, not the end of the segment in the
                # time of roadrunner shifted back
                block[-1, 0] = b
            start_state = block[0].copy()
            end_state = block[-1].copy()
            if not last:
                # the state at `b` is the one before the change at `b`
                block = block[:-1]
        else:
            r.getIntegrator().setValue(VARIABLE_STEP_SIZE, False)
            wanted = times[(times >= a) & ((times <= b) if last else (times < b))]
            shifted = wanted - offset
            grid = np.unique(np.concatenate([[start], shifted, [end_stop]]))
            result = np.array(r.simulate(times=grid.tolist()), dtype=float)
            block = result[np.isin(grid, shifted)]
            if block.shape[0] != wanted.size:
                raise RuntimeError(
                    f"The output times of the segment [{a}, {b}] are not distinct "
                    f"in local time."
                )
            # the output times are the ones asked for, not the shifted ones
            block[:, 0] = wanted
            start_state = result[0].copy()
            end_state = result[-1].copy()
        if states is not None:
            if a > plan.start and a in events:
                start_state[0] = a
                states.after.append(start_state)
            if b > plan.start and b in events:
                end_state[0] = b
                states.before.append(end_state)
        blocks.append(block)

    if plan.end in events:
        # a change at the end is applied after the integration, the last
        # output is the state after it and the events it triggers
        if model_events and plan.end > plan.start:
            triggers = _triggers(
                model_events, r, plan.end if local else float(r.model.getTime())
            )
            _apply(events[plan.end], r, plan)
            _fire_events(model_events, triggers, r, plan.end, model)
        else:
            _apply(events[plan.end], r, plan)
        after = _state(r, columns, plan.end)
        if states is not None and plan.end > plan.start:
            states.after.append(after.copy())
        if blocks and blocks[-1].shape[0] and blocks[-1][-1, 0] == plan.end:
            blocks[-1][-1, :] = after

    if plan.steady_state_output is not None:
        # the steady state after the end, PEtab's measurement at `inf`
        steady_state(model, plan.steady_state_output, start=0.0 if local else plan.end)
        row = _state(r, columns, np.inf)
        blocks.append(row[np.newaxis, :])

    if not blocks:
        return np.empty((0, len(columns)))
    return np.concatenate(blocks, axis=0)


def _steps(
    r: roadrunner.RoadRunner, start: float, end: float, tolerance: float, events: bool
) -> tuple[np.ndarray, bool]:
    """Integrate a segment with the steps of the integrator.

    roadrunner integrates with the steps of CVODE up to the step after the
    end and interpolates its last row linearly to the end, the model is left
    in the state of the later time; `_to_the_end` sets it to the state at the
    end, which the last row takes. roadrunner also fires an event of the
    model it finds in that step after the end, or keeps it pending and fires
    it in its next integration, also of the next simulation: such a segment
    is integrated past its end and gets exact output, see `_PastTheEnd`.

    Args:
        r: the roadrunner instance, in the state at the start.
        start: the start of the segment in the time of roadrunner.
        end: its end in the time of roadrunner.
        tolerance: the resolution of the time: a step of a model with events
            which is closer to the end is an event which roadrunner fired at
            the end.
        events: whether the model has events.

    Returns:
        The values in the time of roadrunner, a row per step, the first at
        the start and the last at the end, and whether the segment ended at
        its end; if not, the last row is the one of roadrunner.
    """
    block = _variable_steps(r, start, end)
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


def _variable_steps(r: roadrunner.RoadRunner, start: float, end: float) -> np.ndarray:
    """Integrate with the steps of the integrator from `start` to `end`.

    roadrunner warns that the end was not reached if the last time is more
    than `end / 10000` before it, which is every negative end; the end is
    checked by `_steps`, so the warning of a negative end is not shown.
    """
    r.getIntegrator().setValue(VARIABLE_STEP_SIZE, True)
    if end >= 0.0:
        return np.array(r.simulate(start, end), dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Simulation requested end time point",
            category=UserWarning,
        )
        return np.array(r.simulate(start, end), dtype=float)


def _to_the_end(r: roadrunner.RoadRunner, end: float, events: bool) -> bool:
    """Set the model to its state at the end of an integration with its steps.

    The integrator is called with exact output at the end, which CVODE
    interpolates from its last step without integrating. An event of the
    model which roadrunner keeps pending fires first, after the end.

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
) -> np.ndarray:
    """Get the output times of a segment with exact output, see `_PastTheEnd`.

    The times are the start, the steps before the end and the end. A step
    closer to the end than the resolution of the time is an event which
    roadrunner fired at the end and is left out. Steps closer to each other
    than the resolution are an event of the model, the state before it and
    the one after it, which roadrunner gives at times which differ by less
    than its tolerance of a root, or at one time: they are the times a
    resolution before the first and after the last of them, so the exact
    output has the state before the event and the one after it wherever
    CVODE finds the event between them.

    Args:
        start: the start of the segment in the time of roadrunner.
        steps: the steps of the integrator inside the segment, sorted.
        end: its end in the time of roadrunner.
        tolerance: the resolution of the time.

    Returns:
        The output times, sorted and distinct.
    """
    steps = np.asarray(steps, dtype=float)
    steps = steps[steps <= end - tolerance]
    times = [start]
    k = 0
    while k < steps.size:
        last = k
        while last + 1 < steps.size and steps[last + 1] - steps[last] < tolerance:
            last += 1
        if last > k:
            times.extend([steps[k] - tolerance, steps[last] + tolerance])
        else:
            times.append(steps[k])
        k = last + 1
    times.append(end)
    output = np.unique(times)
    return output[(output >= start) & (output <= end)]


def _state(r: roadrunner.RoadRunner, columns: list[str], time: float) -> np.ndarray:
    """Get the current values of the selections."""
    return np.array([time if c == "time" else r.getValue(c) for c in columns])


def _apply(event: PlanEvent, r: roadrunner.RoadRunner, plan: Plan) -> None:
    """Apply the assignments of an event, see the module."""
    values: list[tuple[str, TargetKind, float]] = []
    for a in event.assignments:
        if a.value is not None:
            value = a.value
        elif a.formula is None:
            raise ValueError(f"The assignment of '{a.target}' has no value.")
        else:
            formula = compile_formula(a.formula)
            value = formula.evaluate(
                [event.time if s == "time" else r.getValue(s) for s in formula.symbols]
            )
        values.append((a.target, a.kind, value))

    symbols = plan.symbols
    targets = {symbols.entity(target) for target, _, _ in values}
    for target, kind, value in values:
        if kind is not TargetKind.COMPARTMENT:
            continue
        # PEtab: the concentration of a concentration species is kept,
        # roadrunner keeps the amount of every species
        kept = {
            s: r.getValue(f"[{s}]")
            for s, c in symbols.species_compartment.items()
            if c == target and s not in symbols.only_substance and s not in targets
        }
        r.setValue(target, value)
        for species, concentration in kept.items():
            r.setValue(f"[{species}]", concentration)
    for target, kind, value in values:
        if kind is not TargetKind.COMPARTMENT:
            r.setValue(target, value)


def _triggers(
    events: Sequence[EventSymbols], r: roadrunner.RoadRunner, time: float
) -> list[bool | None]:
    """Evaluate the triggers of the events, `None` for one which is not evaluated."""
    values: list[bool | None] = []
    for event in events:
        if event.trigger is None:
            values.append(None)
            continue
        try:
            formula = compile_formula(event.trigger)
        except ValueError:
            values.append(None)
            continue
        values.append(
            bool(
                formula.evaluate(
                    [time if s == "time" else r.getValue(s) for s in formula.symbols]
                )
            )
        )
    return values


#: the most rounds of events which a change triggers, an event may trigger
#: another one
MAX_EVENT_ROUNDS = 100


def _fire_events(
    events: Sequence[EventSymbols],
    before: list[bool | None],
    r: roadrunner.RoadRunner,
    time: float,
    model: RoadrunnerSBMLModel,
) -> None:
    """Fire the events whose triggers a change made true.

    roadrunner does not see a trigger which becomes true by a value which is
    set between two integrations, so the executor fires such events (PEtab
    v2, reinitialization: the events are applied after the changes): the
    values of the assignments of every triggered event are evaluated, then
    set, and the events these assignments trigger fire in the next round. An
    event with a delay is reported and not fired.

    Args:
        events: the events of the model.
        before: the triggers before the change.
        r: the roadrunner instance, in the state after the change.
        time: the time of the change.
        model: the model, for the messages.
    """
    for _ in range(MAX_EVENT_ROUNDS):
        after = _triggers(events, r, time)
        fired = [
            event
            for event, old, new in zip(events, before, after, strict=True)
            if old is False and new is True
        ]
        if not fired:
            return
        values: list[tuple[str, float]] = []
        for event in fired:
            if event.delayed:
                logger.warning(
                    "The event '%s' of the model '%s' is triggered by a change at "
                    "the time %s and has a delay, which the simulator does not "
                    "schedule: it does not fire.",
                    event.eid,
                    model.sid or r.model.getModelName(),
                    time,
                )
                continue
            try:
                formulas = [(t, compile_formula(f)) for t, f in event.assignments]
            except ValueError as err:
                logger.warning(
                    "The event '%s' of the model '%s' is triggered by a change at "
                    "the time %s, its assignments are no formulas the simulator "
                    "evaluates: %s",
                    event.eid,
                    model.sid or r.model.getModelName(),
                    time,
                    err,
                )
                continue
            for target, formula in formulas:
                values.append(
                    (
                        target,
                        formula.evaluate(
                            [
                                time if s == "time" else r.getValue(s)
                                for s in formula.symbols
                            ]
                        ),
                    )
                )
        for target, value in values:
            r.setValue(target, value)
        before = after
    logger.warning(
        "The events of the model '%s' at the time %s trigger each other more than "
        "%s times.",
        model.sid or r.model.getModelName(),
        time,
        MAX_EVENT_ROUNDS,
    )


def steady_state(
    model: RoadrunnerSBMLModel, plan: SteadyStatePlan, start: float = 0.0
) -> float:
    """Integrate a model until its rates of change vanish.

    The model is integrated with the steps of the integrator over horizons
    which grow by a factor of ten, from `start`, until
    `|dx/dt| <= absolute_tolerance + relative_tolerance * |x|` for every
    state, events active. Every horizon ends in the state at its end, see
    `_to_the_end`.

    Args:
        model: the model in the state the integration starts from.
        plan: the steady state.
        start: the time the integration starts at.

    Returns:
        The time the steady state was reached at.

    Raises:
        SteadyStateError: if the steady state is not reached by `max_time`.
    """
    r = model.r_loaded
    events = bool(model.symbols.events)
    time = start
    horizon = 1.0
    while True:
        end = start + min(horizon, plan.max_time)
        _variable_steps(r, time, end)
        # the state at the end of the horizon, an event of the model which
        # roadrunner keeps pending fires in the presimulation
        _to_the_end(r, end, events)
        time = end
        # the rates of the species and of the targets of rate rules, `x'`
        named = r.getRatesOfChangeNamedArray()
        rates = np.abs(np.asarray(named, dtype=float).ravel())
        state = np.abs(np.array([r.getValue(name[:-1]) for name in named.colnames]))
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
