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
   `execute_with_before_changes` gives next to the output, see
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
from collections.abc import Sequence

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


class _PastTheEnd(Exception):
    """roadrunner integrated a segment with the steps of the integrator past its end.

    An event of the model in the last step of roadrunner after the end of the
    segment fired or is pending, see `_steps`; `_execute` runs the plan again
    and integrates the segment with exact output at the times of its steps.

    Attributes:
        segment: the index of the segment.
        times: the output times of the segment in the time of roadrunner, its
            start, its steps before the end and its end.
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
    result, _ = _execute(plan, model, selections, keep_before=False)
    return result


def execute_with_before_changes(
    plan: Plan, model: RoadrunnerSBMLModel, selections: Sequence[str]
) -> tuple[TimecourseResult, TimecourseResult]:
    """Run a plan and keep the state before every change after the start.

    The output is the one of `execute`, the state after a change at its time.
    The state before it is the end of the integration of the segment before
    the change, which the output drops; a fit reads its data at the time of a
    change there, see `sbmlsim.fit.options.FitSettings.data_at_change`.

    Args:
        plan: the plan, compiled against the model.
        model: the loaded model.
        selections: the columns of the result, `time` is added as the first
            column if it is missing.

    Returns:
        The result of `execute` and the states before the changes: a row per
        time of `Plan.change_times`, with the columns of the result and the
        time shifted by `plan.time_shift`.

    Raises:
        SteadyStateError: if the presimulation does not reach a steady state.
        RuntimeError: if roadrunner fails to integrate.
    """
    result, before = _execute(plan, model, selections, keep_before=True)
    if before is None:
        raise RuntimeError("The states before the changes were not kept.")
    return result, before


def _execute(
    plan: Plan,
    model: RoadrunnerSBMLModel,
    selections: Sequence[str],
    keep_before: bool,
) -> tuple[TimecourseResult, TimecourseResult | None]:
    """Run a plan, see `execute` and `execute_with_before_changes`."""
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
    integrator: roadrunner.Integrator = r.getIntegrator()
    variable_step_size = bool(integrator.getValue(VARIABLE_STEP_SIZE))
    # the segments of the steps of the integrator which are integrated with
    # exact output at the times of their steps, see `_PastTheEnd`
    exact: dict[int, np.ndarray] = {}
    while True:
        model.initialize(preinit_targets(plan))
        if list(r.timeCourseSelections) != columns:
            r.timeCourseSelections = columns
        before: list[np.ndarray] | None = [] if keep_before else None
        try:
            if plan.steady_state is not None:
                steady_state(model, plan.steady_state)
            values = _simulate(plan, model, r, columns, local, before, exact)
        except _PastTheEnd as err:
            if err.segment in exact:
                raise RuntimeError(str(err)) from err
            exact[err.segment] = err.times
            continue
        finally:
            integrator.setValue(VARIABLE_STEP_SIZE, variable_step_size)
            model.simulated()
        break

    if plan.time_shift != 0.0:
        values[:, 0] += plan.time_shift
    result = TimecourseResult(columns=tuple(columns), values=values)
    if before is None:
        return result, None
    states = np.array(before, dtype=float).reshape(len(before), len(columns))
    if plan.time_shift != 0.0:
        states[:, 0] += plan.time_shift
    return result, TimecourseResult(columns=tuple(columns), values=states)


def _simulate(
    plan: Plan,
    model: RoadrunnerSBMLModel,
    r: roadrunner.RoadRunner,
    columns: list[str],
    local: bool,
    before: list[np.ndarray] | None = None,
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
        before: the list the state before every change after the start is
            appended to, in the order of the changes; `None` to keep none.
        exact: the output times in the time of roadrunner of the segments of
            the steps of the integrator which are integrated with exact
            output, by the index of the segment, see `_PastTheEnd`.

    Returns:
        The values, a row per output time and a column per selection.

    Raises:
        _PastTheEnd: if roadrunner integrated a segment of the steps of the
            integrator past its end.
    """
    events = {event.time: event for event in plan.events}
    points = sorted({plan.start, plan.end, *events})
    model_events = plan.symbols.events
    integrator: roadrunner.Integrator = r.getIntegrator()
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
            block = _steps(r, start, end_stop, tolerance, k, exact.get(k))
            block[:, 0] += offset
            if last:
                # the end of the simulation, not the end of the segment in the
                # time of roadrunner shifted back
                block[-1, 0] = b
            end_state = block[-1].copy()
            if not last:
                # the state at `b` is the one before the change at `b`
                block = block[:-1]
        else:
            integrator.setValue(VARIABLE_STEP_SIZE, False)
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
            end_state = result[-1].copy()
        if before is not None and b > plan.start and b in events:
            end_state[0] = b
            before.append(end_state)
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
        if blocks and blocks[-1].shape[0] and blocks[-1][-1, 0] == plan.end:
            blocks[-1][-1, :] = _state(r, columns, plan.end)

    if plan.steady_state_output is not None:
        # the steady state after the end, PEtab's measurement at `inf`
        steady_state(model, plan.steady_state_output, start=0.0 if local else plan.end)
        row = _state(r, columns, np.inf)
        blocks.append(row[np.newaxis, :])

    if not blocks:
        return np.empty((0, len(columns)))
    return np.concatenate(blocks, axis=0)


def _steps(
    r: roadrunner.RoadRunner,
    start: float,
    end: float,
    tolerance: float,
    segment: int,
    times: np.ndarray | None,
) -> np.ndarray:
    """Integrate a segment with the steps of the integrator.

    roadrunner integrates with the steps of CVODE up to the step after the
    end and interpolates its last row linearly to the end, the model is left
    in the state of the later time. The integrator is therefore called once
    more with exact output at the end, which interpolates the state at the
    end from the last step of CVODE without integrating: the last row and
    the model are the state at the end. roadrunner also fires an event of
    the model it finds in that step after the end, or keeps it pending and
    fires it in its next integration, also of the next simulation; the call
    at the end fires such a pending event, and a segment past whose end an
    event fired is integrated again with exact output at the times of its
    steps, see `_PastTheEnd`.

    Args:
        r: the roadrunner instance, in the state at the start.
        start: the start of the segment in the time of roadrunner.
        end: its end in the time of roadrunner.
        tolerance: the resolution of the time, a step closer to the end is an
            event of the model which roadrunner fired at the end.
        segment: the index of the segment.
        times: the output times of a segment which is integrated with exact
            output, see `_PastTheEnd`; `None` for the steps of the integrator.

    Returns:
        The values in the time of roadrunner, a row per step, the first at
        the start and the last at the end.

    Raises:
        _PastTheEnd: if roadrunner integrated past the end.
    """
    integrator: roadrunner.Integrator = r.getIntegrator()
    if times is not None:
        integrator.setValue(VARIABLE_STEP_SIZE, False)
        return np.array(r.simulate(times=times.tolist()), dtype=float)
    integrator.setValue(VARIABLE_STEP_SIZE, True)
    block = np.array(r.simulate(start, end), dtype=float)
    integrator.setValue(VARIABLE_STEP_SIZE, False)
    try:
        reached = float(integrator.integrate(end, 0.0))
    except RuntimeError:
        reached = np.nan
    steps = block[1:-1, 0]
    if reached != end or (steps.size and steps[-1] > end - tolerance):
        raise _PastTheEnd(
            segment, np.unique([start, *steps[steps <= end - tolerance], end])
        )
    block[-1] = r.getSelectedValues()
    block[-1, 0] = end
    return block


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
    state, events active.

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
    r.getIntegrator().setValue(VARIABLE_STEP_SIZE, True)
    time = start
    horizon = 1.0
    while True:
        end = start + min(horizon, plan.max_time)
        r.simulate(time, end)
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
