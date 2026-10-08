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
5. A segment is integrated with the output of the plan. A time of a change
   appears once in the output, with the state after the change.

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
)

logger = logging.getLogger(__name__)

#: the setting of the integrator which switches between the output of its
#: steps and exact output times
VARIABLE_STEP_SIZE = "variable_step_size"


class SteadyStateError(RuntimeError):
    """A presimulation did not reach a steady state."""


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
    # a model may have an entity named `time` as well, which is a second
    # column `time` after the time of the simulation (case 01820)
    columns = list(selections)
    if not columns or columns[0] != "time":
        columns = ["time", *columns]
    r = model.r_loaded
    if plan.start != 0.0 and r.model.getNumEvents() > 0 and not model.warned_start:
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
    model.initialize(preinit_targets(plan))
    if list(r.timeCourseSelections) != columns:
        r.timeCourseSelections = columns

    integrator: roadrunner.Integrator = r.getIntegrator()
    variable_step_size = bool(integrator.getValue(VARIABLE_STEP_SIZE))
    try:
        if plan.steady_state is not None:
            steady_state(model, plan.steady_state)
        values = _simulate(plan, model, r, columns)
    finally:
        integrator.setValue(VARIABLE_STEP_SIZE, variable_step_size)
        model.simulated()

    if plan.time_shift != 0.0:
        values[:, 0] += plan.time_shift
    return TimecourseResult(columns=tuple(columns), values=values)


def _simulate(
    plan: Plan,
    model: RoadrunnerSBMLModel,
    r: roadrunner.RoadRunner,
    columns: list[str],
) -> np.ndarray:
    """Integrate the segments of a plan, see the module.

    Returns:
        The values, a row per output time and a column per selection.
    """
    events = {event.time: event for event in plan.events}
    points = sorted({plan.start, plan.end, *events})
    model_events = plan.symbols.events
    integrator: roadrunner.Integrator = r.getIntegrator()
    times = np.asarray(plan.times, dtype=float)

    blocks: list[np.ndarray] = []
    for k in range(len(points) - 1):
        a, b = points[k], points[k + 1]
        last = k == len(points) - 2
        if a in events:
            if model_events and (a > plan.start or plan.steady_state is not None):
                # the triggers at the end of the integration before the change,
                # the one of the steady state for the change at the start
                triggers = _triggers(model_events, r, float(r.model.getTime()))
                _apply(events[a], r, plan)
                _fire_events(model_events, triggers, r, a, model)
            else:
                # at the start after a reset roadrunner evaluates the triggers
                # itself
                _apply(events[a], r, plan)

        # an event of the model at the time of the next change fires after
        # the change (PEtab v2, reinitialization): the integration stops just
        # before it, where roadrunner does not fire it, see `_fire_events`
        b_end = float(np.nextafter(b, -np.inf)) if model_events and not last else b
        if plan.output is OutputMode.INTEGRATOR:
            integrator.setValue(VARIABLE_STEP_SIZE, True)
            block = np.array(r.simulate(a, b_end), dtype=float)
            if not last:
                # the state at `b` is the one before the change at `b`
                block = block[:-1]
        else:
            integrator.setValue(VARIABLE_STEP_SIZE, False)
            wanted = times[(times >= a) & ((times <= b) if last else (times < b))]
            grid = np.unique(np.concatenate([[a], wanted, [b_end]]))
            result = np.array(r.simulate(times=grid.tolist()), dtype=float)
            block = result[np.isin(grid, wanted)]
        blocks.append(block)

    if plan.end in events:
        # a change at the end is applied after the integration, the last
        # output is the state after it and the events it triggers
        if model_events and plan.end > plan.start:
            triggers = _triggers(model_events, r, float(r.model.getTime()))
            _apply(events[plan.end], r, plan)
            _fire_events(model_events, triggers, r, plan.end, model)
        else:
            _apply(events[plan.end], r, plan)
        if blocks and blocks[-1].shape[0] and blocks[-1][-1, 0] == plan.end:
            blocks[-1][-1, :] = _state(r, columns, plan.end)

    if plan.steady_state_output is not None:
        # the steady state after the end, PEtab's measurement at `inf`
        steady_state(model, plan.steady_state_output, start=plan.end)
        row = _state(r, columns, np.inf)
        blocks.append(row[np.newaxis, :])

    if not blocks:
        return np.empty((0, len(columns)))
    return np.concatenate(blocks, axis=0)


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
