"""The changes of a plan and the events a change fires, independent of the engine.

An executor applies the changes of a plan at their times and fires the events
of the model whose triggers a change made true (PEtab v2, reinitialization),
which an engine does not see, since the values are set between two
integrations. Both read and write the values of the model through
`ModelValues`, the `getValue`/`setValue` of the selections of roadrunner,
which `roadrunner.RoadRunner` has and an executor of another engine provides,
see `sbmlsim.simulator.executor`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from sbmlsim.model.symbols import EventSymbols, TargetKind
from sbmlsim.simulator.formula import compile_formula
from sbmlsim.simulator.plan import Plan, PlanEvent

logger = logging.getLogger(__name__)

#: the most times one event fires after a change, to stop event cycles without
#: limiting the number of independent simultaneous events
MAX_EXECUTIONS_PER_EVENT = 100


class ModelValues(Protocol):
    """The current values of a model in the selections of roadrunner.

    A selection is `S` for the amount of a species, `[S]` for its
    concentration, or the id of a parameter or a compartment.
    `roadrunner.RoadRunner` satisfies the protocol.
    """

    def getValue(self, selection: str, /) -> float:
        """Get the current value of a selection."""
        ...

    def setValue(self, selection: str, value: float, /) -> None:
        """Set the current value of a selection."""
        ...


class SBMLModelLike(Protocol):
    """The model of the messages of the events.

    Attributes:
        sid: id of the model, `None` without one.
        reported_triggers: the events whose trigger was reported as no math
            the simulator evaluates, so a trigger is reported once.
    """

    sid: str | None
    reported_triggers: set[str]

    @property
    def label(self) -> str:
        """Get the name of the model in a message, see `SBMLModel.label`."""
        ...


def apply_changes(event: PlanEvent, values: ModelValues, plan: Plan) -> None:
    """Apply the assignments of a change.

    Every value is evaluated first, with the state at the time of the change,
    then every target is set; a compartment keeps the concentration of the
    concentration species in it (PEtab v2).

    Args:
        event: the assignments at one time.
        values: the values of the model.
        plan: the plan, whose symbols resolve the targets.

    Raises:
        ValueError: if an assignment has neither a value nor a formula.
    """
    assigned: list[tuple[str, TargetKind, float]] = []
    for a in event.assignments:
        if a.value is not None:
            value = a.value
        elif a.formula is None:
            raise ValueError(f"The assignment of '{a.target}' has no value.")
        else:
            formula = compile_formula(a.formula)
            value = formula.evaluate(
                [
                    event.time if s == "time" else values.getValue(s)
                    for s in formula.symbols
                ]
            )
        assigned.append((a.target, a.kind, value))

    symbols = plan.symbols
    targets = {symbols.entity(target) for target, _, _ in assigned}
    for target, kind, value in assigned:
        if kind is not TargetKind.COMPARTMENT:
            continue
        # PEtab: the concentration of a concentration species is kept,
        # roadrunner keeps the amount of every species; a species of an
        # assignment rule follows its rule, roadrunner refuses to set it (#300)
        kept = {
            s: values.getValue(f"[{s}]")
            for s, c in symbols.species_compartment.items()
            if c == target
            and s not in symbols.only_substance
            and s not in symbols.assignment_rules
            and s not in targets
        }
        values.setValue(target, value)
        for species, concentration in kept.items():
            values.setValue(f"[{species}]", concentration)
    for target, kind, value in assigned:
        if kind is not TargetKind.COMPARTMENT:
            values.setValue(target, value)


def triggers(
    events: Sequence[EventSymbols],
    values: ModelValues,
    time: float,
    model: SBMLModelLike,
) -> list[bool | None]:
    """Evaluate the triggers of the events, `None` for one which is not evaluated.

    A trigger which is no math the simulator evaluates, one with `rateOf` or
    `delay`, see `sbmlsim.sbml_math.compile_math`, is not evaluated, so a
    change does not fire its event; it is reported once per event of a model
    instance, so once per worker of a pool, which loads its own instance.

    Args:
        events: the events of the model.
        values: the values of the model.
        time: the time of the model.
        model: the model, for the messages.

    Returns:
        The value of every trigger.
    """
    # sbmlmath is imported by the first model with events, see `sbml_math`
    from sbmlsim.sbml_math import compile_math, math_formula

    result: list[bool | None] = []
    for event in events:
        if event.trigger is None:
            result.append(None)
            continue
        try:
            formula = compile_math(event.trigger)
        except ValueError as err:
            result.append(None)
            if event.eid not in model.reported_triggers:
                model.reported_triggers.add(event.eid)
                logger.warning(
                    "The trigger '%s' of the event '%s' of the model '%s' is no "
                    "math the simulator evaluates, a change of a simulation "
                    "does not fire the event: %s",
                    math_formula(event.trigger),
                    event.eid,
                    model.label,
                    err,
                )
            continue
        result.append(
            bool(
                formula.evaluate(
                    [
                        time if s == "time" else values.getValue(s)
                        for s in formula.symbols
                    ]
                )
            )
        )
    return result


def fire_events(
    events: Sequence[EventSymbols],
    before: list[bool | None],
    values: ModelValues,
    time: float,
    model: SBMLModelLike,
    after: Sequence[bool | None] | None = None,
) -> list[int]:
    """Fire the events whose triggers a change made true.

    An engine does not see a trigger which becomes true by a value which is
    set between two integrations, so the executor fires such events (PEtab
    v2, reinitialization: the events are applied after the changes): the
    events execute one at a time in priority order. Priorities and triggers
    are evaluated again after every event, and a nonpersistent pending event
    is cancelled if its trigger becomes false. Assignment values are captured
    at the trigger or evaluated at execution, as the event specifies. An event
    with a delay is reported and not fired, an event whose priority is no
    math the simulator evaluates is reported and fires with the priority 0,
    the one of an event without a priority, and an event whose trigger or
    assignments are no math the simulator evaluates is reported and not
    fired, see `triggers`. One event fires at most `MAX_EXECUTIONS_PER_EVENT`
    times, which stops events which trigger each other in a cycle.

    Args:
        events: the events of the model.
        before: the triggers before the change.
        values: the values of the model, in the state after the change.
        time: the time of the change.
        model: the model, for the messages.
        after: the triggers after the change, evaluated in the state after
            it for `None`; the triggers after an event are always evaluated.
            An executor gives the triggers right after the time of the
            change, see `sbmlsim.simulator.executor_amici`.

    Returns:
        The events which fired, by their positions, in the order of their
        executions; an event which fired twice is listed twice.
    """
    from sbmlsim.sbml_math import compile_math

    def evaluate(mathml: str) -> float:
        compiled = compile_math(mathml)
        return compiled.evaluate(
            [time if s == "time" else values.getValue(s) for s in compiled.symbols]
        )

    def assignments(event: EventSymbols) -> list[tuple[str, float]]:
        return [(target, evaluate(formula)) for target, formula in event.assignments]

    def priority(index: int) -> float:
        formula = events[index].priority
        if formula is None or index in unprioritized:
            return 0.0
        return evaluate(formula)

    fired: list[int] = []
    executions = [0] * len(events)
    pending: dict[int, list[tuple[str, float]] | None] = {}
    # events whose priority is no math the simulator evaluates
    unprioritized: set[int] = set()
    given = after
    while True:
        current = (
            list(given) if given is not None else triggers(events, values, time, model)
        )
        given = None
        for index in list(pending):
            if not events[index].persistent and current[index] is False:
                del pending[index]
        for index, (event, old, new) in enumerate(
            zip(events, before, current, strict=True)
        ):
            if old is not False or new is not True or index in pending:
                continue
            if event.delayed:
                logger.warning(
                    "The event '%s' of the model '%s' is triggered by a change at "
                    "the time %s and has a delay, which the simulator does not "
                    "schedule: it does not fire.",
                    event.eid,
                    model.label,
                    time,
                )
                continue
            try:
                # Compile all assignments even when their values are deferred.
                for _, mathml in event.assignments:
                    compile_math(mathml)
                pending[index] = (
                    assignments(event) if event.use_values_from_trigger_time else None
                )
            except ValueError as err:
                logger.warning(
                    "The event '%s' of the model '%s' is triggered by a change at "
                    "the time %s, its assignments are no math the simulator "
                    "evaluates: %s",
                    event.eid,
                    model.label,
                    time,
                    err,
                )
                continue
            if event.priority is not None and index not in unprioritized:
                try:
                    compile_math(event.priority)
                except ValueError as err:
                    unprioritized.add(index)
                    logger.warning(
                        "The event '%s' of the model '%s' is triggered by a change "
                        "at the time %s, its priority is no math the simulator "
                        "evaluates, it fires with the priority 0: %s",
                        event.eid,
                        model.label,
                        time,
                        err,
                    )
        if not pending:
            return fired
        # the first of the highest priority in document order
        index = max(sorted(pending), key=priority)
        if executions[index] >= MAX_EXECUTIONS_PER_EVENT:
            logger.warning(
                "The event '%s' of the model '%s' fires more than %s times at the "
                "time %s, the events trigger each other in a cycle.",
                events[index].eid,
                model.label,
                MAX_EXECUTIONS_PER_EVENT,
                time,
            )
            return fired
        assigned = pending.pop(index)
        if assigned is None:
            assigned = assignments(events[index])
        for target, value in assigned:
            values.setValue(target, value)
        fired.append(index)
        executions[index] += 1
        before = current
