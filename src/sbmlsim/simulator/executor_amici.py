"""The executor of AMICI runs a plan on a model of AMICI.

It has the semantics of the executor of roadrunner, see
`sbmlsim.simulator.executor`, so the results of the engines are comparable:

1. The model is initialized with the pre-initialization values, see
   `AmiciSBMLModel.initialize`, from the free parameters as loaded, so
   nothing of an earlier simulation is left.
2. The events whose trigger is false before the time 0
   (`AmiciSBMLModel.initial_value_false`) and true in the initial state fire
   at the start, in the initial state before a change at the start, with
   the events they trigger, as in roadrunner (`_Course.start_events`); AMICI
   fires no event at the start of an integration, see
   `sbmlsim.model.model_amici`. With a presimulation they fire at its start,
   the time 0.
3. A presimulation integrates from the time 0 until the rates of change
   vanish, see `steady_state`.
4. The segments between the changes are the ones of
   `sbmlsim.simulator.segments.simulate_segments`, with the events a change
   triggers fired by the executor, the stop of a segment just before a
   change and the events at the end. AMICI integrates (CVODES) in absolute
   time, every segment from the state the one before it ended in, and finds
   the events of the model in an integration itself.
5. AMICI evaluates a trigger as true where the value of its comparisons is
   0 at the start of an integration, also a strict comparison, e.g.
   `time > 1` at the time 1 (its Heaviside variables are 1 for a root
   function of the value 0), and would miss the event when the trigger
   becomes true after the start. At the start of every segment and of every
   horizon of a steady state the model is integrated ahead to tell, the
   resolution of the time first and then ten times as far, `LOOK_AHEAD`
   times at most and not beyond the end of the integration, see
   `_Course.restart`. Such an event whose trigger is true at the resolution
   after the start fires at the start, after the row of the start, as in
   roadrunner, which fires it in its first step; the steps of the
   integrator have it as two rows at the start. One whose trigger becomes
   true later fires at the time it becomes true, located by bisection to
   the resolution of the time, where the integration is split
   (`_Course._integrate`), two rows in the steps of the integrator. A
   trigger which stays at its threshold for all of `LOOK_AHEAD` raises a
   `NotImplementedError` naming the events; an event AMICI itself fires
   within the time ahead changes the state the trigger is evaluated in
   there (not worked around). The same holds after an event: an event
   (of AMICI in an integration, or of the executor at a start) which leaves
   a strict trigger at its threshold, e.g. its own trigger `Y > 2` by
   `Y = 2`, keeps it true for AMICI or makes AMICI fire it. After every
   event the executor fires, the start is evaluated again (an event which
   fired at the start does not fire again there); in a model with a strict
   trigger every execution of an event by AMICI is located and the model
   integrated again to just after it (`_Course._run_once`): a strict
   trigger at its threshold there raises a `NotImplementedError` naming the
   events. A horizon of a steady state does not fire an
   event again which AMICI fired at the end of the horizon before it, at
   its root of 0 (`_Course.horizon`): the time is continuous across the
   horizons.
6. AMICI outputs the states at given times only. The output of
   `OutputMode.TIMES` are the plan times of a segment. The steps of the
   integrator (`OutputMode.INTEGRATOR`) are a substitute: `STEPS` intervals
   of the same length per segment, and every event of the model AMICI fires
   as two rows at its time, the states at the resolution of the time before
   and after it (`_Course.steps`), whose times AMICI records
   (`EVENT_TIME_PREFIX`, an interval in which an event fires more than once
   split until it fires once in every part); the rows at the changes, at the
   events and at the end correspond to the ones of roadrunner. An event at
   the end has the state `AFTER_THE_EVENT` units in the last place after it
   as the state after it.

Known differences to roadrunner:

- roadrunner evaluates the triggers of the events when the model is
  initialized, at the time 0, also for a simulation from a negative start;
  AMICI evaluates them at the start.
- roadrunner integrates a model which does not read the time in local time,
  every segment from its time 0; AMICI integrates every model in absolute
  time.
- The steps of the integrator are a grid with the events, see above; the
  steps of CVODES in AMICI are not exposed.
- AMICI has one absolute tolerance for every state, a concentration species
  in its concentration, see `AmiciSBMLModel.set_integrator_settings`; the
  steps differ between the engines, the rows at the changes, at the events
  and at the end correspond.
- The rows of an event of the model in the steps are the states at the
  resolution of the time before and after it, the ones of roadrunner a
  little before and after it, at a distance which depends on the last step
  of CVODE; both differ from the states at the event by the rates times
  that distance.
- Simultaneous events of the same priority which AMICI fires in an
  integration execute in the order of AMICI, the ones the executor fires in
  the order of the document, roadrunner orders them at random.
- roadrunner fires a strict trigger of the time at the end of a horizon of
  a steady state (`time > 10` at 10) and again in the next horizon, so
  `time > 10` fires twice in a presimulation and `time > 1` (or
  `time > 1.0000000001`) three times in a presimulation and a simulation
  to 1; AMICI fires them once per crossing of the time (1 and 2, as
  COPASI). AMICI fires a strict trigger at its root, e.g. `time > 100` at
  the end 100 of a steady state which is reached there, where COPASI does
  not (roadrunner does as well), and an output time at the root of a
  strict trigger of the time (`time > 1` at 1) has the state after the
  event in AMICI and the state before it in roadrunner.
- AMICI 1.2 does not import a trigger with a negation of a comparison
  (`!(X <= 3)`), the model raises when it is loaded.

The absolute tolerance of AMICI is the one of roadrunner times
`sbmlsim.model.model_amici.AMICI_ATOL_FACTOR`, so AMICI integrates the HCTZ
model at least as accurately as roadrunner at the defaults of `Simulator`
(the largest relative error 1.1e-7 against 1.5e-7 of roadrunner, see the
measurement there).

The delays of events, the `delay` csymbol and the other features AMICI does
not simulate as roadrunner are refused when the model is loaded, see
`AmiciSBMLModel`.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence

import libsbml
import numpy as np

from sbmlsim.model.model_amici import AmiciSBMLModel, run_simulation
from sbmlsim.result.timecourse import TimecourseResult
from sbmlsim.simulator.changes import fire_events, triggers
from sbmlsim.simulator.executor import integrable
from sbmlsim.simulator.plan import (
    TIME_RTOL,
    Plan,
    SteadyStatePlan,
    preinit_targets,
    time_tolerance,
)
from sbmlsim.simulator.segments import simulate_segments, state_row
from sbmlsim.simulator.segments import steady_state as integrate_to_steady_state

#: the intervals of the grid of a segment of the substitute of the steps of
#: the integrator, see the module
STEPS = 100

#: the units in the last place of the time after the last event at the end
#: of the state after it, more than CVODES varies the time of a root between
#: two integrations, see `_Course.event_at_the_end`
AFTER_THE_EVENT = 1024

#: the most times the model is integrated ahead of a start to tell whether a
#: trigger at its threshold becomes true, the resolution of the time and then
#: ten times as far every time, see `_Course.restart`
LOOK_AHEAD = 7

#: the most times the intervals of the grid of the steps are split to tell
#: the times of the events which fire more than once in one, see
#: `_Course.steps`
MAX_SPLITS = 20

#: the relational operators of a trigger at a root of the value 0 as AMICI
#: evaluates them, see `_relaxed`
_RELAXED = {
    libsbml.AST_RELATIONAL_GT: libsbml.AST_RELATIONAL_GEQ,
    libsbml.AST_RELATIONAL_LT: libsbml.AST_RELATIONAL_LEQ,
}

#: the negation of a relational operator
_NEGATED = {
    libsbml.AST_RELATIONAL_GT: libsbml.AST_RELATIONAL_LEQ,
    libsbml.AST_RELATIONAL_GEQ: libsbml.AST_RELATIONAL_LT,
    libsbml.AST_RELATIONAL_LT: libsbml.AST_RELATIONAL_GEQ,
    libsbml.AST_RELATIONAL_LEQ: libsbml.AST_RELATIONAL_GT,
    libsbml.AST_RELATIONAL_EQ: libsbml.AST_RELATIONAL_NEQ,
    libsbml.AST_RELATIONAL_NEQ: libsbml.AST_RELATIONAL_EQ,
}


class _Course:
    """The integrations of a model of AMICI which continue one another.

    Attributes:
        model: the model.
        selections: the selections of the output.
        resolution: the resolution of the time of the simulation.
    """

    def __init__(
        self, model: AmiciSBMLModel, selections: Sequence[str], resolution: float
    ) -> None:
        """Set up the integrations of a model, see `sbmlsim.simulator.segments.Course`."""
        self.model = model
        self.selections = list(selections)
        self.resolution = resolution if resolution > 0.0 else TIME_RTOL
        self._transform = model.output_transform(self.selections)
        self._run = run_simulation()
        self.events = model.symbols.events
        if len(self.events) != len(model.event_ids):
            raise RuntimeError(
                f"AMICI has {len(model.event_ids)} events of the model "
                f"'{model.label}', not {len(self.events)}."
            )
        self._relaxed: list[str | None] | None = None
        self._strict: bool | None = None
        #: the events whose trigger at its threshold at the last start becomes
        #: true later, by their times, which the integrations from the start
        #: fire (`_integrate`)
        self._pending: list[tuple[float, int]] = []
        #: the states before and after the events of `_pending` the last
        #: integration fired, by their times
        self._fired: list[tuple[float, np.ndarray, np.ndarray]] = []
        #: the end of the last horizon of a steady state and the counters of
        #: the events at its start, see `horizon`
        self._horizon: tuple[float, np.ndarray] | None = None

    def start_events(self, time: float) -> None:
        """Fire the events whose trigger is false before the time 0 and true at the start.

        roadrunner fires them at the start in the initial state, with the
        events they trigger in the order of their priorities, before a change
        at the start; AMICI fires no event at the start of an integration.

        Args:
            time: the time of the start.
        """
        if not self.events:
            return
        model = self.model
        model.set_time(time)
        before: list[bool | None] = [
            eid not in model.initial_value_false for eid in model.event_ids
        ]
        fire_events(self.events, before, model.values, time, model)

    def restart(
        self, time: float, end: float, executed: Sequence[int] = ()
    ) -> np.ndarray | None:
        """Fire the events whose trigger becomes true after a time, which AMICI misses.

        AMICI evaluates a trigger at the start of an integration with its
        comparisons at a root of the value 0 as true, also a strict one
        (`_relaxed`), and misses the event when the trigger becomes true. For
        an event whose trigger is false at the time but true for AMICI, the
        model is integrated ahead, the resolution of the time first and then
        ten times as far, at most `LOOK_AHEAD` times, and not beyond the end:

        - if its trigger is true at the resolution after the time, the event
          fires at the time, after its row and with the events it triggers,
          as in roadrunner, which fires it in its first step; the state after
          the events starts the integration;
        - if its trigger is true at a later time ahead, the time it becomes
          true is located by bisection to the resolution of the time, and the
          integrations from the time fire the event there (`_integrate`);
        - if its trigger as AMICI evaluates it is false ahead, AMICI finds the
          crossing after it, and if it is still at its threshold at the end,
          the next start evaluates it again.

        Args:
            time: the time of the current state, after the changes and the
                events of the time.
            end: the end of the integration which starts at the time.
            executed: the events AMICI fired at the time at the end of the
                integration before it, which are not fired again, see
                `horizon`.

        Returns:
            The values of the selections before the events which fire at the
            time, `None` if none fires.

        Raises:
            NotImplementedError: for a trigger still at its threshold after
                `LOOK_AHEAD` steps ahead before the end.
        """
        model = self.model
        model.set_time(time)
        self._pending = []
        events = self.events
        if not events:
            return None
        values = model.values
        head: np.ndarray | None = None
        # the events which fired at the time, which do not fire again at it:
        # their triggers are true right after it, as for AMICI
        fired: set[int] = set()
        while True:
            strict = triggers(events, values, time, model)
            relaxed = self._relaxed_triggers(time)
            zero = [
                k
                for k in range(len(events))
                if strict[k] is False and relaxed[k] is True and k not in executed
            ]
            fire, pending = self._look_ahead(time, end, zero) if zero else ([], [])
            self._pending = sorted(pending)
            fire = [k for k in fire if k not in fired]
            if not fire:
                return head
            if head is None:
                head = state_row(model, self.selections, time)
            after = list(strict)
            for k in fire:
                after[k] = True
            fire_events(events, strict, values, time, model, after=after)
            model.set_time(time)
            fired.update(fire)

    def _look_ahead(
        self, time: float, end: float, zero: Sequence[int]
    ) -> tuple[list[int], list[tuple[float, int]]]:
        """Tell the events whose triggers at their threshold become true right after a time or later.

        See `restart`.

        Returns:
            The events whose triggers are true at the resolution after the
            time, and the times the triggers of the others become true at,
            with their events.

        Raises:
            NotImplementedError: for a trigger still at its threshold after
                `LOOK_AHEAD` steps ahead before the end.
        """
        fire: list[int] = []
        pending: list[tuple[float, int]] = []
        resolution = max(self.resolution, TIME_RTOL * abs(time))
        ahead, previous = resolution, time
        zero = list(zero)
        for _ in range(LOOK_AHEAD):
            later = min(time + ahead, end)
            if not integrable(time, later):
                return fire, pending
            strict_later, relaxed_later = self._triggers_at(time, later)
            for k in zero:
                if strict_later[k] is not True:
                    continue
                if previous == time:
                    fire.append(k)
                    continue
                crossing = self._crossing(time, previous, later, k, resolution)
                if crossing is not None:
                    pending.append((crossing, k))
            zero = [
                k
                for k in zero
                if strict_later[k] is not True and relaxed_later[k] is True
            ]
            if not zero or later >= end:
                return fire, pending
            previous = later
            ahead *= 10.0
        raise NotImplementedError(
            f"The engine AMICI does not simulate the events "
            f"{[self.events[k].eid for k in zero]} of the model "
            f"'{self.model.label}' at the time {time}: AMICI evaluates their "
            f"triggers at their threshold as true, which stay at it longer than "
            f"{ahead / 10.0:.3g} after the time."
        )

    def _crossing(
        self, time: float, lo: float, hi: float, k: int, resolution: float
    ) -> float | None:
        """Locate the time a trigger at its threshold at a time becomes true.

        Bisection between a time ahead at which it is still at its threshold
        and one at which it is true, to the resolution of the time.

        Returns:
            The first time at which it is true, within the resolution, `None`
            if its trigger as AMICI evaluates it becomes false between, so
            that AMICI finds the crossing.
        """
        while hi - lo > resolution:
            middle = lo + (hi - lo) / 2.0
            strict, relaxed = self._triggers_at(time, middle)
            if strict[k] is True:
                hi = middle
            elif relaxed[k] is not True:
                return None
            else:
                lo = middle
        return hi

    def _relaxed_triggers(self, time: float) -> list[bool | None]:
        """Evaluate the triggers as AMICI at the start of an integration, see `_relaxed`."""
        # sbmlmath is imported by the first model with events, see `sbml_math`
        from sbmlsim.sbml_math import compile_math

        if self._relaxed is None:
            self._relaxed = [
                None if event.trigger is None else _relaxed(event.trigger)
                for event in self.events
            ]
        values = self.model.values
        result: list[bool | None] = []
        for mathml in self._relaxed:
            if mathml is None:
                result.append(None)
                continue
            try:
                formula = compile_math(mathml)
            except ValueError:
                result.append(None)
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

    def _triggers_at(
        self, time: float, later: float
    ) -> tuple[list[bool | None], list[bool | None]]:
        """Evaluate the triggers at a later time, and as AMICI evaluates them.

        The model is integrated to it and set back to its state at the time.
        """
        model = self.model
        state = model.state
        self._integrate(time, np.array([later]))
        result = (
            triggers(self.events, model.values, later, model),
            self._relaxed_triggers(later),
        )
        model.set_state(time, state)
        return result

    def horizon(self, start: float, end: float) -> None:
        """Integrate from `start` to `end` without output, see `sbmlsim.simulator.segments.Course`.

        A horizon starts with the events AMICI misses at their threshold
        (`restart`), but not with an event AMICI fired at the end of the
        horizon before it, at its root of 0, e.g. `time > 10` at the end 10:
        the time is continuous across the horizons and the trigger becomes
        true once.
        """
        model = self.model
        executed: list[int] = []
        if self._horizon is not None and self._horizon[0] == start:
            counts = model.event_counts(model.state)
            times = model.event_times(model.state)
            executed = [
                k
                for k in range(counts.size)
                if counts[k] > self._horizon[1][k] and times[k] == start
            ]
        self.restart(start, end, executed)
        self._horizon = (end, model.event_counts(model.state))
        self._integrate(start, np.array([end]))

    def rates(self, states: Sequence[str]) -> np.ndarray:
        """Get the current rates of change of states, see `AmiciSBMLModel.rates`."""
        return self.model.rates(states)

    def _integrate(
        self, start: float, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate from the current state at `start` with output at times, firing the pending events.

        An event of `_pending` between the start and the last time fires at
        its time: the integration is split there, an output time at it has
        the state after it, and the states before and after it are in
        `_fired`. The model is in the state at the last time after it.

        Args:
            start: the start.
            times: the output times, sorted, from the start on.

        Returns:
            The values of the selections and the states of AMICI, a row at
            the start and a row per time.

        Raises:
            RuntimeError: if AMICI fails to integrate, with its messages.
        """
        times = np.asarray(times, dtype=float)
        self._fired = []
        last = float(times[-1]) if times.size else start
        due = [(t, k) for t, k in self._pending if start < t <= last]
        if not due:
            return self._run_once(start, times)
        crossing = due[0][0]
        values, x = self._run_once(start, np.append(times[times < crossing], crossing))
        before = values[-1].copy()
        self._fire_pending(crossing, [k for t, k in due if t == crossing])
        self._check_after_event(crossing, crossing, self.model.state)
        after = state_row(self.model, self.selections, crossing)
        pending = self._pending
        self._pending = [(t, k) for t, k in pending if t > crossing]
        try:
            rest, y = self._integrate(crossing, times[times >= crossing])
        finally:
            fired = [(crossing, before, after), *self._fired]
            self._pending = pending
        self._fired = fired
        return np.vstack([values[:-1], rest[1:]]), np.vstack([x[:-1], y[1:]])

    def _fire_pending(self, time: float, pending: Sequence[int]) -> None:
        """Fire events of `_pending` at their time, with the events they trigger.

        Raises:
            RuntimeError: if the trigger of an event is not true at its time,
                e.g. after an event of AMICI before it which `restart` did not
                see; AMICI would miss the event.
        """
        model = self.model
        model.set_time(time)
        values = model.values
        current = triggers(self.events, values, time, model)
        missed = [self.events[k].eid for k in pending if current[k] is not True]
        if missed:
            raise RuntimeError(
                f"The triggers of the events {missed} of the model '{model.label}' "
                f"are not true at the time {time} at which they become true after "
                f"their threshold; AMICI would miss the events."
            )
        before = list(current)
        for k in pending:
            before[k] = False
        fire_events(self.events, before, values, time, model)
        model.set_time(time)

    def _run_once(
        self, start: float, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate from the current state at `start` with output at times, checking the events of AMICI.

        AMICI evaluates a trigger at a root of the value 0 as true also after
        one of its events in an integration: an event which leaves a strict
        trigger exactly at its threshold, e.g. its own trigger `Y > 2` by
        `Y = 2`, keeps it true for AMICI (which misses its next crossing) or
        fires it (which roadrunner does not, while it stays at its
        threshold). For a model with a strict trigger every execution of an
        event by AMICI is located (`_first_execution`), the model is
        integrated again to `AFTER_THE_EVENT` units in the last place after
        it, and the integration continues from there if no strict trigger
        is at its threshold; else it raises. An event closer than that to
        the last time is left to the start of the next integration
        (`restart`).

        Args:
            start: the start.
            times: the output times, sorted, from the start on.

        Returns:
            The values of the selections and the states of AMICI, a row at
            the start and a row per time.

        Raises:
            NotImplementedError: for a strict trigger at its threshold after
                an event of AMICI.
            RuntimeError: if AMICI fails to integrate, with its messages.
        """
        model = self.model
        if not self._strict_triggers():
            return self._run_amici(start, times)
        state = model.state
        initial = model.event_counts(state)
        values, x = self._run_amici(start, times)
        counts = model.event_counts(x)
        if np.array_equal(counts[-1], initial):
            return values, x
        grid = np.concatenate([[start], times])
        if not np.array_equal(counts[0], initial):
            # AMICI fired an event of the time within the rounding of the
            # time at the start, the integration is not repeated from it
            self._check_after_event(start, start, x[0])
        changed = np.flatnonzero(np.any(counts[1:] != counts[:-1], axis=1)) + 1
        if changed.size == 0:
            return values, x
        i = int(changed[0])
        executed, lo, lo_state = self._first_execution(grid[i - 1], x[i - 1], grid[i])
        after = executed + AFTER_THE_EVENT * float(
            np.spacing(max(abs(executed), abs(lo)))
        )
        if after > grid[-1]:
            # an event at the end of the integration, the start of the next
            # one evaluates the triggers, see `restart`
            return values, x
        middle = grid[(grid > lo) & (grid <= after)]
        model.set_state(lo, lo_state)
        between, y = self._run_amici(lo, np.append(middle, after))
        if np.any(model.event_counts(y[-1]) > counts[i]):
            raise RuntimeError(
                f"AMICI fires more events of the model '{model.label}' between "
                f"{lo} and {after} in a repeated integration."
            )
        self._check_after_event(executed, after, y[-1])
        later = times[times > after]
        rest, z = self._run_once(after, later)
        kept = grid.size - later.size - middle.size
        rows = np.vstack([values[:kept], between[1:-1]])
        states = np.vstack([x[:kept], y[1:-1]])
        return np.vstack([rows, rest[1:]]), np.vstack([states, z[1:]])

    def _strict_triggers(self) -> bool:
        """Whether a trigger has a comparison AMICI evaluates otherwise at a root of 0, see `_relaxed`."""
        if self._strict is None:
            self._strict = any(
                event.trigger is not None
                and _relaxed(event.trigger)
                != libsbml.writeMathMLToString(
                    libsbml.readMathMLFromString(event.trigger)
                )
                for event in self.events
            )
        return self._strict

    def _first_execution(
        self, lo: float, lo_state: np.ndarray, hi: float
    ) -> tuple[float, float, np.ndarray]:
        """Locate the first execution of an event by AMICI between two times.

        The interval is split until one event executes in it once (its time
        the time of its last execution, `EVENT_TIME_PREFIX`), or it is
        shorter than the resolution of the time (the first time of the
        executions in it).

        Returns:
            The time of the execution, and the time and the states of AMICI of
            the start of the interval which contains it.
        """
        model = self.model
        resolution = max(self.resolution, TIME_RTOL * abs(hi))
        for _ in range(MAX_SPLITS + 1):
            model.set_state(lo, lo_state)
            grid = np.linspace(lo, hi, 9)[1:]
            _, y = self._run_amici(lo, grid)
            counts = model.event_counts(y)
            j = int(np.flatnonzero(np.any(counts[1:] != counts[:-1], axis=1))[0]) + 1
            executions = counts[j] - counts[j - 1]
            times = np.concatenate([[lo], grid])
            if executions.sum() == 1 or times[j] - times[j - 1] <= resolution:
                executed = float(np.min(model.event_times(y[j])[executions > 0]))
                return executed, float(times[j - 1]), y[j - 1]
            lo, lo_state, hi = float(times[j - 1]), y[j - 1], float(times[j])
        raise RuntimeError(
            f"AMICI fires the events of the model '{self.model.label}' too often "
            f"at the time {lo} to tell their times."
        )

    def _check_after_event(
        self, executed: float, time: float, state: np.ndarray
    ) -> None:
        """Refuse a strict trigger at its threshold after an event of AMICI, see `_run_once`.

        Raises:
            NotImplementedError: naming the events.
        """
        model = self.model
        model.set_state(time, state)
        strict = triggers(self.events, model.values, time, model)
        relaxed = self._relaxed_triggers(time)
        held = [
            event.eid
            for event, s, r in zip(self.events, strict, relaxed, strict=True)
            if s is False and r is True
        ]
        if held:
            raise NotImplementedError(
                f"The engine AMICI does not simulate the events {held} of the model "
                f"'{model.label}' after the event at the time {executed}: their "
                f"triggers stay at their threshold, which AMICI evaluates as true "
                f"in an integration."
            )

    def _run_amici(
        self, start: float, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate from the current state at `start` with output at times with AMICI, see `_run_once`.

        Args:
            start: the start.
            times: the output times, sorted, from the start on.

        Returns:
            The values of the selections and the states of AMICI, a row at
            the start and a row per time.

        Raises:
            RuntimeError: if AMICI fails to integrate, with its messages.
        """
        model = self.model
        grid = np.concatenate([[start], np.asarray(times, dtype=float)])
        model.set_time(start)
        model.prepare(grid)
        amodel = model.amici_model
        rdata = self._run(amodel, model.solver)
        if rdata.status != 0:
            sundials = importlib.import_module("amici.sim.sundials")
            messages = "; ".join(str(m.message) for m in rdata.messages)
            raise RuntimeError(
                f"AMICI failed to integrate the model '{model.label}' from the time "
                f"{start} to {grid[-1]}: "
                f"{sundials.simulation_status_to_str(rdata.status)}, {messages}"
            )
        x = np.array(rdata.x, dtype=float).reshape(grid.size, -1)
        w = (
            np.array(rdata.w, dtype=float).reshape(grid.size, amodel.nw)
            if amodel.nw
            else np.zeros((grid.size, 0))
        )
        values = self._transform(grid, x, w)
        model.set_state(float(grid[-1]), x[-1])
        return values, x

    def times(self, start: float, times: np.ndarray, end: float) -> np.ndarray:
        """Integrate from `start` with output at times, see `sbmlsim.simulator.segments.Course`.

        AMICI outputs at the times exactly and finds the events of the model
        up to the last one, so the integration does not reach the end after
        a last time before it.
        """
        return self._integrate(start, times)[0]

    def steps(self, start: float, end: float) -> np.ndarray:
        """Integrate from `start` to `end` with the substitute of the steps of the integrator.

        The output times are `STEPS` intervals of the same length. An event of
        the model AMICI fires is two rows at its time, the states at the
        resolution of the time before and after it, from a second
        integration with these times; events within twice the resolution are
        one pair, whose rows have the time of the first one. The times of the
        events are the times of their last executions between two output
        times (`EVENT_TIME_PREFIX`); an interval in which an event fires more
        than once is split until it fires once in every part, or the part is
        shorter than the resolution.

        Returns:
            The values, a row per time, the first at the start and the last
            at the end, an event of the model as two rows at its time.

        Raises:
            RuntimeError: if AMICI fails to integrate, or finds an event of the
                second integration outside of its rows.
        """
        model = self.model
        state = model.state
        # AMICI fires an event of the time within the rounding of the time
        # after the start at the start, its row at the start is after it
        initial = model.event_counts(state)
        resolution = self.resolution
        grid = np.linspace(start, end, STEPS + 1)[1:]
        grid[-1] = end
        for _ in range(MAX_SPLITS + 1):
            values, x = self._integrate(start, grid)
            counts = np.diff(np.vstack([initial, model.event_counts(x)]), axis=0)
            if not counts.any():
                return self._with_fired(values)
            # the intervals of the rows, the first one the start
            times = np.concatenate([[start, start], grid])
            split = np.flatnonzero(
                np.any(counts > 1, axis=1) & (np.diff(times) > 2.0 * resolution)
            )
            if split.size == 0:
                break
            grid = np.unique(
                np.concatenate(
                    [grid, *(np.linspace(times[i], times[i + 1], 9) for i in split)]
                )
            )
            grid = grid[grid > start]
            model.set_state(start, state)
        else:
            raise RuntimeError(
                f"AMICI fires the events of the model '{model.label}' too often "
                f"between {start} and {end} to tell their times."
            )
        executed = model.event_times(x)
        executions = sorted(set(executed[counts > 0].tolist()))
        windows: list[list[float]] = []
        for t in executions:
            if windows and t - windows[-1][1] <= 2.0 * resolution:
                windows[-1][1] = t
            else:
                windows.append([t, t])
        bounds = [
            (max(first - resolution, start), min(last + resolution, end))
            for first, last in windows
        ]
        inside = np.zeros(grid.size, dtype=bool)
        for lo, hi in bounds:
            inside |= (grid > lo) & (grid < hi)
        output = np.sort(
            np.concatenate([grid[~inside], [t for pair in bounds for t in pair]])
        )
        model.set_state(start, state)
        head = state_row(model, self.selections, start)
        values, x = self._integrate(start, output)
        times = np.concatenate([[start], output])
        counted = model.event_counts(x).sum(axis=1)
        if np.any(model.event_counts(x[0]) != initial):
            # the state before an event AMICI fired at the start
            values = np.vstack([head, values])
            times = np.concatenate([[start], times])
            counted = np.concatenate([[initial.sum()], counted])
        for (lo, hi), (first, _) in zip(bounds, windows, strict=True):
            # the first row at the time after, the last row at the time before
            # which is before the event
            after = int(np.flatnonzero(times == hi)[0])
            before = [
                i for i in np.flatnonzero(times == lo) if counted[i] < counted[after]
            ]
            if not before:
                raise RuntimeError(
                    f"AMICI found the event of the model '{model.label}' at the "
                    f"time {first} outside of its rows."
                )
            values[[before[-1], after], 0] = first
            if before[-1] < after - 1:
                # the rows between are the state at the start after the event
                values = np.delete(values, range(before[-1] + 1, after), axis=0)
                times = np.delete(times, range(before[-1] + 1, after))
                counted = np.delete(counted, range(before[-1] + 1, after))
        return self._with_fired(values)

    def _with_fired(self, values: np.ndarray) -> np.ndarray:
        """Insert the states before and after the pending events the last integration fired, at their times."""
        for time, before, after in self._fired:
            index = int(np.searchsorted(values[:, 0], time, side="left"))
            values = np.insert(values, index, [before, after], axis=0)
            values[index : index + 2, 0] = time
        return values

    def event_at_the_end(
        self, start: float, end: float, tolerance: float
    ) -> np.ndarray | None:
        """Get the state after the events of the model at the end, see `sbmlsim.simulator.segments.Course`.

        The events right after the start fire first (`restart`), the state
        after them, or after the events the executor fires later
        (`_integrate`), is the state after the events if AMICI fires none. The
        model is integrated from `start` to the resolution after the end; an
        event AMICI fires is counted (`COUNT_PREFIX`). The state after it is
        the one `AFTER_THE_EVENT` units in the last place after the last
        execution of an event (`EVENT_TIME_PREFIX`), from a second
        integration, if AMICI fires every event before it again, else, or for
        an event closer than that before the resolution after the end, the
        one at the resolution after the end.
        """
        model = self.model
        stop = end + tolerance
        fired = self.restart(start, stop) is not None
        after_start = state_row(model, self.selections, start) if fired else None
        # AMICI fires an event of the time within the rounding of the time
        # after the start at the start
        state = model.state
        initial = model.event_counts(state)
        values, x = self._integrate(start, np.array([stop]))
        counts = model.event_counts(x[-1]) - initial
        if not counts.any():
            # the state after the events the executor fired
            return self._fired[-1][2].copy() if self._fired else after_start
        last = float(np.max(model.event_times(x[-1])[counts > 0]))
        after = last + AFTER_THE_EVENT * float(np.spacing(max(abs(last), abs(start))))
        if after < stop:
            at_the_end = x[-1]
            model.set_state(start, state)
            exact, y = self._integrate(start, np.array([after]))
            if np.array_equal(
                model.event_counts(y[-1]), model.event_counts(at_the_end)
            ):
                return exact[-1].copy()
            model.set_state(stop, at_the_end)
        return values[-1].copy()

    def steady_state(self, plan: SteadyStatePlan, start: float) -> float:
        """Integrate the model from a time until its rates of change vanish, see `steady_state`."""
        return steady_state(self.model, plan, start=start)


def _relaxed(mathml: str) -> str:
    """Get the trigger as AMICI evaluates it at the start of an integration.

    AMICI evaluates a trigger by its root function (`max` of an `or`, `min`
    of an `and`, the difference of a comparison) and sets it true where its
    value is 0: a strict comparison is the one which is not strict, a
    negation is moved into the comparisons, `!=` is true.

    Args:
        mathml: the MathML of the trigger.

    Returns:
        The MathML of the trigger as AMICI evaluates it.
    """
    ast = libsbml.readMathMLFromString(mathml)
    return libsbml.writeMathMLToString(_relaxed_node(ast, negate=False))


def _relaxed_node(node: libsbml.ASTNode, negate: bool) -> libsbml.ASTNode:
    """Get a node of a trigger as AMICI evaluates it, see `_relaxed`."""
    kind = node.getType()
    if kind == libsbml.AST_LOGICAL_NOT and node.getNumChildren() == 1:
        return _relaxed_node(node.getChild(0), not negate)
    if kind in (libsbml.AST_LOGICAL_AND, libsbml.AST_LOGICAL_OR):
        if negate:
            kind = (
                libsbml.AST_LOGICAL_OR
                if kind == libsbml.AST_LOGICAL_AND
                else libsbml.AST_LOGICAL_AND
            )
        result = libsbml.ASTNode(kind)
        for k in range(node.getNumChildren()):
            result.addChild(_relaxed_node(node.getChild(k), negate))
        return result
    if kind in _NEGATED and node.getNumChildren() == 2:
        if negate:
            kind = _NEGATED[kind]
        if kind == libsbml.AST_RELATIONAL_NEQ:
            return libsbml.ASTNode(libsbml.AST_CONSTANT_TRUE)
        result = libsbml.ASTNode(_RELAXED.get(kind, kind))
        for k in range(node.getNumChildren()):
            result.addChild(node.getChild(k).deepCopy())
        return result
    if kind in (libsbml.AST_CONSTANT_TRUE, libsbml.AST_CONSTANT_FALSE) and negate:
        return libsbml.ASTNode(
            libsbml.AST_CONSTANT_FALSE
            if kind == libsbml.AST_CONSTANT_TRUE
            else libsbml.AST_CONSTANT_TRUE
        )
    copy = node.deepCopy()
    if not negate:
        return copy
    result = libsbml.ASTNode(libsbml.AST_LOGICAL_NOT)
    result.addChild(copy)
    return result


def execute(
    plan: Plan, model: AmiciSBMLModel, selections: Sequence[str]
) -> TimecourseResult:
    """Run a plan on a model of AMICI.

    The contract of `sbmlsim.simulator.executor.execute`: the output has two
    rows at the time of a change after the start, the state before the
    change and the state after it, and the steps of the integrator have two
    at an event of the model, see the module.

    Args:
        plan: the plan, compiled against the model.
        model: the model.
        selections: the columns of the result, `time` is added as the first
            column if it is missing.

    Returns:
        The result, with the time shifted by `plan.time_shift`.

    Raises:
        SteadyStateError: if the presimulation does not reach a steady state.
        RuntimeError: if AMICI fails to integrate, with its message.
        NotImplementedError: for a plan AMICI does not simulate, see
            `check_supported`.
    """
    check_supported(plan, model)
    columns = list(selections)
    if not columns or columns[0] != "time":
        columns = ["time", *columns]
    model.initialize(preinit_targets(plan))
    try:
        course = _Course(model, columns, time_tolerance(plan.start, plan.end))
        if plan.steady_state is not None:
            # the horizons of the steady state start with the events AMICI
            # misses at their threshold, see `_Course.restart`
            course.start_events(0.0)
            steady_state(model, plan.steady_state)
        else:
            course.start_events(plan.start)
        values = simulate_segments(plan, model, model.values, columns, course)
    finally:
        model.simulated()
    if plan.time_shift != 0.0:
        values[:, 0] += plan.time_shift
    return TimecourseResult(columns=tuple(columns), values=values)


def check_supported(plan: Plan, model: AmiciSBMLModel) -> None:
    """Refuse a plan which sets a compartment AMICI keeps out of its states.

    `execute` checks every plan, and `Simulator.run` the plan of the first
    point of every simulation and model before it runs a point, see
    `sbmlsim.simulator.engines.check_plan`. A compartment whose size in the
    SBML is not a number and which has no initial assignment is a number of
    AMICI, which integrates no state which is not finite, see
    `sbmlsim.model.model_amici`.

    Raises:
        NotImplementedError: for a plan which sets such a compartment.
    """
    assignments = [
        *plan.preinit,
        *(plan.steady_state.preinit if plan.steady_state is not None else ()),
        *(a for event in plan.events for a in event.assignments),
    ]
    symbols = plan.symbols
    outside = sorted(
        {symbols.entity(a.target) for a in assignments} & model.not_a_number
    )
    if outside:
        raise NotImplementedError(
            f"The engine AMICI does not simulate a change of {outside} of the model "
            f"'{model.label}': its value in the SBML is not a number, which AMICI "
            f"keeps out of its states."
        )


def steady_state(
    model: AmiciSBMLModel, plan: SteadyStatePlan, start: float = 0.0
) -> float:
    """Integrate a model until its rates of change vanish, see `sbmlsim.simulator.segments.steady_state`.

    Every horizon starts with the events whose trigger AMICI evaluates as
    true at their threshold, see `_Course.restart`.

    Args:
        model: the model in the state the integration starts from.
        plan: the steady state.
        start: the time the integration starts at.

    Returns:
        The time the steady state was reached at.

    Raises:
        SteadyStateError: if the steady state is not reached by `max_time`.
        RuntimeError: if AMICI fails to integrate, with its message.
    """
    return integrate_to_steady_state(model, _Course(model, [], 0.0), plan, start)
