"""The executor of COPASI runs a plan on a model of COPASI.

It has the semantics of the executor of roadrunner, see
`sbmlsim.simulator.executor`, so the results of both engines are
comparable:

1. The model is initialized with the pre-initialization values, see
   `CopasiSBMLModel.initialize`, which sets the initial state as loaded back
   first, so nothing of an earlier simulation is left, also no event of the
   model which was pending at its end.
2. A presimulation integrates from the time 0 until the rates of change
   vanish, see `steady_state`.
3. The interval is split at the times of the changes. At the start of a
   segment the changes of its time are applied, every value evaluated with
   the state at that time before any target is set, see
   `sbmlsim.simulator.changes.apply_changes`; after a change after the start
   (or at the start after a presimulation) the events whose triggers the
   change made true fire, which COPASI does not see, see
   `sbmlsim.simulator.changes.fire_events`. At the start COPASI evaluates the
   triggers of the events in the initial state, as roadrunner after its
   reset, so a change at the start does not fire an event.
4. COPASI integrates (LSODA) in absolute time: the first segment from the
   state the model is initialized to, every later one continues from the
   state the one before it ended in (`processRaw(False)`); the states of the
   triggers are set from the current values before every run
   (`resetEventsFound`), since COPASI does not see the values the executor
   sets between two runs, see `check_supported` for the delayed events; a
   delayed event COPASI has scheduled stays pending across the runs, and
   one closer after the start of a run than LSODA integrates, e.g. at the
   end after the stop before it, fires at its time from the state at the
   start (`_TimeCourse.steps`). The output of `OutputMode.TIMES` is the plan
   times of the segment, of `OutputMode.INTEGRATOR` the steps of LSODA,
   which have an event of the model as two rows at its time, the states
   before and after it. A change after the start, the one at the end included, has two rows
   at its time, the state before the change, which ends the segment before
   it, and the state after it, which starts the next one.
5. A segment of a model with events stops just before its end, see
   `sbmlsim.simulator.executor.segment_stop`, so an event of the model at the
   time of a change fires after the change; a segment too short for LSODA,
   whose condition is the one of CVODE, is not integrated, its state is the
   one at its start, see `sbmlsim.simulator.executor.integrable`.
6. An event of the model within the resolution of the time after the end is
   at the end, see `_TimeCourse.event_at_the_end`: the steps of the
   integrator have it as two rows, an output time at the end the state after
   it. The steady state after the end starts after it.

Steps 3, 5 and 6 are the segments of `sbmlsim.simulator.segments`, which the
executor of AMICI shares.

Known differences to roadrunner:

- roadrunner evaluates the triggers of the events when the model is
  initialized, at the time 0, also for a simulation from a negative start;
  COPASI evaluates them at the start.
- roadrunner fires a delayed event of the model late in the steps of the
  integrator, at the end of the next step of CVODE (roadrunner #1382);
  COPASI at its time.
- roadrunner keeps an event pending after the steps of the integrator found
  its root after the end, which the executor of roadrunner applies after the
  run (`apply_pending_step_event`); COPASI keeps none.
- roadrunner integrates a model which does not read the time in local time,
  every segment from its time 0; COPASI integrates every model in absolute
  time.
- An event whose trigger becomes true within the rounding of the state one
  unit in the last place before the end (e.g. `X >= 8` of `X' = 1` from 2 at
  the end 6) fires before it in roadrunner, whose steps have one row at the
  end, the state after it; COPASI fires it at the end, its steps have the
  states before and after it at the end.
- A model with a delayed event (an SBML `delay` of an event) is refused in
  a simulation with a change after the start or a presimulation
  (`NotImplementedError`): COPASI keeps a delayed event pending across the
  values the executor sets between two runs, and setting the states of the
  triggers from them brings back an event whose trigger a change made
  false, or loses one which a crossing after the change schedules, which
  roadrunner cancels and schedules. Without such writes COPASI simulates
  the delayed events itself.
- roadrunner does not fire an event whose trigger, e.g. `X > 0.5 && X < 0.8`,
  is true only within one step of CVODE, which finds the roots of both
  comparisons in it; COPASI fires it.
- The steps of LSODA in COPASI can fail on a delayed event whose time LSODA
  has stepped past when it becomes its critical time (`TCRIT behind TCUR`),
  which raises a `RuntimeError` with the message of COPASI; an output of
  times integrates such a model.
- COPASI integrates with LSODA and one absolute tolerance for every state,
  see `CopasiSBMLModel.set_integrator_settings`; the steps of the integrator
  differ between the engines, the rows at the changes, at the events and at
  the end correspond.

The `delay` csymbol is refused when the model is loaded, see
`CopasiSBMLModel`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from sbmlsim.model.event_rows import EPSILON
from sbmlsim.model.model_copasi import COPASI, CopasiSBMLModel, drain_messages
from sbmlsim.result.timecourse import TimecourseResult
from sbmlsim.simulator.executor import (
    integrable,
    intermediate_time,
)
from sbmlsim.simulator.plan import Plan, SteadyStatePlan, preinit_targets
from sbmlsim.simulator.segments import simulate_segments, state_row
from sbmlsim.simulator.segments import steady_state as integrate_to_steady_state

#: the output of a run of COPASI into a data handler, without the report of
#: the task, which has no file and of which COPASI warns otherwise; SWIG sets
#: the flag on the class when COPASI is imported
OUTPUT = COPASI.CCopasiTask.OUTPUT_DURING  # ty: ignore[unresolved-attribute]


class _TimeCourse:
    """The time course task of COPASI of a model, with an output of selections.

    Attributes:
        model: the model.
        names: the common names of the values of the selections, see
            `CopasiSBMLModel.output_cns`.
    """

    def __init__(self, model: CopasiSBMLModel, selections: Sequence[str]) -> None:
        """Set up the task of a model for runs which continue one another."""
        self.model = model
        self.selections = list(selections)
        self.names = model.output_cns(selections)
        self._transform = model.output_transform(selections)
        self._task: COPASI.CTrajectoryTask = model.dm.getTask("Time-Course")
        # the model of COPASI keeps its initial state, a run integrates the
        # current state of the math container
        self._task.setUpdateModel(False)
        self._problem: COPASI.CTrajectoryProblem = self._task.getProblem()
        self._problem.setTimeSeriesRequested(False)
        self._problem.setStartInSteadyState(False)
        self._problem.setOutputStartTime(0.0)

    def steps(self, start: float, end: float) -> np.ndarray:
        """Integrate from `start` to `end` with the output of the steps of LSODA.

        A delayed event of the model which COPASI has scheduled closer after
        the start than LSODA integrates (`TOUT too close to T`, the condition
        of `integrable`), e.g. at the end after the stop before it, fires at
        its time with the state at the start, as the events at the end of a
        segment which is not integrated; the integration continues from it.

        Returns:
            The values, a row per step, the first at the start and the last
            at the end, an event of the model as two rows at its time.
        """
        problem = self._problem
        problem.setAutomaticStepSize(True)
        problem.setOutputEvent(True)
        problem.setUseValues(False)
        model = self.model
        blocks: list[np.ndarray] = []
        pending = model.pending_event_time()
        while start < pending <= end and not integrable(start, pending):
            # the state at the start, before the event at its time
            blocks.append(state_row(model, self.selections, start)[np.newaxis, :])
            blocks.append(state_row(model, self.selections, pending)[np.newaxis, :])
            model.fire_pending_events(pending)
            start = pending
            pending = model.pending_event_time()
        if not blocks:
            return self._run(start, end)
        if integrable(start, end):
            blocks.append(self._run(start, end))
        else:
            # the state after the events, at the start and at the end
            blocks += [
                state_row(model, self.selections, t)[np.newaxis, :]
                for t in sorted({start, end})
            ]
        return np.concatenate(blocks, axis=0)

    def times(self, start: float, times: np.ndarray, end: float) -> np.ndarray:
        """Integrate from `start` with output at times, the last one the stop.

        COPASI integrates every output time from the time before it, to
        `t + (next - t)`, which can round up by one unit in the last place:
        a last time just before the end of the segment, the stop of a model
        with events, see `sbmlsim.simulator.executor.segment_stop`, may round
        to the end, where an event of the model at the time of the change
        fires before the change. Such a last interval gets a time between,
        from which the sum is exact, see
        `sbmlsim.simulator.executor.intermediate_time`; its row is not
        returned.

        Args:
            start: the time of the current state.
            times: the output times, sorted, distinct, after the start.
            end: the end of the segment, which the integration must not
                reach if the last time is before it.

        Returns:
            The values, a row at the start and a row per time, at the times.

        Raises:
            RuntimeError: if COPASI does not give the rows at the times.
        """
        grid = [float(t) for t in times]
        previous = grid[-2] if len(grid) > 1 else start
        between = intermediate_time(previous, grid[-1], end)
        if between is not None:
            grid.insert(len(grid) - 1, between)
        problem = self._problem
        problem.setAutomaticStepSize(False)
        problem.setOutputEvent(False)
        # the overload of a vector keeps six digits of a time, the text is
        # read exactly and independent of the locale
        problem.setValues(" ".join(repr(t) for t in grid))
        problem.setUseValues(True)
        values = self._run(start, grid[-1], past=True)
        if between is not None:
            values = np.delete(values, values.shape[0] - 2, axis=0)
        # the first column is the time, a row is at its time within the
        # rounding of the sum
        expected = np.concatenate([[start], times])
        rounding = 4.0 * EPSILON * float(np.max(np.abs(expected)))
        if values.shape[0] != expected.size or not np.allclose(
            values[:, 0], expected, rtol=0.0, atol=rounding
        ):
            raise RuntimeError(
                f"COPASI did not give the output times {times.tolist()} of the "
                f"model '{self.model.label}' from the time {start}."
            )
        values[:, 0] = expected
        return values

    def event_at_the_end(
        self, start: float, end: float, tolerance: float
    ) -> np.ndarray | None:
        """Get the state after the events of the model at the end of a simulation.

        The model is integrated with the steps of the integrator from
        `start` to the resolution after the end, which have an event as two
        rows at its time, see `sbmlsim.simulator.segments.Course`.

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
        block = self.steps(start, end + tolerance)
        pairs = np.flatnonzero(block[1:, 0] == block[:-1, 0])
        if pairs.size == 0:
            return None
        return block[pairs[-1] + 1].copy()

    def horizon(self, start: float, end: float) -> None:
        """Integrate from `start` to `end` with the steps of LSODA, without output."""
        self.steps(start, end)

    def rates(self, states: Sequence[str]) -> np.ndarray:
        """Get the current rates of change of states, see `CopasiSBMLModel.rates`."""
        return self.model.rates(states)

    def steady_state(self, plan: SteadyStatePlan, start: float) -> float:
        """Integrate the model from a time until its rates of change vanish, see `steady_state`."""
        return steady_state(self.model, plan, start=start)

    def _run(self, start: float, end: float, past: bool = False) -> np.ndarray:
        """Integrate from `start` to `end` from the current state of the model.

        Args:
            start: the start.
            end: the end.
            past: whether the run may end after the end, see `_duration`.

        Raises:
            RuntimeError: if COPASI fails to integrate, with its message.
        """
        model = self.model
        model.set_time(start)
        # COPASI keeps the states of the triggers of the events from the end of
        # the run before, which a change, an event the executor fired or the
        # time set since then made stale: an event would fire again, or not
        # when its trigger crosses next (the root of COPASI toggles the stale
        # state); the states are set from the current values, without firing.
        # A delayed event COPASI has scheduled stays in its queue, e.g. across
        # the stop before the end into the events at the end and the steady
        # state after the end
        model.mc.resetEventsFound()
        self._problem.setDuration(_duration(start, end, past))
        dm = model.dm
        handler = COPASI.CDataHandler()
        for name in self.names:
            handler.addDuringName(COPASI.CRegisteredCommonName(name))
        # the data model is the output handler of the run, which updates the
        # values of the selections the integration does not need, e.g. an
        # assignment rule which no rate reads, before its interfaces read
        # them; a data handler as the output handler reads them stale
        dm.addInterface(handler)
        try:
            self._task.initializeRawWithOutputHandler(OUTPUT, dm)
            integrated = self._task.processRaw(False)
        except COPASI.CCopasiException:
            integrated = False
        finally:
            self._task.restore(True)
            dm.removeInterface(handler)
        try:
            drain_messages(model.label)
        except ValueError as err:
            raise RuntimeError(str(err)) from None
        if not integrated:
            raise RuntimeError(
                f"COPASI failed to integrate the model '{model.label}' from the "
                f"time {start} to {end}."
            )
        n = handler.getNumRowsDuring()
        rows = np.array(
            [list(handler.getNthRow(k)) for k in range(n)], dtype=float
        ).reshape(n, len(self.names))
        return self._transform(rows)


def _duration(start: float, end: float, past: bool) -> float:
    """Get the duration of COPASI which ends a run that starts at `start` at `end`.

    COPASI ends a run at `start + duration`, which may round to a neighbour of
    `end`: the duration is the one of the neighbours of `end - start` whose
    sum is `end`. A start of a larger magnitude than the end may reach no
    such sum, its rounding of the time is coarser: the duration is then the
    one whose sum is the closest before the end, so that the steps of the
    integrator stop before an event at the end, or after it (`past`), so that
    the run reaches the last output time.
    """
    duration = end - start
    candidates = [duration]
    up = down = duration
    for _ in range(4):
        up = math.nextafter(up, math.inf)
        down = math.nextafter(down, -math.inf)
        candidates += [up, down]
    exact = [d for d in candidates if start + d == end]
    if exact:
        return exact[0]
    if past:
        return min((d for d in candidates if start + d > end), default=duration)
    return max((d for d in candidates if start + d < end), default=duration)


def execute(
    plan: Plan, model: CopasiSBMLModel, selections: Sequence[str]
) -> TimecourseResult:
    """Run a plan on a model of COPASI.

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
        RuntimeError: if COPASI fails to integrate, with its message.
    """
    check_supported(plan, model)
    columns = list(selections)
    if not columns or columns[0] != "time":
        columns = ["time", *columns]
    model.initialize(preinit_targets(plan))
    try:
        if plan.steady_state is not None:
            steady_state(model, plan.steady_state)
        values = simulate_segments(
            plan, model, model.values, columns, _TimeCourse(model, columns)
        )
    finally:
        model.simulated()
    if plan.time_shift != 0.0:
        values[:, 0] += plan.time_shift
    return TimecourseResult(columns=tuple(columns), values=values)


def check_supported(plan: Plan, model: CopasiSBMLModel) -> None:
    """Refuse a plan which writes values while COPASI schedules delayed events.

    `execute` checks every plan, and `Simulator.run` the plan of the first
    point of every simulation and model before it runs a point, see
    `sbmlsim.simulator.engines.check_plan`.

    COPASI keeps the events of the model with a delay pending across the
    runs of a plan, and the states of the triggers are set from the values
    the executor writes between two runs (`resetEventsFound`), which brings
    a pending event back whose trigger a change made false, or loses the
    one a crossing after the change schedules, where roadrunner cancels and
    schedules them. A model with a delayed event is simulated natively in a
    plan without such writes, without a change after the start and without
    a presimulation.

    Raises:
        NotImplementedError: for a model with a delayed event and a plan with
            a change after the start or a presimulation.
    """
    delayed = [event.eid for event in plan.symbols.events if event.delayed]
    if not delayed:
        return
    writes: list[str] = []
    if any(event.time > plan.start for event in plan.events):
        writes.append("a change after the start")
    if plan.steady_state is not None:
        writes.append("a presimulation")
    if writes:
        raise NotImplementedError(
            f"The engine COPASI does not simulate the delayed events {delayed} of "
            f"the model '{model.label}' in a simulation with "
            f"{' and '.join(writes)}: COPASI keeps a delayed event pending "
            f"across the values the executor sets, other than roadrunner."
        )


def steady_state(
    model: CopasiSBMLModel, plan: SteadyStatePlan, start: float = 0.0
) -> float:
    """Integrate a model until its rates of change vanish, see `sbmlsim.simulator.segments.steady_state`.

    Args:
        model: the model in the state the integration starts from.
        plan: the steady state.
        start: the time the integration starts at.

    Returns:
        The time the steady state was reached at.

    Raises:
        SteadyStateError: if the steady state is not reached by `max_time`.
        RuntimeError: if COPASI fails to integrate, with its message.
    """
    return integrate_to_steady_state(model, _TimeCourse(model, []), plan, start)
