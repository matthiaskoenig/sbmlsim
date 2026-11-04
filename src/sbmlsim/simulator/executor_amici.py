"""The executor of AMICI runs a plan on a model of AMICI.

AMICI is the integrator (CVODES) and the root finder, the executor fires the
events of the model: the model AMICI compiles keeps the trigger of every
event and records, instead of its assignments, that the event fired in an
integration and the time of its first execution, see
`sbmlsim.model.model_amici`. An integration from a time runs on the model
without the assignments of its events, so its rows up to the first root of a
trigger are the ones of the model; there the executor applies the semantics
of SBML, see `sbmlsim.simulator.changes.fire_events` (the order of the
priorities, the persistence of the triggers, the values of the assignments at
the trigger or at the execution, the events an assignment triggers, the
events of one time one after the other), and integrates on from the state
after the events (`_Course._advance`), so no row of the output is past an
event without its assignments.

It has the semantics of the executor of roadrunner, see
`sbmlsim.simulator.executor`, so the results of the engines are comparable:

1. The model is initialized with the pre-initialization values, see
   `AmiciSBMLModel.initialize`, from the free parameters as loaded, so
   nothing of an earlier simulation is left.
2. The events whose trigger is false before the time 0
   (`AmiciSBMLModel.start_events`) and true in the initial state fire
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
   change and the events at the end. AMICI integrates in absolute time,
   every segment from the state the one before it ended in.
5. AMICI evaluates a comparison which is exactly at its threshold at the
   start of an integration as written and keeps the value for the whole
   integration, since CVODES reports no root at the start and the root
   function does not change its sign afterwards, see
   `sbmlsim.model.model_amici`: a comparison of a trigger as true, also a
   strict one, e.g. `time > 1` from the time 1 after a change there or
   `X > 2` after a change which sets `X = 2`, and does not find the crossing
   when the comparison becomes true. Every strict comparison of a trigger is
   compiled shifted by a parameter, `a > b + shift`, which is 0 but at the
   start of an integration at which the comparison is exactly at its
   threshold (`_Course._arm`): there the executor integrates the resolution
   of the time ahead (`_Course._probe`) to tell the events whose trigger is
   true right after the start. Such an event which did not fire at the start
   fires there, after the row of the start, as in roadrunner, which fires it
   in its first step (the steps of the integrator have it as two rows at the
   start); one which fired at the start, e.g. `time > 10` at the end of a
   horizon of a steady state, stays true for AMICI and does not fire again.
   Every other comparison at its threshold is shifted by `THRESHOLD_ULPS`
   units in the last place of the largest term of its operands
   (`_Course._shift`), so AMICI finds the crossing when it comes: an event which sets its own trigger back to its threshold,
   `Y > 2` by `Y = 2`, fires again when `Y` rises above 2 later, as in
   roadrunner. A comparison of a math of the model, e.g. the condition of a
   piecewise in a rate, `piecewise(2, time <= tend, 0)`, is compiled shifted
   as well (`AmiciSBMLModel.switches`); before every integration from a
   time at which it is exactly at its threshold, e.g. from the time `tend`
   after an event or a change there, the executor shifts it the same way
   (`_Course._arm_switches`, for a comparison of the time by at least its
   change over the resolution of the time, so the explicit root of AMICI at
   the shifted time is after the start, also `time > tlag` with `tlag = 0`
   at the start 0, and the comparison turns at most about the resolution
   late whatever its scale, `time/86400 > tlag`), so it keeps its value until the operands move by the shift in the
   direction which changes it, `time <= tend` false right after `tend`, as
   roadrunner evaluates it at every step. An output row exactly at the
   threshold of such a comparison, e.g. at `tend`, has the rates and the
   assignment rules evaluated with the comparison as written, the value of
   the SBML, as roadrunner and COPASI have it (`Rate = Ri` at `tend`); AMICI
   evaluates the expressions of the row after the root (`_Course._switch_rows`).
6. AMICI finds the root of a trigger where it crosses its threshold, so the
   event is at its time; a strict trigger of the time is at its time
   exactly. The events of a root within the resolution of the time of a
   grid time are at the grid time, whose state the integration gave. An
   output time at the root has the state before the events there unless the
   trigger is true at the time, every comparison within the tolerance of
   the integrator of its threshold taken at it (`_Course._visible`): a
   comparison which is not strict is true there, a strict one false, so the
   output has the state after `X >= 2` and before `X > 2` at the time `X`
   reaches 2, as in roadrunner, where a strict trigger fires in the first
   step after the time. A root AMICI finds again at the start of the next
   integration within the resolution of the time, the crossing fired for, is
   no event: the integration starts again from it (`_Course._advance`). A
   trigger which an integration makes true within the rounding of its stop,
   without a root reported by AMICI, fires at the stop, where the next
   integration would start with the trigger true.
7. The output of `OutputMode.TIMES` are the plan times of a segment. AMICI
   outputs the states at given times only, so the steps of the integrator
   (`OutputMode.INTEGRATOR`) are a substitute: a grid of `STEPS` intervals
   of the same length per segment, refined where the linear interpolation
   between two rows misses the solution at their midpoint by more than the
   square root of the relative tolerance of the integrator times the
   largest magnitude of the selection in the segment, so the rows
   interpolate as accurately as the steps of roadrunner, whose density
   CVODE chose (`_Course.steps`, `_Course._refine`: `1e-5` of the largest
   value at the default tolerances, `6e-6` on the HCTZ model against `3e-5`
   with the steps of roadrunner, at about twice the rows of roadrunner and
   one integration per round of the refinement and piece between two
   events, three to four times the cost of the grid alone), and every event
   of the model as two rows at its time, the states before and after it;
   the rows at the changes, at the events and at the end correspond to the
   ones of roadrunner.
8. The integration from a time runs to the end of the segment, after an
   event over the distance between the last two events, doubled after every
   integration without an event (`_Course._advance`): the integration past
   the first root, which is discarded, is at most as long as the one before
   it, so the cost of a simulation is linear in the number of its events.

Known differences to roadrunner:

- roadrunner evaluates the triggers of the events when the model is
  initialized, at the time 0, also for a simulation from a negative start;
  AMICI evaluates them at the start.
- An event whose trigger is false before the time 0 and which a
  pre-initialization makes true, e.g. a dose `dose > 0` with `dose` set
  before the initialization, fires at the start in AMICI and COPASI, as in
  libroadrunner after `resetAll` and in the SBML Test Suite; the executor of
  roadrunner does not fire it (#439).
- roadrunner integrates a model which does not read the time in local time,
  every segment from its time 0; AMICI integrates every model in absolute
  time.
- The steps of the integrator are the refined grid with the events, see
  7.; the steps of CVODES in AMICI are not exposed.
- AMICI has one absolute tolerance for every state, a concentration species
  in its concentration, see `AmiciSBMLModel.set_integrator_settings`; the
  steps differ between the engines, the rows at the changes, at the events
  and at the end correspond.
- The rows of an event of the model in the steps are the states before and
  after it at its time, the ones of roadrunner a little before and after
  it, at a distance which depends on the last step of CVODE.
- Simultaneous events of the same priority execute in the order of the
  document, roadrunner orders them at random.
- The output row at the exact time of a root of a strict trigger of the time
  shows the state before the event (the trigger is false at the time, see
  6.); the row of roadrunner depends on where CVODE lands its root, e.g. the
  state before at `time > 0.1`, `time > 0.7` and `time > 12` and the state
  after at `time > 1/3`.
- A trigger which compares for equality, `a == b` or `a != b`, raises when
  the model is loaded, see `sbmlsim.model.model_amici`.
- AMICI 1.2 computes the time of a trigger of the time with a precision of
  about eight digits (`time >= 1 + 1e-8` triggers at 1 in AMICI, with
  `time >= 1`): the executor fires the event when AMICI finds it, after the
  events of the roots before it, so its assignments see the state after
  them, as in roadrunner, at a time a little before the one of roadrunner.
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
import itertools
import logging
import math
import sys
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass

import libsbml
import numpy as np

from sbmlsim.model.comparisons import COMPARISONS, trigger_comparisons
from sbmlsim.model.model_amici import AmiciSBMLModel, run_simulation, vector_array
from sbmlsim.result.timecourse import TimecourseResult
from sbmlsim.simulator.changes import fire_events, fire_start_events, triggers
from sbmlsim.simulator.executor import output_columns, output_time
from sbmlsim.simulator.formula import CompiledFormula
from sbmlsim.simulator.plan import (
    TIME_RTOL,
    Plan,
    SteadyStatePlan,
    preinit_targets,
    time_tolerance,
)
from sbmlsim.simulator.segments import simulate_segments, state_row
from sbmlsim.simulator.segments import steady_state as integrate_to_steady_state

logger = logging.getLogger(__name__)

#: the intervals of the grid of a segment which the substitute of the steps
#: of the integrator starts from, see the module and `_Course.steps`
STEPS = 100

#: the rows of a segment of the substitute of the steps of the integrator at
#: most, see `_Course._refine`
MAX_STEPS = 100_000

#: the units in the last place of the largest term of the operands of a
#: comparison exactly at its threshold by which it is shifted, so that AMICI
#: evaluates it as written at the start of an integration and finds its
#: crossing, see the module and `_Course._shift`
THRESHOLD_ULPS = 2.0

#: the smallest shift, a normal number, for the terms 0
SMALLEST_SHIFT = sys.float_info.min

#: the prefix of the symbols of the tolerances of the comparisons of a
#: trigger at the events, see `_Course._visible`
_TOLERANCE = "__sbmlsim_tolerance_"


@dataclass(frozen=True)
class _Comparison:
    """The compiled operands of a comparison of a trigger or of a math, see `_compile_operands`.

    Attributes:
        left: the left operand.
        right: the right operand.
        terms: the additive terms of the operands, whose magnitudes size the
            shift of the comparison at its threshold, see `_Course._shift`.
        reads_time: whether an operand reads the time.
    """

    left: CompiledFormula
    right: CompiledFormula
    terms: tuple[CompiledFormula, ...]
    reads_time: bool


@dataclass(frozen=True)
class _Point:
    """A row of the output with the state of AMICI it was computed from.

    Attributes:
        time: the time.
        row: the values of the selections.
        state: the states of AMICI.
    """

    time: float
    row: np.ndarray
    state: np.ndarray


@dataclass(frozen=True)
class _Event:
    """Events of the model at one time in the output.

    Attributes:
        time: the time.
        before: the state before the events, with its row.
        after: the state after them, with its row.
        visible: whether an output time at the time has the state after the
            events, i.e. a trigger is true at the time, see the module.
    """

    time: float
    before: _Point
    after: _Point
    visible: bool


class _Course:
    """The integrations of a model of AMICI which continue one another.

    Attributes:
        model: the model.
        selections: the selections of the output.
        resolution: the resolution of the time of the simulation.
        events: the events of the model.
        comparisons: the strict comparisons of their triggers, which AMICI
            compiles shifted, see `AmiciSBMLModel.comparisons`.
        switches: the comparisons of the maths of the model, which AMICI
            compiles shifted, see `AmiciSBMLModel.switches`.
    """

    def __init__(
        self, model: AmiciSBMLModel, selections: Sequence[str], resolution: float
    ) -> None:
        """Set up the integrations of a model, see `sbmlsim.simulator.segments.Course`."""
        self.model = model
        self.selections = list(selections)
        self.resolution = resolution if resolution > 0.0 else TIME_RTOL
        self._transform = model.output_transform(self.selections)
        self._run_simulation = run_simulation()
        self.events = model.symbols.events
        if len(self.events) != len(model.event_ids):
            raise RuntimeError(
                f"AMICI has {len(model.event_ids)} events of the model "
                f"'{model.label}', not {len(self.events)}."
            )
        self.comparisons = model.comparisons
        self.switches = model.switches
        #: the compiled operands of the comparisons and of the switches,
        #: `None` for a comparison whose operand the simulator does not
        #: evaluate
        self._operands: list[_Comparison | None] | None = None
        self._switch_operands: list[_Comparison | None] | None = None
        #: the shifts of the switches as armed last, see `_arm_switches`
        self._armed_switches: list[float] = [0.0] * len(self.switches)
        #: the selections the operands of the switches read, with their
        #: transform, see `_switch_rows`
        self._switch_columns: (
            tuple[list[str], Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]
            | None
        ) = None
        #: the time the executor fired events last and the events
        self._fired: tuple[float, frozenset[int]] = (np.nan, frozenset())
        #: the events fired at that time whose trigger was true at it and
        #: false after their assignments, armed again, see `_fire`
        self._rearmed: tuple[float, frozenset[int]] = (np.nan, frozenset())
        #: the comparisons at their threshold since the executor fired their
        #: events, with the values of their operands, see `_arm`
        self._seen: dict[int, tuple[float, float]] = {}
        #: the triggers tolerant of their thresholds, see `_visible`
        self._tolerant: (
            list[
                tuple[CompiledFormula, list[tuple[CompiledFormula, CompiledFormula]]]
                | None
            ]
            | None
        ) = None

    def start_events(self, time: float) -> None:
        """Fire the events whose trigger is false before the time 0 and true at the start.

        roadrunner fires them at the start in the initial state, with the
        events they trigger in the order of their priorities, before a change
        at the start; AMICI fires no event at the start of an integration.

        Args:
            time: the time of the start.
        """
        model = self.model
        if not any(model.start_events):
            return
        model.set_time(time)
        self._record(
            time,
            fire_start_events(
                self.events, model.start_events, model.values, time, model
            ),
        )

    def times(self, start: float, times: np.ndarray, end: float) -> np.ndarray:
        """Integrate from `start` with output at times, see `sbmlsim.simulator.segments.Course`.

        AMICI outputs at the times exactly and the integration runs to the
        last one, so it does not reach the end after a last time before it.
        An output time at the events of the model has the state after them
        if their trigger is true at the time, else the state before them,
        see the module.
        """
        times = np.asarray(times, dtype=float)
        head = state_row(self.model, self.selections, start)
        rows, events = self._advance(start, float(times[-1]), times)
        if len(rows) != times.size:
            raise RuntimeError(
                f"AMICI gave {len(rows)} rows at the {times.size} output times "
                f"of the model '{self.model.label}' from the time {start}."
            )
        result = np.vstack([head, *(point.row for point in rows)])
        at = np.concatenate([[start], times])
        for event in events:
            index = np.flatnonzero(at == event.time)
            if event.visible and index.size:
                result[index[0]] = event.after.row
        return result

    def steps(self, start: float, end: float) -> np.ndarray:
        """Integrate from `start` to `end` with the substitute of the steps of the integrator.

        The output times are a grid of `STEPS` intervals of the same length,
        refined until the linear interpolation between the rows is accurate
        (`_refine`); the events of the model at a time are two rows at it,
        the states before and after them, the row before them left out when
        the row before it is at the time already (the row of a grid time or
        of the events before them at the time).

        Returns:
            The values, a row per time, the first at the start and the last
            at the end.
        """
        grid = np.linspace(start, end, STEPS + 1)[1:]
        grid[-1] = end
        model = self.model
        head = _Point(start, state_row(model, self.selections, start), model.state)
        rows, events = self._advance(start, end, grid)
        # the rows in the order of their times, the events after the row of
        # a grid time
        order = [(point.time, 0, k) for k, point in enumerate(rows)]
        order += [(event.time, 1, k) for k, event in enumerate(events)]
        order.sort(key=lambda point: point[:2])
        points = [head]
        for time, kind, k in order:
            if kind == 0:
                points.append(rows[k])
                continue
            if points[-1].time != time:
                points.append(events[k].before)
            points.append(events[k].after)
        return np.vstack([point.row for point in self._refine(points)])

    def _refine(self, points: list[_Point]) -> list[_Point]:
        """Refine the rows of a segment until the linear interpolation between them is accurate.

        The rows are a piecewise linear curve, which every consumer of the
        steps of the integrator interpolates (a `time=` grid of a run, the
        bands, `ScanResult.interpolate`, the figures, the NCA), as the steps
        of roadrunner, whose density CVODE chose. Every interval between two
        rows at different times (two rows at one time are a jump, the states
        before and after the events there) is checked at its midpoint, which
        is integrated from the state of its first row: where the value of a
        selection differs from the mean of the two rows by more than the
        tolerance, the midpoint becomes a row and both halves are checked
        again, until every interval passes or its halves would be shorter
        than the resolution of the time (`_splits`). The tolerance is the
        square root of the relative tolerance of the integrator (`1e-5` at
        the default `1e-10`) times the largest magnitude of the selection in
        the segment: the chord error of the rows of roadrunner is `2e-5` to
        `1e-4` of the largest value at the defaults on the HCTZ model, the
        tolerance of the integrator itself would need thousands of rows per
        time constant. One integration per round and per piece between two
        jumps checks the midpoints of all its pending intervals, so a
        segment costs a handful of integrations more than its grid. A
        segment of more than `MAX_STEPS` rows is reported and refined no
        further.

        Args:
            points: the rows of the segment in the order of the output.

        Returns:
            The rows with the ones added, in the order of the output.
        """
        model = self.model
        held = (model.time, model.state, self._armed_switches)
        tolerance = math.sqrt(float(model.solver.get_relative_tolerance()))
        scale = np.max(np.abs(np.vstack([point.row for point in points])), axis=0)
        # the piece of every point, which a jump starts, and the pending
        # intervals of every piece
        pieces: dict[int, list[tuple[_Point, _Point]]] = {}
        piece = 0
        for a, b in itertools.pairwise(points):
            if a.time == b.time:
                piece += 1
            elif self._splits(a, b):
                pieces.setdefault(piece, []).append((a, b))
        added: list[_Point] = []
        while pieces and len(points) + len(added) < MAX_STEPS:
            pending, pieces = pieces, {}
            for piece, intervals in pending.items():
                intervals.sort(key=lambda pair: pair[0].time)
                first = intervals[0][0]
                midpoints = np.array(
                    [a.time + (b.time - a.time) / 2.0 for a, b in intervals]
                )
                model.set_state(first.time, first.state)
                values, x = self._integrate(first.time, midpoints)
                for k, (a, b) in enumerate(intervals):
                    middle = _Point(float(midpoints[k]), values[k + 1], x[k + 1])
                    scale = np.maximum(scale, np.abs(middle.row))
                    chord = (a.row + b.row) / 2.0
                    if np.all(np.abs(middle.row - chord) <= tolerance * scale):
                        continue
                    added.append(middle)
                    for pair in ((a, middle), (middle, b)):
                        if self._splits(*pair):
                            pieces.setdefault(piece, []).append(pair)
        if pieces:
            logger.warning(
                "The steps of the integrator of the model '%s' between the times "
                "%s and %s have %s rows, the substitute of AMICI is refined no "
                "further.",
                model.label,
                points[0].time,
                points[-1].time,
                len(points) + len(added),
            )
        model.set_state(held[0], held[1])
        self._set_switch_shifts(held[2])
        # the rows added are strictly between the rows they refine, the rows
        # of a jump keep their order
        return sorted([*points, *added], key=lambda point: point.time)

    def _splits(self, a: _Point, b: _Point) -> bool:
        """Whether the interval between two rows is long enough to be refined at its midpoint.

        Both halves are longer than the resolution of the time, so no two
        rows are within the resolution, which the interpolation of a grid
        takes as one time, see `sbmlsim.result.timecourse.grid_weights`.
        """
        return (b.time - a.time) / 2.0 > self._ahead(b.time)

    def event_at_the_end(
        self, start: float, end: float, tolerance: float
    ) -> np.ndarray | None:
        """Get the state after the events of the model at the end, see `sbmlsim.simulator.segments.Course`.

        The model is integrated from `start` to the resolution after the
        end, the events in it fired, see `_advance`.
        """
        _, events = self._advance(start, end + tolerance, np.empty(0))
        return events[-1].after.row.copy() if events else None

    def horizon(self, start: float, end: float) -> None:
        """Integrate from `start` to `end` without output, see `sbmlsim.simulator.segments.Course`."""
        self._advance(start, end, np.empty(0))

    def rates(self, states: Sequence[str]) -> np.ndarray:
        """Get the current rates of change of states, see `AmiciSBMLModel.rates`."""
        return self.model.rates(states)

    def steady_state(self, plan: SteadyStatePlan, start: float) -> float:
        """Integrate the model from a time until its rates of change vanish, see `steady_state`."""
        return integrate_to_steady_state(self.model, self, plan, start)

    def _advance(
        self, start: float, end: float, times: np.ndarray
    ) -> tuple[list[_Point], list[_Event]]:
        """Integrate from the current state at `start` to `end`, firing the events of the model.

        The model is integrated from `start` with AMICI, see the module: the
        first root of a trigger in the integration is the time of the events
        of the model which are due there, the state at the root is the one
        of the integration (at an output time, or from the last output time
        before it), the events fire there (`_fire`) and the integration
        starts again from the state after them. Every integration starts
        with the events whose trigger becomes true right after its start and
        the shifts of the comparisons at their threshold (`_arm`). An
        integration runs to the end or, after an event, over the distance
        between the last two events, doubled after every integration without
        an event, so the part past the first root, which is discarded, is at
        most as long as the one before it. A root within the resolution of
        the time after the stop of the integration before is at the stop,
        whose state that integration gave (an output time there has the
        state after the events, as at a grid time), and a trigger which an
        integration makes true within the rounding of its stop without a
        root reported by AMICI, e.g. `X >= 1` with `X` at 1 up to its last
        bit at the stop, fires at the stop: the next integration would
        start with the trigger true and never find its root.

        Args:
            start: the start.
            end: the end, which the model is at afterwards, after the events
                there.
            times: the output times, sorted, after the start and up to the
                end.

        Returns:
            The rows at the output times with their states, the state before
            the events at a time, and the events, by their times.

        Raises:
            RuntimeError: if AMICI fails to integrate, with its messages.
        """
        model = self.model
        rows: list[_Point] = []
        events: list[_Event] = []
        t = start
        previous: float | None = None
        chunk: float | None = None
        # whether `t` is the stop of the integration before, without a root
        stopped = False
        while t < end:
            events += self._arm(t)
            stop = end if chunk is None else min(end, t + chunk)
            wanted = times[(times > t) & (times <= stop)]
            if wanted.size and stop - wanted[-1] <= self._ahead(stop):
                # two output times within the resolution are one time: AMICI
                # gives a row of NaN for the second one at a root of the time
                stop = float(wanted[-1])
            grid = (
                wanted
                if wanted.size and wanted[-1] == stop
                else np.append(wanted, stop)
            )
            started = triggers(self.events, model.values, t, model)
            values, x = self._integrate(t, grid)
            found = self._root(x[-1], t)
            if found is None:
                rows += _points(grid, values, x, wanted.size)
                # a trigger which the integration made true within the
                # rounding of its stop, whose root AMICI did not report: the
                # root is at the stop, which the next integration would start
                # past, the trigger true from its start; a strict comparison
                # at its threshold since its event fired is true, see `_arm`,
                # and an event which fired at the start fired for the
                # crossing right after it unless its assignments set its
                # trigger back, see `_fire`
                reached = triggers(self.events, model.values, stop, model)
                crossed = {self.comparisons[j].event for j in self._seen}
                fired_here, rearmed = self._fired_at(t), self._rearmed_at(t)
                due = [
                    k
                    for k in range(len(self.events))
                    if started[k] is False
                    and reached[k] is True
                    and k not in crossed
                    and (k not in fired_here or k in rearmed)
                ]
                if due:
                    stopped = False
                    events += self._fire_due(
                        stop, [(stop, k) for k in due], _Point(stop, values[-1], x[-1])
                    )
                    chunk = max(
                        stop - (start if previous is None else previous),
                        self._ahead(stop),
                    )
                    previous = t = stop
                    continue
                if chunk is not None:
                    chunk *= 2.0
                t = stop
                stopped = True
                continue
            root, flagged = found
            # the crossing an event fired for at the start, which AMICI finds
            # again within the resolution after it (the state at the start a
            # hair before the root): the integration starts again from the
            # root, without firing
            silent = flagged[0][1] in self._fired_at(t) and root <= t + self._ahead(t)
            # the rows before the root are the ones of the model; the events
            # are at a grid time within the resolution of the time of the
            # root, whose state the integration gave, at the start for a root
            # within the resolution after it (the stop of the integration
            # before, a hair before the root), else at the root, whose state
            # is integrated from the last grid time before it
            at = int(np.argmin(np.abs(grid - root)))
            if abs(float(grid[at]) - root) <= self._ahead(root) and not (
                silent and grid[at] < root
            ):
                time, row, state = float(grid[at]), values[at + 1], x[at + 1]
            elif root <= t or (stopped and not silent and root <= t + self._ahead(t)):
                time, row, state = t, values[0], x[0]
            else:
                time = root
                at = int(np.searchsorted(grid, root, side="left"))
                last = float(grid[at - 1]) if at else t
                model.set_state(last, x[at])
                between, y = self._integrate(last, np.array([root]))
                row, state = between[-1], y[-1]
            stopped = False
            kept = int(np.searchsorted(wanted, time, side="left"))
            rows += _points(grid, values, x, kept)
            model.set_state(time, state)
            before = _Point(time, row, state)
            if kept < wanted.size and wanted[kept] == time:
                rows.append(before)
            if silent:
                # the events fired at the start are the ones fired at the root
                self._fired = (time, self._fired_at(t))
                t = time
                continue
            # the roots up to the time, one after the other
            limit = max(root, time)
            events += self._fire_due(
                time, [d for d in flagged if d[0] <= limit], before
            )
            chunk = max(
                time - (start if previous is None else previous), self._ahead(time)
            )
            previous = t = time
        return rows, events

    def _arm(self, time: float) -> list[_Event]:
        """Prepare an integration from the current state at a time, see the module.

        The records of the events are set back. A strict comparison exactly
        at its threshold is true for AMICI, not shifted, when it is at its
        threshold since its event fired, with the operands it had then (the
        crossing the event fired for, e.g. `time > 10` at 10, or `X > 2` with
        `X` at 2 within the rounding after it crossed, see `_fire`), and
        shifted otherwise, so that AMICI finds its crossing: one which the
        assignments of the events or a change set to its threshold arms its
        event again (`Y > 2` by `Y = 2`). For every event with a shifted
        comparison whose trigger is false and which did not fire at the
        time, the triggers at the resolution of the time after the time tell
        whether it becomes true right after it (`_probe`): then the event
        fires at the time, after its row.

        Returns:
            The events which fire right after the time.
        """
        model = self.model
        model.set_time(time)
        model.reset_events()
        if not self.comparisons:
            return []
        events = self.events
        result: list[_Event] = []
        while True:
            values = model.values
            strict = triggers(events, values, time, model)
            held = self._held(time)
            shifts = [0.0] * len(self.comparisons)
            # a comparison which left its threshold is at it again only by a
            # change or an assignment, which arms its event again
            self._seen = {
                j: operands
                for j, operands in self._seen.items()
                if any(j == h[0] for h in held)
            }
            if not held:
                break
            known = {j for j, a, b in held if self._seen.get(j) == (a, b)}
            fired = self._fired_at(time)
            candidates = sorted(
                {
                    self.comparisons[j].event
                    for j, _, _ in held
                    if j not in known and strict[self.comparisons[j].event] is False
                }
                - fired
            )
            ahead: list[bool | None] = (
                self._probe(time) if candidates else [None] * len(events)
            )
            due = [k for k in candidates if ahead[k] is True]
            if due:
                head = _Point(
                    time, state_row(model, self.selections, time), model.state
                )
                result.append(self._fire(time, due, head, visible=False))
                continue
            for j, _, _ in held:
                if j not in known:
                    shifts[j] = self._shift(self._operands, j, time)
            break
        model.set_shifts(shifts)
        return result

    def _arm_switches(self, time: float) -> None:
        """Shift every comparison of the maths of the model which is exactly at its threshold, see the module.

        Before every integration from the current state at a time: a
        comparison at its threshold is shifted (`_shift`), so that it keeps
        its value until the operands move and AMICI finds the crossing,
        every other one is as written.

        Args:
            time: the time of the start of the integration.
        """
        if not self.switches:
            return
        if self._switch_operands is None:
            self._switch_operands = _compile_operands(
                (s.left, s.right) for s in self.switches
            )
        shifts = [0.0] * len(self.switches)
        for j, _, _ in self._at_threshold(self._switch_operands, time):
            shifts[j] = self._shift(self._switch_operands, j, time)
        self._set_switch_shifts(shifts)

    def _set_switch_shifts(self, shifts: list[float]) -> None:
        """Set the shifts of the comparisons of the maths of the model and keep them, see `_arm_switches`."""
        self._armed_switches = shifts
        if self.switches:
            self.model.set_switch_shifts(shifts)

    def _shift(
        self, comparisons: Sequence[_Comparison | None] | None, j: int, time: float
    ) -> float:
        """Get the shift of a comparison exactly at its threshold at a time.

        `THRESHOLD_ULPS` units in the last place of the largest magnitude of
        the additive terms of its operands, so that the shifted operand
        differs from the operand in the arithmetic of the terms (`time - 10`
        compared with 0 at the time 10 is shifted by the units of 10, not of
        0), and for an operand which reads the time at least the change of
        the difference of the operands over the resolution of the time at
        the time, the time advanced and the states held (`time/86400 > tlag`
        is shifted by the resolution over 86400), so that the explicit root
        of AMICI at the shifted time, e.g. `tlag + shift` with `tlag = 0` at
        the start 0, is after the time for CVODES and the comparison turns
        at most about the resolution of the time late whatever its scale;
        at least `SMALLEST_SHIFT`.

        Args:
            comparisons: the compiled comparisons, see `_compile_operands`.
            j: the position of the comparison, which is compiled.
            time: the time.
        """
        assert comparisons is not None
        comparison = comparisons[j]
        assert comparison is not None
        values = self.model.values

        def evaluate(f: CompiledFormula, at: float) -> float:
            return f.evaluate(
                [at if s == "time" else values.getValue(s) for s in f.symbols]
            )

        magnitude = max(
            abs(evaluate(f, time)) for f in (comparison.left, *comparison.terms)
        )
        shift = max(THRESHOLD_ULPS * float(np.spacing(magnitude)), SMALLEST_SHIFT)
        if comparison.reads_time:
            later = time + self._ahead(time)
            difference = [
                evaluate(comparison.left, at) - evaluate(comparison.right, at)
                for at in (time, later)
            ]
            shift = max(shift, abs(difference[1] - difference[0]))
        return shift

    def _held(self, time: float) -> list[tuple[int, float, float]]:
        """Get the strict comparisons exactly at their threshold, with the values of their operands."""
        if self._operands is None:
            self._operands = _compile_operands(
                (c.left, c.right) for c in self.comparisons
            )
        return self._at_threshold(self._operands, time)

    def _at_threshold(
        self, operands: Sequence[_Comparison | None], time: float
    ) -> list[tuple[int, float, float]]:
        """Get the comparisons exactly at their threshold at a time, from the current state.

        Returns:
            The positions of the comparisons whose operands are equal, with
            the values of their operands.
        """
        values = self.model.values
        result: list[tuple[int, float, float]] = []
        for j, pair in enumerate(operands):
            if pair is None:
                continue
            left, right = (
                f.evaluate(
                    [time if s == "time" else values.getValue(s) for s in f.symbols]
                )
                for f in (pair.left, pair.right)
            )
            if left == right:
                result.append((j, left, right))
        return result

    def _probe(self, time: float) -> list[bool | None]:
        """Evaluate the triggers at the resolution of the time after a time, from the current state.

        The model is set back to its state at the time.
        """
        model = self.model
        state = model.state
        later = time + self._ahead(time)
        self._integrate(time, np.array([later]))
        result = triggers(self.events, model.values, later, model)
        model.set_state(time, state)
        return result

    def _ahead(self, time: float) -> float:
        """Get the resolution of the time at a time, which CVODES integrates from it."""
        return max(self.resolution, TIME_RTOL * abs(time))

    def _visible(self, k: int, time: float) -> bool:
        """Whether the trigger of an event is true at a time, every comparison at its threshold within the tolerance of the integrator taken at it.

        The state at a root is the one at its threshold up to the tolerance
        of the integrator (and the resolution of the time, at a grid time
        within it), so a comparison within `atol + rtol * max(|a|, |b|)` of
        its threshold, `a >= b` with `a` a little below `b`, is at it: true
        when it is not strict, false when it is strict, which fires right
        after the time, see the module.

        Args:
            k: the event.
            time: the time.
        """
        tolerant = self._tolerant_triggers()[k]
        if tolerant is None:
            return triggers(self.events, self.model.values, time, self.model)[k] is True
        formula, operands = tolerant
        values = self.model.values
        solver = self.model.solver
        atol = float(solver.get_absolute_tolerance())
        rtol = float(solver.get_relative_tolerance())

        def value(symbol: str) -> float:
            if symbol == "time":
                return time
            if not symbol.startswith(_TOLERANCE):
                return values.getValue(symbol)
            left, right = (
                f.evaluate(
                    [time if s == "time" else values.getValue(s) for s in f.symbols]
                )
                for f in operands[int(symbol[len(_TOLERANCE) :])]
            )
            return atol + rtol * max(abs(left), abs(right))

        return bool(formula.evaluate([value(s) for s in formula.symbols]))

    def _tolerant_triggers(
        self,
    ) -> list[
        tuple[CompiledFormula, list[tuple[CompiledFormula, CompiledFormula]]] | None
    ]:
        """Get the triggers with every comparison tolerant of its threshold, compiled, see `_visible`.

        `None` for a trigger the simulator does not evaluate.
        """
        from sbmlsim.sbml_math import compile_math

        if self._tolerant is None:
            self._tolerant = []
            for event in self.events:
                try:
                    if event.trigger is None:
                        raise ValueError("no trigger")
                    mathml, operands = _tolerant_trigger(event.trigger)
                    self._tolerant.append(
                        (
                            compile_math(mathml),
                            [(compile_math(a), compile_math(b)) for a, b in operands],
                        )
                    )
                except ValueError:
                    self._tolerant.append(None)
        return self._tolerant

    def _root(
        self, state: np.ndarray, time: float
    ) -> tuple[float, list[tuple[float, int]]] | None:
        """Get the first root of the triggers in an integration from a time, from its last state.

        A root of an event at the time which the executor fired at the time
        is the crossing it fired for (AMICI fires a trigger of the time
        within the rounding of the start at the start), not a root.

        Returns:
            The time of the first root and the events which fired, with the
            times of their first executions, by these times; `None` without
            a root.
        """
        model = self.model
        fired = model.fired(state)
        first = model.first_times(state)
        here = self._fired_at(time)
        flagged = sorted(
            (float(first[k]), k)
            for k in range(len(self.events))
            if fired[k] and not (first[k] <= time and k in here)
        )
        if not flagged:
            return None
        return flagged[0][0], flagged

    def _fire_due(
        self, time: float, due: Sequence[tuple[float, int]], before: _Point
    ) -> list[_Event]:
        """Fire the events of roots at a time, by their roots, the ones whose trigger is true at the time first.

        Args:
            time: the time.
            due: the events with the times of their roots, which are at the
                time within its resolution.
            before: the state before them, with its row.

        Returns:
            The events, the ones of a root in the order of the roots, the
            visible ones (their trigger true at the time) before the ones
            which fire right after the time.
        """
        result: list[_Event] = []
        for root in sorted({d[0] for d in due}):
            group = [k for first, k in due if first == root]
            visible = [k for k in group if self._visible(k, time)]
            later = [k for k in group if k not in visible]
            for part, seen in ((visible, True), (later, False)):
                if part:
                    result.append(
                        self._fire(
                            time, part, result[-1].after if result else before, seen
                        )
                    )
        return result

    def _fire(
        self, time: float, due: Sequence[int], before: _Point, visible: bool
    ) -> _Event:
        """Fire events at a time, with the events they trigger, see `sbmlsim.simulator.changes.fire_events`.

        The triggers of the events are false before and true after the time,
        the others as they are. A strict comparison of an event which fires
        at its threshold before the events fire is the crossing the event
        fires for: it is true from then on, as long as it is at its
        threshold with the same operands, see `_arm`.

        Args:
            time: the time.
            due: the events.
            before: the state before them, with its row.
            visible: whether an output time at the time has the state after
                them, see `_Event`.
        """
        model = self.model
        values = model.values
        current = triggers(self.events, values, time, model)
        old, new = list(current), list(current)
        for k in due:
            old[k], new[k] = False, True
        held = self._held(time)
        fired = fire_events(self.events, old, values, time, model, after=new)
        # an event which does not fire, e.g. a reported one, is handled at
        # the time as well
        handled = {*due, *fired}
        self._record(time, handled)
        if visible:
            # an event whose assignments set its trigger back, false after
            # them within the tolerance of its threshold, fires again when
            # the trigger becomes true, also within the rounding of the stop
            # of an integration, see `_advance`
            rearmed = frozenset(k for k in handled if not self._visible(k, time))
            if rearmed:
                self._rearmed = (time, self._rearmed_at(time) | rearmed)
        for j, left, right in held:
            if self.comparisons[j].event in handled:
                self._seen[j] = (left, right)
        model.set_time(time)
        after = _Point(time, state_row(model, self.selections, time), model.state)
        return _Event(time, before, after, visible)

    def _record(self, time: float, fired: Collection[int]) -> None:
        """Record the events the executor fired at a time."""
        if fired:
            self._fired = (time, self._fired_at(time) | frozenset(fired))

    def _fired_at(self, time: float) -> frozenset[int]:
        """Get the events the executor fired at a time."""
        return self._fired[1] if self._fired[0] == time else frozenset()

    def _rearmed_at(self, time: float) -> frozenset[int]:
        """Get the events fired at a time whose assignments set their trigger back, see `_fire`."""
        return self._rearmed[1] if self._rearmed[0] == time else frozenset()

    def _integrate(
        self, start: float, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate from the current state at `start` with output at times with AMICI.

        The model is in the state at the last time afterwards. AMICI 1.2
        gives a row of NaN for a last output time within its rounding of a
        trigger of the time it handles there (the row is right when an
        output time follows): such a row is integrated again from the row
        before it with an output time the resolution of the time after it.

        Args:
            start: the start.
            times: the output times, sorted, from the start on.

        Returns:
            The values of the selections and the states of AMICI, a row at
            the start and a row per time.

        Raises:
            RuntimeError: if AMICI fails to integrate, with its messages, or
                gives a row which is not a number.
        """
        model = self.model
        grid = np.concatenate([[start], np.asarray(times, dtype=float)])
        x, w = self._run(grid)
        if np.isnan(x[-1]).any() and grid.size > 1 and not np.isnan(x[-2]).any():
            last = float(grid[-1])
            model.set_state(float(grid[-2]), x[-2])
            again, w_again = self._run(
                np.array([grid[-2], last, last + self._ahead(last)])
            )
            x[-1], w[-1] = again[1], w_again[1]
        if np.isnan(x).any():
            raise RuntimeError(
                f"AMICI gave values which are not numbers for the model "
                f"'{model.label}' between the times {start} and {grid[-1]}."
            )
        values = self._transform(grid, x, w)
        for k in self._switch_rows(grid, x, w):
            # the expressions at a row exactly at the threshold of a
            # comparison of a math, with the comparison as written
            model.set_state(float(grid[k]), x[k])
            values[k] = state_row(model, self.selections, float(grid[k]))
        model.set_state(float(grid[-1]), x[-1])
        return values, x

    def _switch_rows(self, grid: np.ndarray, x: np.ndarray, w: np.ndarray) -> list[int]:
        """Get the rows of an integration after its first at which a comparison of a math is exactly at its threshold.

        AMICI evaluates the expressions of an output time at the root of a
        comparison of a math, e.g. `time <= tend` at `tend`, after the
        root, so the rates and the assignment rules of the row are the ones
        after the switch; roadrunner evaluates them with the comparison as
        written, the value of the SBML. The operands of the comparisons are
        evaluated on the rows at once; the executor evaluates the rows found
        again with the comparison as written, see `_integrate`.

        Args:
            grid: the times of the integration, the start first.
            x: the states of AMICI, a row per time.
            w: the expressions of AMICI, a row per time.
        """
        if not self.switches or grid.size < 2:
            return []
        if self._switch_operands is None:
            self._switch_operands = _compile_operands(
                (s.left, s.right) for s in self.switches
            )
        if self._switch_columns is None:
            symbols = sorted(
                {
                    symbol
                    for comparison in self._switch_operands
                    if comparison is not None
                    for formula in (comparison.left, comparison.right)
                    for symbol in formula.symbols
                }
            )
            self._switch_columns = (symbols, self.model.output_transform(symbols))
        symbols, transform = self._switch_columns
        columns = transform(grid[1:], x[1:], w[1:])
        values = {symbol: columns[:, k] for k, symbol in enumerate(symbols)}
        at = np.zeros(grid.size - 1, dtype=bool)
        for comparison in self._switch_operands:
            if comparison is None:
                continue
            left, right = (
                np.asarray(f.apply([values[s] for s in f.symbols]), dtype=float)
                for f in (comparison.left, comparison.right)
            )
            at |= left == right
        return [int(k) + 1 for k in np.flatnonzero(at)]

    def _run(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run AMICI from the current state at the first time of a grid with output at the others.

        Returns:
            The states and the expressions of AMICI, a row per time of the
            grid.

        Raises:
            RuntimeError: if AMICI fails to integrate, with its messages.
        """
        model = self.model
        model.set_time(float(grid[0]))
        self._arm_switches(float(grid[0]))
        model.prepare(grid)
        amodel = model.amici_model
        rdata = self._run_simulation(amodel, model.solver)
        if rdata.status != 0:
            sundials = importlib.import_module("amici.sim.sundials")
            messages = "; ".join(str(m.message) for m in rdata.messages)
            raise RuntimeError(
                f"AMICI failed to integrate the model '{model.label}' from the time "
                f"{grid[0]} to {grid[-1]}: "
                f"{sundials.simulation_status_to_str(rdata.status)}, {messages}"
            )
        # the SWIG object of the return data, whose vectors `vector_array`
        # copies at once; a model without states, e.g. of assignment rules
        # only, has an empty `x`
        data = rdata.ptr
        x = vector_array(data.x).reshape(grid.size, amodel.nx_rdata)
        w = vector_array(data.w).reshape(grid.size, amodel.nw)
        return x, w


def _points(
    times: np.ndarray, values: np.ndarray, x: np.ndarray, count: int
) -> list[_Point]:
    """Get the first rows of an integration after its start as points, see `_Point`.

    Args:
        times: the output times of the integration after its start.
        values: the values of the selections, a row at the start and a row
            per time.
        x: the states of AMICI, a row at the start and a row per time.
        count: the number of rows after the start.
    """
    return [_Point(float(times[k]), values[k + 1], x[k + 1]) for k in range(count)]


def _compile_operands(
    pairs: Iterable[tuple[str, str]],
) -> list[_Comparison | None]:
    """Compile the operands of comparisons, MathML of selections, see `sbmlsim.sbml_math.compile_math`.

    Returns:
        The compiled operands of every comparison with their additive terms
        (`_terms`), `None` for one which the simulator does not evaluate,
        e.g. of `rateOf`, which is left to AMICI as written.
    """
    # sbmlmath is imported by the first model with events, see `sbml_math`
    from sbmlsim.sbml_math import compile_math

    result: list[_Comparison | None] = []
    for left, right in pairs:
        try:
            compiled_left, compiled_right = compile_math(left), compile_math(right)
            terms = tuple(
                compile_math(term) for term in (*_terms(left), *_terms(right))
            )
        except ValueError:
            result.append(None)
            continue
        result.append(
            _Comparison(
                compiled_left,
                compiled_right,
                terms,
                any("time" in f.symbols for f in (compiled_left, compiled_right)),
            )
        )
    return result


def _terms(mathml: str) -> list[str]:
    """Get the additive terms of a math, MathML of selections, as MathML.

    The operands of the sums and differences of the math, through the nested
    ones, e.g. `time` and `tend` of `time - tend`; a math which is no sum or
    difference is its own term.

    Raises:
        ValueError: if the MathML is not read.
    """
    node: libsbml.ASTNode | None = libsbml.readMathMLFromString(mathml)
    if node is None:
        raise ValueError(f"The math {mathml!r} cannot be read.")
    terms: list[str] = []
    stack = [node]
    while stack:
        n = stack.pop()
        if n.getType() in (libsbml.AST_PLUS, libsbml.AST_MINUS):
            stack.extend(n.getChild(k) for k in range(n.getNumChildren()))
        else:
            terms.append(str(libsbml.writeMathMLToString(n)))
    return terms


def _tolerant_trigger(mathml: str) -> tuple[str, list[tuple[str, str]]]:
    """Get a trigger with every comparison tolerant of its threshold, see `_Course._visible`.

    A comparison `a > b` becomes `a > b + T`, `a < b` `a + T < b`, `a >= b`
    `a >= b - T` and `a <= b` `a <= b + T`, with `T` the symbol of its
    tolerance (`_TOLERANCE` and the position of the comparison); a trigger
    has no comparison for equality, see `AmiciSBMLModel`.

    Args:
        mathml: the MathML of the trigger, of selections.

    Returns:
        The MathML of the tolerant trigger and the MathML of the operands of
        every comparison, in the order of the symbols of the tolerances.

    Raises:
        ValueError: if the MathML is not read.
    """
    node: libsbml.ASTNode | None = libsbml.readMathMLFromString(mathml)
    if node is None:
        raise ValueError(f"The trigger {mathml!r} cannot be read.")

    def apply(kind: int, *children: libsbml.ASTNode) -> libsbml.ASTNode:
        result = libsbml.ASTNode(kind)
        for child in children:
            result.addChild(child)
        return result

    operands: list[tuple[str, str]] = []
    for parent, index, comparison in trigger_comparisons(node, COMPARISONS):
        tolerance = libsbml.ASTNode(libsbml.AST_NAME)
        tolerance.setName(f"{_TOLERANCE}{len(operands)}")
        left: libsbml.ASTNode = comparison.getChild(0).deepCopy()
        right: libsbml.ASTNode = comparison.getChild(1).deepCopy()
        operands.append(
            (
                str(libsbml.writeMathMLToString(left)),
                str(libsbml.writeMathMLToString(right)),
            )
        )
        kind = comparison.getType()
        if kind == libsbml.AST_RELATIONAL_LT:
            tolerant = apply(kind, apply(libsbml.AST_PLUS, left, tolerance), right)
        elif kind == libsbml.AST_RELATIONAL_GEQ:
            tolerant = apply(kind, left, apply(libsbml.AST_MINUS, right, tolerance))
        else:
            # `>` and `<=`
            tolerant = apply(kind, left, apply(libsbml.AST_PLUS, right, tolerance))
        if parent is None:
            node = tolerant
        else:
            parent.replaceChild(index, tolerant, True)
    return str(libsbml.writeMathMLToString(node)), operands


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
    columns = output_columns(selections)
    model.initialize(preinit_targets(plan))
    try:
        course = _Course(model, columns, time_tolerance(plan.start, plan.end))
        if plan.steady_state is not None:
            course.start_events(0.0)
            course.steady_state(plan.steady_state, 0.0)
        else:
            course.start_events(plan.start)
        values = simulate_segments(plan, model, model.values, columns, course)
    finally:
        model.simulated()
    values = output_time(values, columns, plan, model.symbols)
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

    Every horizon fires the events of the model, see `_Course`.

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
