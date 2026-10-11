"""The time roadrunner reports the rows of an event of the model at.

libroadrunner reports an event of the model in the steps of the integrator
(`variable_step_size`) as two rows: the state at the root of its trigger,
before the event, at `root - roottol`, and the state after the event at
`root + roottol`; an event which roadrunner finds at the end of a step of
CVODE rather than as a root has its first row at the end of the step less the
machine epsilon (`CVODEIntegrator::integrate` of libroadrunner 2.10). A first
step of CVODE after an event which ends within `roottol` of it is not
reported, a second one is, after the row at `root + roottol`, so the times are
not sorted. `roottol` is a `static const` of the function, which the first call
of the process fixes to `100 * 32 * eps * (|t0| + |h|)` of that call, `t0` and
`h` the start and the step of the first integration of the process, of any
model: the steps of the integrator over `0..10000`, whose first call
integrates the whole interval, fix it to 7.1e-9, roadrunner's default
`simulate(0, 10000)`, 51 output times 200 apart, to 1.42e-10. After the first
of them the rows of every event are 7.1e-9 before and after it, more than
the resolution of the time of a simulation over `0..2.5`, and a worker of a
pool, whose first integration is its own, has other rows than the process
which runs the scan.

`root_tolerance` runs a probe once per process, before the first model
sbmlsim loads (`RoadrunnerSBMLModel`) and when a worker of a pool starts
(`sbmlsim.parallel`), unless the main module, which a worker imports first,
integrates when it is imported: a model with one event, whose first
integration spans `PIN_SPAN`, so `roottol` is 6.2e-31 in every process in
which sbmlsim integrates first, the two rows of an event have the time of the
event and the first step of CVODE after it is reported. The probe then
measures `roottol` from the rows of its event, also where an integration
before fixed it, and `at_event_times` gives the two rows of every event the
time of the event; the executor restores the rows of an event less than
`roottol` before or after the end of a segment, whose row after it roadrunner
folds into its last row, see `sbmlsim.simulator.executor._events_at_the_end`.
In a process in which roadrunner integrated before sbmlsim loaded a model, the
steps inside a segment lack the first step of CVODE after an event if it is
shorter than that `roottol`, which roadrunner does not report, and the rows of
an event which CVODE finds in that step, which roadrunner applies without
rows; a worker of a pool has both, so the rows differ by them and the values
on a grid by the interpolation across them. The probe warns once.

A roadrunner whose `roottol` is no constant of the process, which the probe
cannot measure or which reports both rows of an event at its time degrades
without an error and without a warning: the probe gives a `roottol` of 0 and
the rows of an event stay at the times roadrunner reports them, which a time
of a grid sees as a chain of jumps closer than the resolution of the time,
see `sbmlsim.result.timecourse.grid_weights`. The behavior of libroadrunner
2.10 which sbmlsim depends on is pinned by the tests a new release of
roadrunner has to pass: `tests/model/test_roottol.py` (the pin, the
measurement and the warning), `tests/simulator/test_event_rows.py` (the rows
of events after a first integration with roadrunner, in fresh processes),
`tests/simulator/test_executor.py` (the events at the end of a segment and
of a simulation, the folded rows, the rounding of the output times by
roadrunner and the runs again past the end) and
`tests/fit/test_data_at_change.py` (the data at an event).
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

import numpy as np
import roadrunner

logger = logging.getLogger(__name__)

#: the machine epsilon, which roadrunner subtracts from the end of a step of
#: CVODE at which it finds an event
EPSILON = float(np.finfo(float).eps)

#: the span of the first integration of the probe, which fixes `roottol` to
#: `100 * 32 * eps * PIN_SPAN` if it is the first integration of the process
PIN_SPAN = 2.0**-60

#: the `roottol` the probe fixes if it is the first integration of the process
PINNED = 100.0 * 32.0 * EPSILON * PIN_SPAN

#: the time of the event of the probe, a power of two
PROBE_EVENT = 2.0**-10

#: the end of the integration of the probe, the largest `roottol` it measures
#: is a little less
PROBE_END = 2.0**10

#: the end of the second integration of the probe, a `roottol` which is no
#: constant of the process differs between the two
PROBE_SHORT_END = 2.0**5

#: a state `sbmlsim_roottol_x` which grows with rate 1 and an event at
#: `PROBE_EVENT` which resets it to 0; roadrunner adds an attribute of the name
#: of a parameter to its class, so the ids are the ones of sbmlsim, no name of a
#: method, which they would hide for every later instance (case 00952)
PROBE_SBML = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="sbmlsim_roottol_probe">
    <listOfParameters>
      <parameter id="sbmlsim_roottol_x" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="sbmlsim_roottol_x">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
      </rateRule>
    </listOfRules>
    <listOfEvents>
      <event id="sbmlsim_roottol_event" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <geq/>
              <csymbol encoding="text"
                definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
              <cn>{PROBE_EVENT!r}</cn>
            </apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="sbmlsim_roottol_x">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>
"""


@dataclass(frozen=True)
class RootTolerance:
    """The `roottol` of roadrunner in the process, see the module.

    Attributes:
        value: the time between an event and the row of the state after it.
        precision: the precision of the measurement of `value`.
    """

    value: float
    precision: float


@functools.cache
def root_tolerance() -> RootTolerance:
    """Get the `roottol` of roadrunner in the process, measured once by a probe.

    The probe fixes `roottol` to `PINNED` if it is the first integration of
    the process, see the module, and measures it as half the time between the
    two rows of its event, in two integrations to `PROBE_END` and to
    `PROBE_SHORT_END`. If roadrunner integrated before, the measured
    `roottol` is larger and a warning names the consequence and the remedy,
    once per process. A process which forks keeps the `roottol` of roadrunner
    and the measurement.

    Returns:
        The root tolerance; `0` if the probe fails, has no rows of its event,
        or measures two values, a `roottol` which is no constant of the
        process, so `at_event_times` leaves the rows of an event as
        roadrunner reports them.
    """
    try:
        gaps = _probe()
    except Exception as err:  # every failure of the probe degrades, see the module
        logger.debug(
            "The probe of the root tolerance of roadrunner failed, the rows of an "
            "event of a model are left at the times roadrunner reports them: %s",
            err,
        )
        return RootTolerance(value=0.0, precision=0.0)
    if None in gaps:
        logger.debug(
            "The probe of the root tolerance of roadrunner has no rows of its "
            "event, the rows of an event of a model are left at the times "
            "roadrunner reports them."
        )
        return RootTolerance(value=0.0, precision=0.0)
    gap, short = (float(g) for g in gaps if g is not None)
    precision = float(np.spacing(PROBE_EVENT + gap))
    if abs(gap - short) > 4.0 * precision:
        logger.debug(
            "roadrunner reports the rows of the event of the probe %s and %s "
            "apart in two integrations, its root tolerance is no constant of the "
            "process: the rows of an event of a model are left at the times "
            "roadrunner reports them.",
            gap,
            short,
        )
        return RootTolerance(value=0.0, precision=0.0)
    tolerance = RootTolerance(value=gap / 2.0, precision=precision)
    logger.debug("The root tolerance of roadrunner is %s.", tolerance.value)
    if tolerance.value > PINNED:
        logger.warning(
            "roadrunner integrated in this process before sbmlsim loaded a model "
            "(or before a worker of sbmlsim started, if the main module integrates "
            "when it is imported): roadrunner reports the rows of an event in the "
            "steps of the integrator %.3g before and after it, sbmlsim gives them "
            "the time of the event, but the first step of the integrator after an "
            "event is missing if it is shorter than that, and so are the rows of an "
            "event in that step, which a worker of a pool has: the rows of a run "
            "differ from the ones of a run with workers by them, and the values on "
            "a grid by the interpolation across them. Import sbmlsim and load a "
            "model before integrating with roadrunner directly.",
            tolerance.value,
        )
    return tolerance


def _probe() -> tuple[float | None, float | None]:
    """Pin `roottol` and measure the gap of the rows of the event of the probe.

    Returns:
        The times between the two rows of the event in an integration with
        the steps of the integrator to `PROBE_END` and to `PROBE_SHORT_END`,
        `None` for an integration without the rows of the event.
    """
    r = roadrunner.RoadRunner(PROBE_SBML)
    integrator: roadrunner.Integrator = r.getIntegrator()
    integrator.integrate(0.0, PIN_SPAN)
    integrator.setValue("variable_step_size", True)
    r.timeCourseSelections = ["time", "sbmlsim_roottol_x"]
    gaps: list[float | None] = []
    for end in (PROBE_END, PROBE_SHORT_END):
        r.resetAll()
        rows = np.array(r.simulate(0.0, end), dtype=float)
        # the state after the event, `sbmlsim_roottol_x` reset to 0, after the
        # start, and the state before it
        after = np.flatnonzero(rows[1:, 1] == 0.0) + 1
        if after.size == 0 or rows[after[0] - 1, 1] <= 0.0:
            gaps.append(None)
        else:
            gaps.append(float(rows[after[0], 0] - rows[after[0] - 1, 0]))
    return gaps[0], gaps[1]


def at_event_times(time: np.ndarray, tolerance: RootTolerance) -> np.ndarray:
    """Give the two rows of every event of the model the time of the event.

    Two rows of the steps of the integrator are an event if the second is
    `2 * roottol` after the first, the root of a trigger, or `roottol` and at
    most the machine epsilon after it, an event at the end of a step of CVODE,
    to the rounding of the times and of the measurement of `roottol`. Both
    rows get the time of the second less `roottol`, the time of the event, so
    an event is a jump at one time, as a change is, see the module. A row
    belongs to one event, the rows are taken from the first. roadrunner's last
    row, interpolated to the end, is no row of an event and is left out by the
    caller, see `sbmlsim.simulator.executor._at_event_times`.

    Args:
        time: the times of the rows, in their order.
        tolerance: the root tolerance of roadrunner in the process, see
            `root_tolerance`.

    Returns:
        The times, a copy.
    """
    t = np.array(time, dtype=float)
    if t.size < 2:
        return t
    value = tolerance.value
    gap = t[1:] - t[:-1]
    slack = 4.0 * (
        np.spacing(np.abs(t[:-1])) + np.spacing(np.abs(t[1:])) + tolerance.precision
    )
    pairs = np.flatnonzero(
        (np.abs(gap - 2.0 * value) <= slack) | (np.abs(gap - value) <= EPSILON + slack)
    )
    if pairs.size == 0:
        return t
    # of the pairs of adjacent rows in a row, the first, third, ... are events
    index = np.arange(pairs.size)
    first = np.maximum.accumulate(np.where(np.diff(pairs, prepend=-2) > 1, index, 0))
    pairs = pairs[(index - first) % 2 == 0]
    event = t[pairs + 1] - value
    t[pairs] = event
    t[pairs + 1] = event
    return t
