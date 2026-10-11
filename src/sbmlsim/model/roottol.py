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
of the process fixes to `100 * 32 * eps * (|t0| + |h|)` of that call: after a
first integration over `0..10000`, of any model, the rows of every event are
7.1e-9 before and after it, more than the resolution of the time of a
simulation over `0..2.5`, and a worker of a pool, whose first integration is
its own, has other rows than the process which runs the scan.

`root_tolerance` runs a probe once per process, before the first model
sbmlsim loads (`RoadrunnerSBMLModel`) and when a worker of a pool starts
(`sbmlsim.parallel`): a model with one event, whose first integration spans
`PIN_SPAN`, so `roottol` is 6.2e-31 in every process in which sbmlsim
integrates first, the two rows of an event have the time of the event and the
first step of CVODE after it is reported. The probe then
measures `roottol` from the rows of its event, also where an integration
before fixed it, and `at_event_times` gives the two rows of every event the
time of the event. In a process in which roadrunner integrated before sbmlsim
loaded a model, a first step of CVODE after an event which is shorter than
that `roottol` stays missing, which a worker of a pool has.
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

#: the time of the event of the probe, a power of two
PROBE_EVENT = 2.0**-10

#: the end of the integration of the probe, the largest `roottol` it measures
#: is a little less
PROBE_END = 2.0**10

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

    The probe fixes `roottol` to `100 * 32 * eps * PIN_SPAN` if it is the first
    integration of the process, see the module, and measures it as half the
    time between the two rows of its event. A process which forks keeps the
    `roottol` of roadrunner and the measurement.

    Returns:
        The root tolerance; `0` if the probe has no rows of its event, so
        `at_event_times` leaves the rows of an event as roadrunner reports
        them.
    """
    r = roadrunner.RoadRunner(PROBE_SBML)
    integrator: roadrunner.Integrator = r.getIntegrator()
    integrator.integrate(0.0, PIN_SPAN)
    r.resetAll()
    integrator.setValue("variable_step_size", True)
    r.timeCourseSelections = ["time", "sbmlsim_roottol_x"]
    rows = np.array(r.simulate(0.0, PROBE_END), dtype=float)
    # the state after the event, `sbmlsim_roottol_x` reset to 0, after the start
    after = np.flatnonzero(rows[1:, 1] == 0.0) + 1
    if after.size == 0 or rows[after[0] - 1, 1] <= 0.0:
        logger.warning(
            "The probe of the root tolerance of roadrunner has no rows of its "
            "event, the rows of an event of a model are left at the times "
            "roadrunner reports them."
        )
        return RootTolerance(value=0.0, precision=0.0)
    gap = float(rows[after[0], 0] - rows[after[0] - 1, 0])
    tolerance = RootTolerance(
        value=gap / 2.0, precision=float(np.spacing(PROBE_EVENT + gap))
    )
    logger.debug("The root tolerance of roadrunner is %s.", tolerance.value)
    return tolerance


def at_event_times(time: np.ndarray, tolerance: RootTolerance) -> np.ndarray:
    """Give the two rows of every event of the model the time of the event.

    Two rows of the steps of the integrator are an event if the second is
    `2 * roottol` after the first, the root of a trigger, or `roottol` and at
    most the machine epsilon after it, an event at the end of a step of CVODE,
    to the rounding of the times and of the measurement of `roottol`. Both
    rows get the time of the second less `roottol`, the time of the event, so
    an event is a jump at one time, as a change is, see the module. A row
    belongs to one event, the rows are taken from the first.

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
