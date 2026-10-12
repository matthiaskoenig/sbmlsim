"""Running a semantic case of the SBML Test Suite.

`run_case` reads the model of a case, simulates it as its settings ask and
compares the results with the expected ones. It answers with a `CaseResult`
rather than raising: a case whose model cannot be read or whose integration
fails is a result of the suite like any other, and the report groups the cases
by exactly these outcomes. A case which passes is simulated again on its
loaded model, as every point of a scan after the first and every simulation
of an experiment after the first are, and these simulations must equal the
first, bitwise (#305): the same simulation again, the simulation after a
shorter one, and a steady state before it, compared with that of a fresh
model, see `repeat_check`.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from sbmlsim import parallel
from sbmlsim.comparison.diff import within_tolerance
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.result import TimecourseResult
from sbmlsim.simulation import Simulation, SteadyState
from sbmlsim.simulator import Simulator
from sbmlsim.testsuite.cases import SemanticCase, SemanticSuite
from sbmlsim.testsuite.comparison import CaseComparison, compare_case

logger = logging.getLogger(__name__)

#: tolerances of the integration. They are not the tolerances of the case,
#: which decide whether a result is correct: integrating no more accurately
#: than the comparison demands is how a suite runner passes cases it should
#: not. The integration is an order of magnitude tighter than the tightest
#: tolerance the cases compare with
INTEGRATOR_ABSOLUTE_TOLERANCE = 1e-12
INTEGRATOR_RELATIVE_TOLERANCE = 1e-12


class CaseStatus(StrEnum):
    """The outcome of a case of the suite.

    `pass` : the results are within the tolerances of the case.

    `tolerance` : the case was simulated and its results are outside the
    tolerances, i.e. the simulation is wrong.

    `not_read` : the model could not be loaded, e.g. an SBML construct which
    the simulator does not support.

    `simulation_error` : the model was loaded and the integration failed.

    `missing_variable` : the simulation ran and did not produce a variable the
    case compares, e.g. a symbol the simulator does not expose.

    `repeat` : the results are within the tolerances, and a simulation of the
    loaded model after it differs from the first or fails, i.e. a simulation
    depends on the ones before it on the same model, see `repeat_check`.

    `not_read`, `simulation_error` and `missing_variable` are failures of the
    simulator and not of its numerics, so a report separates them: they say
    something different about what is missing.
    """

    PASS = "pass"
    TOLERANCE = "tolerance"
    NOT_READ = "not_read"
    SIMULATION_ERROR = "simulation_error"
    MISSING_VARIABLE = "missing_variable"
    REPEAT = "repeat"


#: the outcomes which are not a pass
FAILED_STATUS: tuple[CaseStatus, ...] = (
    CaseStatus.TOLERANCE,
    CaseStatus.NOT_READ,
    CaseStatus.SIMULATION_ERROR,
    CaseStatus.MISSING_VARIABLE,
    CaseStatus.REPEAT,
)


@dataclass(frozen=True)
class CaseResult:
    """The result of running a case of the suite.

    Attributes:
        cid: identifier of the case.
        status: outcome of the case.
        message: what went wrong, empty for a case which passed.
        duration: seconds the case took, including reading the model.
        comparison: comparison with the expected results, `None` if the case
            was never simulated.
        component_tags: SBML components of the case, for the report.
        test_tags: what the case tests, for the report.
        encoding: SBML encoding which was simulated, e.g. `l3v2`.
    """

    cid: str
    status: CaseStatus
    message: str = ""
    duration: float = 0.0
    comparison: CaseComparison | None = None
    component_tags: frozenset[str] = field(default_factory=frozenset)
    test_tags: frozenset[str] = field(default_factory=frozenset)
    encoding: str = ""

    @property
    def passed(self) -> bool:
        """Check whether the case passed."""
        return self.status is CaseStatus.PASS

    def to_dict(self) -> dict[str, object]:
        """Convert to a dictionary of JSON serializable values."""
        return {
            "cid": self.cid,
            "status": self.status.value,
            "message": self.message,
            "duration": self.duration,
            "component_tags": sorted(self.component_tags),
            "test_tags": sorted(self.test_tags),
            "encoding": self.encoding,
        }

    def __str__(self) -> str:
        """Get string representation."""
        return f"{self.cid}: {self.status.value}{f' ({self.message})' if self.message else ''}"


#: the test tag of the cases whose simultaneous events of the same priority
#: are executed in a random order, see `repeat_check`
RANDOM_EVENT_EXECUTION = "RandomEventExecution"


def case_simulation(
    case: SemanticCase, fraction: float = 1.0, steady_state: bool = False
) -> Simulation:
    """Get the simulation of a case, as its settings ask.

    Args:
        case: the case.
        fraction: the part of its duration, with as many of its steps.
        steady_state: integrate to a steady state before it, for at most the
            duration of the case; one which is not reached raises.

    Returns:
        The simulation.
    """
    steps = max(1, round(case.steps * fraction))
    return Simulation(
        start=case.start,
        end=case.start + case.duration * fraction,
        steps=steps,
        presimulation=SteadyState(max_time=case.duration) if steady_state else None,
    )


def simulate_case(
    case: SemanticCase,
    simulator: Simulator,
    model: RoadrunnerSBMLModel,
    simulation: Simulation | None = None,
) -> TimecourseResult:
    """Simulate a case with the selections and the output of its settings.

    Args:
        case: the case.
        simulator: the simulator with the tolerances of the test suite.
        model: the model of the case, loaded by the simulator.
        simulation: the simulation, the one of the settings of the case by
            default, see `case_simulation`.

    Returns:
        The values of the selections of the case at its output times, a `time`
        column and one column per selection.
    """
    model.set_selections(case.selections)
    # the model is initialized with the binding of `resetAll` and not with the
    # method: roadrunner exposes the symbols of a model as attributes of the
    # instance, a model with a species or a parameter named `reset` hides the
    # method of that name (case 00952), see `model_roadrunner.reset_all`
    return simulator.simulate(model, simulation or case_simulation(case))


def seed_event_order(seed: int = 1) -> None:
    """Seed the random generator which orders simultaneous events.

    libroadrunner executes simultaneous events of the same priority in a
    random order, drawn with `std::rand` of the C library, whose state
    belongs to the process: a second simulation of a model with such events
    draws other numbers than the first and differs in the last bits (cases
    00952 and 01588), and so does a case after other cases in a process. The
    generator is seeded before every simulation of a case, so a case gives the
    same results in every process and the repeat check compares simulations of
    the same order; a C library which is not found keeps its state.

    Args:
        seed: the seed of `srand`.
    """
    try:
        libc = ctypes.CDLL("ucrtbase") if sys.platform == "win32" else ctypes.CDLL(None)
        libc.srand(ctypes.c_uint(seed))
    except (OSError, AttributeError) as err:
        logger.debug("The random generator of the C library is not seeded: %s", err)


def repeat_difference(
    first: TimecourseResult,
    second: TimecourseResult,
    what: str = "the second simulation of the loaded model",
) -> str:
    """Describe how a later simulation differs from the first, bitwise.

    Args:
        first: the first simulation.
        second: the later simulation.
        what: what the later simulation is, for the description.

    Returns:
        An empty string for equal results, `NaN` equal to `NaN`, else what
        differs.
    """
    a = np.asarray(first.values, dtype=float)
    b = np.asarray(second.values, dtype=float)
    if first.columns != second.columns or a.shape != b.shape:
        return f"{what} has the shape {b.shape} and not {a.shape} of the first"
    differ = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    if not differ.any():
        return ""
    worst = int(np.argmax(np.where(differ, np.abs(a - b), -1.0).ravel()))
    row, col = divmod(worst, a.shape[1])
    return (
        f"{what} differs from the first in {int(differ.sum())} values, at most "
        f"{abs(a[row, col] - b[row, col]):.3g} in '{first.columns[col]}' at the "
        f"time {a[row, 0]:g} ({a[row, col]!r} and {b[row, col]!r})"
    )


def _steady_state(
    case: SemanticCase, simulator: Simulator, model: RoadrunnerSBMLModel
) -> TimecourseResult | str:
    """Simulate a case after a steady state, the results or why it failed."""
    seed_event_order()
    try:
        return simulate_case(
            case, simulator, model, case_simulation(case, steady_state=True)
        )
    except Exception as err:
        return type(err).__name__


def repeat_check(
    case: SemanticCase,
    simulator: Simulator,
    model: RoadrunnerSBMLModel,
    first: TimecourseResult,
) -> str:
    """Simulate a case again on its loaded model and describe what differs.

    Every point of a scan after the first of its chunk and every simulation of
    an experiment after the first runs on a model which was simulated before
    (#292, #303, #305). After the first simulation of a case, which passed:

    1. the same simulation again;
    2. a simulation of half the duration, then the simulation of the case,
       which integrates beyond it, so the pending events of the shorter one
       are discarded (a model with an event with a delay is generated again);
    3. the simulation with a steady state before it, which integrates without
       end, compared with that simulation of a fresh model; both may fail to
       reach a steady state, then with the same error.

    Each must equal the first, bitwise. The C random generator is seeded
    before every simulation, see `seed_event_order`. A regeneration draws from
    it as well, so a case of random event execution (the test tag
    `RandomEventExecution`) is compared with its expected results in 2 and
    within the tolerances of the case in 3 instead.

    Args:
        case: the case.
        simulator: the simulator of the case.
        model: the loaded model, simulated once.
        first: the first simulation.

    Returns:
        An empty string, or what differs.
    """
    random = RANDOM_EVENT_EXECUTION in case.test_tags

    seed_event_order()
    second = simulate_case(case, simulator, model)
    difference = repeat_difference(first, second)
    if difference:
        return difference

    simulate_case(case, simulator, model, case_simulation(case, fraction=0.5))
    seed_event_order()
    third = simulate_case(case, simulator, model)
    difference = repeat_difference(
        first, third, "a simulation of the loaded model after a shorter one"
    )
    if difference and not (random and compare_case(case, third).valid):
        return difference

    fresh = simulator.load(AbstractModel(source=case.model_path))
    expected = _steady_state(case, simulator, fresh)
    observed = _steady_state(case, simulator, model)
    what = "a steady state after the simulations of the loaded model"
    if isinstance(expected, str) or isinstance(observed, str):
        if expected != observed:
            ref = expected if isinstance(expected, str) else "results"
            got = observed if isinstance(observed, str) else "results"
            return f"{what} gives {got}, a steady state of a fresh model {ref}"
        return ""
    difference = repeat_difference(expected, observed, what)
    if difference and random and expected.values.shape == observed.values.shape:
        agree = within_tolerance(
            np.asarray(expected.values),
            np.asarray(observed.values),
            case.absolute_tolerance,
            case.relative_tolerance,
        )
        if np.all(agree):
            return ""
    return difference


def map_cases[T](
    function: Callable[[SemanticCase], T],
    cases: Sequence[SemanticCase],
    workers: int | None = None,
) -> list[T]:
    """Apply a function to every case, in parallel processes.

    The cases are independent of each other, so they are spread over a pool of
    processes. A case takes from milliseconds to seconds, so they are handed
    out in small chunks, which keeps every process busy until the end. The
    pool is the kept pool of `sbmlsim.parallel`; a run which is interrupted,
    e.g. by Ctrl-C, which the workers ignore, stops it.

    Args:
        function: a module level function, the processes import it.
        cases: the cases.
        workers: number of processes, the cores available to this process by
            default; `1` applies the function in this process.

    Returns:
        The answers of the function, in the order of the cases.
    """
    workers = workers or os.process_cpu_count() or 1
    if workers == 1 or len(cases) <= 1:
        return [function(case) for case in cases]
    chunksize = max(1, min(8, len(cases) // (4 * workers)))
    executor = parallel.pool(workers)
    try:
        return list(executor.map(function, cases, chunksize=chunksize))
    except BaseException as err:
        if not isinstance(err, Exception):
            # e.g. Ctrl-C: the workers still run their chunks, the pool stops;
            # after an error of a case the pending chunks are cancelled
            parallel.stop(executor)
        raise


def run_suite(
    suite: SemanticSuite,
    cids: Iterable[str] | None = None,
    workers: int | None = None,
) -> list[CaseResult]:
    """Run the cases of a suite.

    Args:
        suite: the suite to run.
        cids: identifiers to run, all timecourse cases by default.
        workers: number of processes the cases are run in, the cores available
            to this process by default; `1` runs them in this process.

    Returns:
        The results, in the order of the case identifiers.
    """
    selected = set(cids) if cids is not None else None
    cases = [case for case in suite.cases() if selected is None or case.cid in selected]
    return map_cases(run_case, cases, workers=workers)


def run_case(case: SemanticCase) -> CaseResult:
    """Run a case of the suite and compare it with the expected results.

    Every failure is a result: a model which cannot be read, an integration
    which fails and results outside the tolerances are the outcomes the report
    is grouped by, so nothing is raised.

    Args:
        case: the case to run.

    Returns:
        The result of the case with its status and its comparison.
    """
    start = time.time()

    def result(
        status: CaseStatus,
        message: str = "",
        comparison: CaseComparison | None = None,
    ) -> CaseResult:
        """Build the result of the case with its tags and its duration."""
        return CaseResult(
            cid=case.cid,
            status=status,
            message=message,
            duration=time.time() - start,
            comparison=comparison,
            component_tags=case.component_tags,
            test_tags=case.test_tags,
            encoding=case.encoding,
        )

    simulator = Simulator(
        n_workers=1,
        absolute_tolerance=INTEGRATOR_ABSOLUTE_TOLERANCE,
        relative_tolerance=INTEGRATOR_RELATIVE_TOLERANCE,
        variable_step_size=False,
    )
    try:
        model = simulator.load(AbstractModel(source=case.model_path))
    except Exception as err:
        logger.debug("'%s': the model could not be read: %s", case.cid, err)
        return result(CaseStatus.NOT_READ, str(err).strip().splitlines()[0][:300])

    try:
        seed_event_order()
        observed = simulate_case(case, simulator, model)
    except Exception as err:
        logger.debug("'%s': the simulation failed: %s", case.cid, err)
        return result(
            CaseStatus.SIMULATION_ERROR, str(err).strip().splitlines()[0][:300]
        )

    comparison = compare_case(case, observed)
    if comparison.missing:
        return result(CaseStatus.MISSING_VARIABLE, comparison.summary, comparison)
    if not comparison.valid:
        return result(CaseStatus.TOLERANCE, comparison.summary, comparison)

    # the repeat check: the path of every point of a scan after the first
    try:
        difference = repeat_check(case, simulator, model, observed)
    except Exception as err:
        logger.debug("'%s': a repeated simulation failed: %s", case.cid, err)
        message = str(err).strip().splitlines()[0][:250]
        return result(
            CaseStatus.REPEAT,
            f"a simulation of the loaded model after the first failed: {message}",
            comparison,
        )
    if difference:
        return result(CaseStatus.REPEAT, difference, comparison)
    return result(CaseStatus.PASS, comparison=comparison)
