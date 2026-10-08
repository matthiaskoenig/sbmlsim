"""Running a semantic case of the SBML Test Suite.

`run_case` reads the model of a case, simulates it as its settings ask and
compares the results with the expected ones. It answers with a `CaseResult`
rather than raising: a case whose model cannot be read or whose integration
fails is a result of the suite like any other, and the report groups the cases
by exactly these outcomes.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from sbmlsim import parallel
from sbmlsim.model import AbstractModel
from sbmlsim.result import TimecourseResult
from sbmlsim.simulation import Simulation
from sbmlsim.simulator.simulation_serial import SimulatorSerial
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

    The last three are failures of the simulator and not of its numerics, so a
    report separates them: they say something different about what is missing.
    """

    PASS = "pass"
    TOLERANCE = "tolerance"
    NOT_READ = "not_read"
    SIMULATION_ERROR = "simulation_error"
    MISSING_VARIABLE = "missing_variable"


#: the outcomes which are not a pass
FAILED_STATUS: tuple[CaseStatus, ...] = (
    CaseStatus.TOLERANCE,
    CaseStatus.NOT_READ,
    CaseStatus.SIMULATION_ERROR,
    CaseStatus.MISSING_VARIABLE,
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


def simulate_case(case: SemanticCase, simulator: SimulatorSerial) -> TimecourseResult:
    """Simulate a case on a simulator which has its model loaded.

    Args:
        case: the case to simulate.
        simulator: simulator with the model of the case.

    Returns:
        The results with a `time` column and one column per selection.
    """
    simulator.set_timecourse_selections(selections=case.selections)
    # the model is initialized with the binding of `resetAll` and not with the
    # method: roadrunner exposes the symbols of a model as attributes of the
    # instance, a model with a species or a parameter named `reset` hides the
    # method of that name (case 00952), see `model_roadrunner.reset_all`
    simulation = Simulation(
        start=case.start, end=case.start + case.duration, steps=case.steps
    )
    return simulator.simulate(simulation)


def map_cases[T](
    function: Callable[[SemanticCase], T],
    cases: Sequence[SemanticCase],
    workers: int | None = None,
) -> list[T]:
    """Apply a function to every case, in parallel processes.

    The cases are independent of each other, so they are spread over a pool of
    processes. A case takes from milliseconds to seconds, so they are handed
    out in small chunks, which keeps every process busy until the end. The
    pool is the kept pool of `sbmlsim.parallel`.

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
    executor = parallel.pool(min(workers, len(cases)))
    return list(executor.map(function, cases, chunksize=chunksize))


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

    simulator = SimulatorSerial(
        absolute_tolerance=INTEGRATOR_ABSOLUTE_TOLERANCE,
        relative_tolerance=INTEGRATOR_RELATIVE_TOLERANCE,
        variable_step_size=False,
    )
    try:
        simulator.set_model(model=AbstractModel(source=case.model_path))
    except Exception as err:
        logger.debug("'%s': the model could not be read: %s", case.cid, err)
        return result(CaseStatus.NOT_READ, str(err).strip().splitlines()[0][:300])

    try:
        observed = simulate_case(case, simulator)
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
    return result(CaseStatus.PASS, comparison=comparison)
