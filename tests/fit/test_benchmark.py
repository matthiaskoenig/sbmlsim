"""The speed of the evaluation of a fit: `pytest -m benchmark -n 0 -s tests/fit/test_benchmark.py`.

A fit simulates a group of fit mappings at the start and the times of its
data, with the selections of its mappings, and only the evaluation of a fit
simulates the output of the simulation for the curves of its figures. The
benchmark compares the two in the same process on the reference problem of
the HCTZ example, so it does not depend on the speed of the machine, and
reports what a simulation of all selections would cost.
"""

import statistics
import time
from collections.abc import Callable

import numpy as np
import pytest

from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.simulator.executor import execute

pytestmark = pytest.mark.benchmark

#: repeats of a measurement, the median is reported
REPEATS = 30


def _median_ms(f: Callable[[], object]) -> float:
    f()
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        f()
        times.append(time.perf_counter() - start)
    return 1000 * statistics.median(times)


@pytest.fixture(scope="module")
def problem() -> OptimizationProblem:
    """Get the initialized problem of the pharmacokinetics of HCTZ."""
    definition = FIT_DEFINITIONS["PK"]
    problem = definition.problem(opid="benchmark")
    problem.initialize(definition.settings)
    return problem


def test_the_fit_simulates_less_than_its_evaluation(
    problem: OptimizationProblem,
) -> None:
    """The simulations of a step of the fit cost less than the ones of its curves."""
    x = np.asarray(problem.x0, dtype=float)
    groups = [
        k_group
        for k_group, group in enumerate(problem.mapping_groups)
        if any(problem.mapping_kinds[k] is MappingKind.TRAINING for k in group)
    ]
    model = problem.models[0]
    everything = [
        "time",
        *model.r_loaded.model.getFloatingSpeciesIds(),
        *model.r_loaded.model.getGlobalParameterIds(),
    ]

    def simulate(curve: bool, selections: list[str] | None) -> None:
        for k_group in groups:
            group = problem.mapping_groups[k_group]
            mapped = sorted({s for k in group for s in problem.selections[k]})
            execute(
                problem.evaluated_plan(k_group, x, curve=curve),
                problem.models[group[0]],
                selections or mapped,
            )

    fit = _median_ms(lambda: simulate(False, None))
    curves = _median_ms(lambda: simulate(True, None))
    curves_all = _median_ms(lambda: simulate(True, everything))
    residuals = _median_ms(lambda: problem.residuals(problem.to_scale(x)))
    print(
        f"\n{len(groups)} groups: times of the data {fit:.2f} ms, output of the "
        f"simulation {curves:.2f} ms, with {len(everything)} selections "
        f"{curves_all:.2f} ms; residuals {residuals:.2f} ms"
    )
    assert fit < curves < curves_all
    assert np.isfinite(residuals)
