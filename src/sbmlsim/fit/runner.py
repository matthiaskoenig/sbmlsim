"""Module for running parameter optimizations.

The optimization can run either run serial or in a parallel version.

The parallel optimization uses multiprocessing, i.e. the parallel runner
starts processes on the n_cores which run optimization problems.

How multiprocessing works, in a nutshell:

    Process() spawns (fork or similar on Unix-like systems) a copy of the
    original program.
    The copy communicates with the original to figure out that
        (a) it's a copy and
        (b) it should go off and invoke the target= function (see below).
    At this point, the original and copy are now different and independent,
    and can run simultaneously.

Since these are independent processes, they now have independent Global Interpreter
Locks (in CPython) so both can use up to 100% of a CPU on a multi-cpu box, as long as
they dont contend for other lower-level (OS) resources. That's the "multiprocessing"
part.

The `OptimizationProblem` is pickled and sent to the workers, so it must be
picklable: it is initialized in the worker, not before.
"""

import logging
import multiprocessing
import os
from typing import Any

import numpy as np

from sbmlsim.console import console
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    LossFunctionType,
    OptimizationAlgorithmType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.utils import timeit

logger = logging.getLogger(__name__)


def resolve_n_cores(n_cores: int | None) -> int:
    """Resolve the number of worker processes.

    Args:
        n_cores: requested number of workers, `None` uses all available cores
            but one.

    Returns:
        Number of workers, at least one and at most the number of available cores.
    """
    cpu_count = os.process_cpu_count() or 1
    if n_cores is None:
        return max(1, cpu_count - 1)
    if n_cores > cpu_count:
        logger.error(
            "More cores '%s' then cpus '%s' requested, reducing cores.",
            n_cores,
            cpu_count,
        )
        return max(1, cpu_count - 1)
    return max(1, n_cores)


@timeit
def run_optimization(
    problem: OptimizationProblem,
    size: int = 5,
    algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
    residual: ResidualType = ResidualType.ABSOLUTE,
    loss_function: LossFunctionType = LossFunctionType.LINEAR,
    weighting_curves: list[WeightingCurvesType] | None = None,
    weighting_points: WeightingPointsType = WeightingPointsType.NO_WEIGHTING,
    seed: int | None = None,
    variable_step_size: bool = True,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-6,
    n_cores: int | None = 1,
    serial: bool = False,
    **kwargs: Any,
) -> OptimizationResult:
    """Run optimization in parallel.

    The runner executes the given OptimizationProblem and returns
    the OptimizationResults. The size defines the repeated optimizations
    of the problem. Every repeat uses different initial values.

    To get access to the optimization problem this has to be initialized with the
    arguments of the runner.

    Args:
        problem: uninitialized problem to optimize (picklable).
        size: number of optimizations.
        algorithm: optimization algorithm to use.
        residual: handling of residuals.
        loss_function: loss function for handling outliers/residual transformation.
        weighting_curves: list of options for weighting curves (fit mappings).
        weighting_points: weighting of points.
        seed: random seed (for sampling of the start values).
        variable_step_size: use variable step size in the solver.
        relative_tolerance: relative tolerance of the simulator.
        absolute_tolerance: absolute tolerance of the simulator.
        n_cores: number of workers, `None` uses all available cores but one.
        serial: run the optimization in a serial fashion (debugging).
        kwargs: additional arguments for the optimizer, e.g. xtol.

    Returns:
        OptimizationResult with the fits of all repeats.

    Raises:
        ValueError: for the removed parameters `fitting_type` and `weighting_local`.
    """
    for deprecated, replacement in [
        ("fitting_type", "fitting_strategy"),
        ("weighting_local", "weighting_points"),
    ]:
        if deprecated in kwargs:
            raise ValueError(
                f"Deprecated parameter '{deprecated}', use '{replacement}' instead."
            )

    if weighting_curves is None:
        weighting_curves = []

    problem_kwargs: dict[str, Any] = {
        "problem": problem,
        "algorithm": algorithm,
        "residual": residual,
        "loss_function": loss_function,
        "weighting_curves": weighting_curves,
        "weighting_points": weighting_points,
        "variable_step_size": variable_step_size,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        **kwargs,
    }

    opt_result: OptimizationResult
    if serial:
        console.rule("Start optimization", align="left", style="white")
        console.log("Running serial")
        opt_result = _run_optimization_serial(size=size, seed=seed, **problem_kwargs)
    else:
        n_cores = resolve_n_cores(n_cores)
        console.rule("Start optimization", align="left", style="white")
        console.log(f"Running {n_cores} workers")
        if size < n_cores:
            logger.warning(
                "Less simulations then cores: '%s < %s', increasing number of simulations to '%s'.",
                size,
                n_cores,
                n_cores,
            )
            size = n_cores

        # distribute the repeats over the workers
        sizes = [len(c) for c in np.array_split(range(size), n_cores)]

        # every worker needs its own seed to get different start values
        seed_sequence = np.random.SeedSequence(seed)
        seeds = [int(s) for s in seed_sequence.generate_state(n_cores)]

        args_list = [
            {"size": sizes[k], "seed": seeds[k], **problem_kwargs}
            for k in range(n_cores)
        ]

        # worker pool
        with multiprocessing.Pool(processes=n_cores) as pool:
            opt_results: list[OptimizationResult] = pool.map(worker, args_list)

        # combine simulation results
        opt_result = OptimizationResult.combine(opt_results)

    console.rule("FINISHED OPTIMIZATION", align="left", style="white")
    return opt_result


def worker(kwargs: dict[str, Any]) -> OptimizationResult:
    """Worker for running optimization problem."""
    logger.info("worker <%s> running optimization ...", os.getpid())
    return _run_optimization_serial(**kwargs)


def _run_optimization_serial(
    problem: OptimizationProblem,
    size: int = 5,
    algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
    residual: ResidualType = ResidualType.ABSOLUTE,
    loss_function: LossFunctionType = LossFunctionType.LINEAR,
    weighting_curves: list[WeightingCurvesType] | None = None,
    weighting_points: WeightingPointsType = WeightingPointsType.NO_WEIGHTING,
    seed: int | None = None,
    variable_step_size: bool = True,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-6,
    **kwargs: Any,
) -> OptimizationResult:
    """Run the given optimization problem in a serial fashion.

    This function should not be called directly, but the 'run_optimization'
    should be used for executing simulations.
    See run_optimization for more detailed documentation.
    """
    if weighting_curves is None:
        weighting_curves = []

    if "n_cores" in kwargs:
        # remove parallel arguments
        logger.warning(
            "Parameter 'n_cores' does not have any effect in serial optimization."
        )
        kwargs.pop("n_cores")

    # initialize problem, which calculates errors
    problem.initialize(
        residual=residual,
        loss_function=loss_function,
        weighting_points=weighting_points,
        weighting_curves=weighting_curves,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        variable_step_size=variable_step_size,
    )

    # optimize
    fits, trajectories = problem.optimize(
        size=size, seed=seed, algorithm=algorithm, **kwargs
    )

    # process results and plots
    return OptimizationResult(
        parameters=problem.parameters, fits=fits, trajectories=trajectories
    )
