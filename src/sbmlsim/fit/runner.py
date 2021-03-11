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
"""
import os
import multiprocessing
from typing import Optional

import numpy as np

import logging

from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.options import (
    FittingStrategyType, OptimizationAlgorithmType, WeightingPointsType, ResidualType
)
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.utils import timeit


logger = logging.getLogger(__name__)
lock = multiprocessing.Lock()


@timeit
def run_optimization(
    problem: OptimizationProblem,
    size: int = 5,
    fitting_strategy: FittingStrategyType = FittingStrategyType.ABSOLUTE_VALUES,
    residual_type: ResidualType = ResidualType.ABSOLUTE_RESIDUALS,
    weighting_points: WeightingPointsType = WeightingPointsType.NO_WEIGHTING,
    seed: Optional[int] = None,
    **kwargs
) -> OptimizationResult:
    """Run the given optimization problem in a serial fashion.

    The runner executes the given OptimizationProblem and returns
    the OptimizationResults. The size defines the repeated optimizations
    of the problem. Every repeat uses different initial values.

    :param problem: problem to optimize
    :param size: integer number of optimizations
    :param weighting_points:
    :param residual_type:
    :param fitting_strategy:
    :param seed: integer random seed (for sampling of parameters)
    :param kwargs: additional arguments for optimizer, e.g. xtol
    :return: OptimizationResult
    """
    if "n_cores" in kwargs:
        # remove parallel arguments
        logger.warning(
            "Parameter 'n_cores' does not have any effect in serial optimization."
        )
        kwargs.pop("n_cores")

    # initialize problem, which calculates errors
    problem.initialize(
        fitting_strategy=fitting_strategy,
        weighting_points=weighting_points,
        residual_type=residual_type,
    )

    # new simulator instance with arguments
    simulator = SimulatorSerial(**kwargs)  # sets tolerances
    problem.set_simulator(simulator)

    # optimize
    fits, trajectories = problem.optimize(
        size=size, seed=seed, **kwargs
    )

    # process results and plots
    return OptimizationResult(
        parameters=problem.parameters, fits=fits, trajectories=trajectories
    )


@timeit
def run_optimization_parallel(
    problem: OptimizationProblem,
    size: int = 5,
    n_cores: Optional[int] = None,
    fitting_strategy: FittingStrategyType = FittingStrategyType.ABSOLUTE_VALUES,
    residual_type: ResidualType = ResidualType.ABSOLUTE_RESIDUALS,
    weighting_points: WeightingPointsType = WeightingPointsType.NO_WEIGHTING,
    seed: Optional[int] = None,
    **kwargs,
) -> OptimizationResult:
    """Run optimization in parallel.

    See run_optimization for more details.

    :param problem: uninitialized optimization problem (pickable)
    :param n_cores: number of workers
    :param size: total number of optimizations
    :param weighting_points:
    :param residual_type:
    :param fitting_strategy:
    :param seed:

    :return:
    """
    # set number of cores
    cpu_count = multiprocessing.cpu_count()
    if n_cores is None:
        n_cores = max(1, multiprocessing.cpu_count() - 1)
    if n_cores > cpu_count:
        n_cores = max(1, multiprocessing.cpu_count() - 1)
        logger.error(f"More cores then cpus requested, reducing cores to '{n_cores}'")

    print("\n--- STARTING OPTIMIZATION ---\n")
    print(f"Running {n_cores} workers")
    if size < n_cores:
        logger.warning(
            f"Less simulations then cores: '{size} < {n_cores}', "
            f"increasing number of simulations to '{n_cores}'."
        )
        size = n_cores

    sizes = [len(c) for c in np.array_split(range(size), n_cores)]

    # setting arguments
    if seed is not None:
        # set seed before getting worker seeds
        np.random.seed(seed)

    # we require seeds for the workers to get different results
    seeds = list(np.random.randint(low=1, high=2000, size=n_cores))

    args_list = []
    for k in range(n_cores):
        d = {
            "problem": problem,
            "size": sizes[k],
            "weighting_points": weighting_points,
            "residual_type": residual_type,
            "fitting_strategy": fitting_strategy,
            "seed": seeds[k],
            **kwargs
        }
        args_list.append(d)

    # worker pool
    with multiprocessing.Pool(processes=n_cores) as pool:
        opt_results = pool.map(worker, args_list)

    # combine simulation results
    print("\n--- FINISHED OPTIMIZATION ---\n")
    return OptimizationResult.combine(opt_results)


def worker(kwargs) -> OptimizationResult:
    """Worker for running optimization problem."""
    lock.acquire()
    try:
        print(f"worker <{os.getpid()}> running optimization ...")
    finally:
        lock.release()

    return run_optimization(**kwargs)
