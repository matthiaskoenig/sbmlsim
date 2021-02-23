"""Module for running parameter fits in parallel.

The implementation uses multiprocessing.
Starts processes on the n_cores which run optimization problems.

-------------------------------------------------------------------------------
How multiprocessing works, in a nutshell:

    Process() spawns (fork or similar on Unix-like systems) a copy of the
    original program.
    The copy communicates with the original to figure out that
        (a) it's a copy and
        (b) it should go off and invoke the target= function (see below).
    At this point, the original and copy are now different and independent,
    and can run simultaneously.

Since these are independent processes, they now have independent Global Interpreter Locks
(in CPython) so both can use up to 100% of a CPU on a multi-cpu box, as long as they don't
contend for other lower-level (OS) resources. That's the "multiprocessing" part.
-------------------------------------------------------------------------------
"""
import logging
import multiprocessing
import os
from multiprocessing import Lock

import numpy as np

from sbmlsim.fit.analysis import OptimizationResult
from sbmlsim.fit.fit import run_optimization
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.utils import timeit


logger = logging.getLogger(__name__)
lock = Lock()


@timeit
def run_optimization_parallel(
    problem: OptimizationProblem,
    size: int,
    n_cores: int = None,
    seed: int = None,
    **kwargs,
) -> OptimizationResult:
    """Run optimization in parallel.

    :param problem: uninitialized optimization problem (pickable)
    :param n_cores: number of workers
    :param size: total number of optimizations
    :param op_dict: optimization problem

    :return:
    """
    print(problem)

    # set number of cores
    cpu_count = multiprocessing.cpu_count()
    if n_cores is None:
        n_cores = max(1, multiprocessing.cpu_count() - 1)
    if n_cores > cpu_count:
        n_cores = max(1, multiprocessing.cpu_count() - 1)
        logger.error(f"More cores then cpus requested, reducing cores to '{n_cores}'")

    print("\n--- STARTING OPTIMIZATION ---\n")
    print(f"Running {n_cores} workers")
    # FIXME: remove this bugfix
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
        d = {"problem": problem, "size": sizes[k], "seed": seeds[k], **kwargs}
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
