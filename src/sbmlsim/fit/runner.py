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

import multiprocessing
import os
from typing import Optional

import numpy as np
from pymetadata import log
from pymetadata.console import console

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


logger = log.get_logger(__name__)
lock = multiprocessing.Lock()


@timeit
def run_optimization(
    problem: OptimizationProblem,
    size: int = 5,
    algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
    residual: ResidualType = ResidualType.ABSOLUTE,
    loss_function: LossFunctionType = LossFunctionType.LINEAR,
    weighting_curves: list[WeightingCurvesType] = None,
    weighting_points: WeightingPointsType = WeightingPointsType.NO_WEIGHTING,
    seed: Optional[int] = None,
    variable_step_size: bool = True,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-6,
    n_cores: Optional[int] = 1,
    serial: bool = False,
    **kwargs,
) -> OptimizationResult:
    """Run optimization in parallel.

    The runner executes the given OptimizationProblem and returns
    the OptimizationResults. The size defines the repeated optimizations
    of the problem. Every repeat uses different initial values.

    To get access to the optimization problem this has to be initialized with the
    arguments of the runner.

    :param problem: uninitialized problem to optimize (pickable)
    :param size: integer number of optimizations
    :param algorithm: optimization algorithm to use
    :param residual: handling of residuals
    :param loss_function: loss function for handling outliers/residual transformation
    :param weighting_curves: list of options for weighting curves (fit mappings)
    :param weighting_points: weighting of points
    :param seed: integer random seed (for sampling of parameters)
    :param absolute_tolerance: absolute tolerance of simulator
    :param relative_tolerance: relative tolerance of simulator
    :param variable_step_size: use variable step size in solver
    :param n_cores: number of workers
    :param serial: boolean flag to execute optimization in serial fashion (debugging)
    :param kwargs: additional arguments for optimizer, e.g. xtol
    :return: OptimizationResult
    """
    saved_args = locals()
    if "fitting_type" in saved_args:
        raise ValueError(
            "Deprecated parameter 'fitting_type', use 'fitting_strategy' instead."
        )
    if "weighting_local" in saved_args:
        raise ValueError(
            "Deprecated parameter 'weighting_local', use 'weighting_points' instead."
        )

    if weighting_curves is None:
        weighting_curves = []

    # set number of cores
    cpu_count = multiprocessing.cpu_count()
    if n_cores is None:
        n_cores = max(1, multiprocessing.cpu_count() - 1)
    if n_cores > cpu_count:
        n_cores = max(1, multiprocessing.cpu_count() - 1)
        logger.error(f"More cores then cpus requested, reducing cores to '{n_cores}'")

    console.rule("Start optimization", align="left", style="white")
    console.log(f"Running {n_cores} workers")
    if size < n_cores:
        logger.warning(
            f"Less simulations then cores: '{size} < {n_cores}', "
            f"increasing number of simulations to '{n_cores}'."
        )
        size = n_cores

    opt_result: OptimizationResult
    if serial:
        # serial parameter fitting
        saved_args.pop("n_cores")
        saved_args.pop("serial")
        kwargs = saved_args.pop("kwargs")
        opt_result = _run_optimization_serial(**saved_args, **kwargs)

    else:
        # parallel parameter fitting
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
                "algorithm": algorithm,
                "residual": residual,
                "loss_function": loss_function,
                "weighting_curves": weighting_curves,
                "weighting_points": weighting_points,
                "absolute_tolerance": absolute_tolerance,
                "relative_tolerance": relative_tolerance,
                "variable_step_size": variable_step_size,
                "seed": seeds[k],
                **kwargs,
            }
            args_list.append(d)

        # worker pool
        with multiprocessing.Pool(processes=n_cores) as pool:
            opt_results: list[OptimizationResult] = pool.map(worker, args_list)

        # combine simulation results
        opt_result = OptimizationResult.combine(opt_results)

    console.rule("FINISHED OPTIMIZATION", align="left", style="white")
    return opt_result


def worker(kwargs) -> OptimizationResult:
    """Worker for running optimization problem."""
    lock.acquire()
    try:
        logger.info(f"worker <{os.getpid()}> running optimization ...")
    finally:
        lock.release()

    return _run_optimization_serial(**kwargs)  # type: ignore


def _run_optimization_serial(
    problem: OptimizationProblem,
    size: int = 5,
    algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
    residual: ResidualType = ResidualType.ABSOLUTE,
    loss_function: LossFunctionType = LossFunctionType.LINEAR,
    weighting_curves: list[WeightingCurvesType] = None,
    weighting_points: WeightingPointsType = WeightingPointsType.NO_WEIGHTING,
    seed: Optional[int] = None,
    variable_step_size: bool = True,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-6,
    **kwargs,
) -> OptimizationResult:
    """Run the given optimization problem in a serial fashion.

    This function should not be called directly, but the 'run_optimization'
    should be used for executing simulations.
    See run_optimization for more detailed documentation.

    :param problem: uninitialized problem to optimize (pickable)
    :param size: integer number of optimizations
    :param algorithm: optimization algorithm to use
    :param residual: handling of residuals
    :param loss_function: loss function for handling outliers/residual transformation
    :param weighting_curves: list of options for weighting curves (fit mappings)
    :param weighting_points: weighting of points
    :param seed: integer random seed (for sampling of parameters)
    :param absolute_tolerance: absolute tolerance of simulator
    :param relative_tolerance: relative tolerance of simulator
    :param variable_step_size: use variable step size in solver
    :param kwargs: additional arguments for optimizer, e.g. xtol
    :return: OptimizationResult
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
