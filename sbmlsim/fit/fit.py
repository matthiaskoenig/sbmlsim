from pathlib import Path

from sbmlsim.simulator import SimulatorSerial
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.analysis import OptimizationResult
from sbmlsim.utils import timeit


@timeit
def run_optimization(
        problem: OptimizationProblem,
        size: int = 5, seed: int = None, verbose=False, **kwargs) -> OptimizationResult:
    """Runs the given optimization problem.

    This changes the internal state of the optimization problem
    and provides simple access to parameters and costs after
    optimization.

    :param size: number of optimization with different start values
    :param seed: random seed (for sampling of parameters)
    :param plot_results: should standard plots be generated
    :param output_path: path (directory) to store results
    :param kwargs: additional arguments for optimizer, e.g. xtol
    :return: list of optimization results
    """
    # initialize problem
    problem.initialize()

    # new simulator instance
    # FIXME: handle tolerances here
    simulator = SimulatorSerial()
    problem.set_simulator(simulator)

    # optimize
    fits, trajectories = problem.optimize(size=size, seed=seed, verbose=verbose, **kwargs)

    # process results and plots
    return OptimizationResult(parameters=problem.parameters, fits=fits, trajectories=trajectories)


def analyze_optimization(opt_result: OptimizationResult,
                         output_path: Path=None, problem: OptimizationProblem=None,
                         show_plots=True,
                         weighting_local=None, weighting_global=None,
                         variable_step_size=True, absolute_tolerance=1E-6, relative_tolerance=1E-6):
    # write report (additional folders based on runs)
    
    opt_result.report(output_path=output_path)

    # FIXME: save and load the results
    # opt_result.save(output_path=output_path)

    if opt_result.size > 1:
        opt_result.plot_waterfall(output_path=output_path, show_plots=show_plots)
    opt_result.plot_traces(output_path=output_path, show_plots=show_plots)

    # plot top fit
    if problem:
        # FIMXE: problem references not initialized on multi-core and
        # don't have a simulator yet
        # FIXME: tolerances
        problem.initialize()
        problem.set_simulator(simulator=SimulatorSerial())
        problem.variable_step_size = variable_step_size
        problem.absolute_tolerance = absolute_tolerance
        problem.relative_tolerance = relative_tolerance
        problem.weighting_local = weighting_local
        problem.weighting_global = weighting_global

        problem.report(output_path=output_path)
        problem.plot_fits(x=opt_result.xopt, output_path=output_path, show_plots=show_plots)
        problem.plot_costs(x=opt_result.xopt, output_path=output_path, show_plots=show_plots)
        problem.plot_residuals(x=opt_result.xopt, output_path=output_path, show_plots=show_plots)

    opt_result.plot_correlation(output_path=output_path, show_plots=show_plots)

