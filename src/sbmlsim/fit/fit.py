from pathlib import Path

from sbmlsim.fit.analysis import OptimizationResult
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.utils import timeit


@timeit
def run_optimization(
    problem: OptimizationProblem,
    size: int = 5,
    seed: int = None,
    verbose=False,
    **kwargs
) -> OptimizationResult:
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
    # here the additional information must be injected
    weighting_local = kwargs["weighting_local"]
    residual_type = kwargs["residual_type"]

    # initialize problem, which calculates errors
    problem.initialize(weighting_local=weighting_local, residual_type=residual_type)

    # new simulator instance
    simulator = SimulatorSerial(**kwargs)  # sets tolerances
    problem.set_simulator(simulator)

    # optimize
    fits, trajectories = problem.optimize(
        size=size, seed=seed, verbose=verbose, **kwargs
    )

    # process results and plots
    return OptimizationResult(
        parameters=problem.parameters, fits=fits, trajectories=trajectories
    )


def analyze_optimization(
    opt_result: OptimizationResult,
    output_path: Path = None,
    problem: OptimizationProblem = None,
    show_plots=True,
    weighting_local=None,
    residual_type=None,
    variable_step_size=True,
    absolute_tolerance=1e-6,
    relative_tolerance=1e-6,
):
    # write report (additional folders based on runs)

    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    opt_result.report(output_path=output_path)

    # FIXME: save and load the results
    # opt_result.save(output_path=output_path)

    if opt_result.size > 1:
        opt_result.plot_waterfall(output_path=output_path, show_plots=show_plots)
    opt_result.plot_traces(output_path=output_path, show_plots=show_plots)

    # plot top fit
    if problem:
        # FIMXE: problem references not initialized on multi-core and don't have a simulator yet

        problem.initialize(weighting_local=weighting_local, residual_type=residual_type)
        # FIXME: tolerances
        problem.set_simulator(simulator=SimulatorSerial())
        problem.variable_step_size = variable_step_size
        problem.absolute_tolerance = absolute_tolerance
        problem.relative_tolerance = relative_tolerance

        problem.report(output_path=output_path)
        problem.plot_fits(
            x=opt_result.xopt, output_path=output_path, show_plots=show_plots
        )
        problem.plot_costs(
            x=opt_result.xopt, output_path=output_path, show_plots=show_plots
        )
        problem.plot_residuals(
            x=opt_result.xopt, output_path=output_path, show_plots=show_plots
        )

    opt_result.plot_correlation(output_path=output_path, show_plots=show_plots)
