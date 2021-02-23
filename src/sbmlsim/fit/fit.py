"""Run optimizations."""

import logging
from pathlib import Path

from sbmlsim.fit.analysis import OptimizationResult
from sbmlsim.fit.optimization import OptimizationAnalysis, OptimizationProblem
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.utils import timeit


logger = logging.getLogger(__name__)


@timeit
def run_optimization(
    problem: OptimizationProblem,
    size: int = 5,
    seed: int = None,
    verbose=False,
    **kwargs,
) -> OptimizationResult:
    """Run the given optimization problem.

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
    if "n_cores" in kwargs:
        # remove parallel arguments
        logger.warning(
            "Parameter 'n_cores' does not have any effect in serial " "optimization."
        )
        kwargs.pop("n_cores")

    # here the additional information must be injected
    fitting_type = kwargs["fitting_type"]
    weighting_local = kwargs["weighting_local"]
    residual_type = kwargs["residual_type"]

    # initialize problem, which calculates errors
    problem.initialize(
        fitting_type=fitting_type,
        weighting_local=weighting_local,
        residual_type=residual_type,
    )

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


def process_optimization_result(
    opt_result: OptimizationResult,
    output_path: Path,
    problem: OptimizationProblem = None,
    show_plots=True,
    fitting_type=None,
    weighting_local=None,
    residual_type=None,
    variable_step_size=True,
    absolute_tolerance=1e-6,
    relative_tolerance=1e-6,
):
    """Process the optimization results.

    Creates reports and stores figures and results.
    """
    results_path = output_path / opt_result.sid
    if not results_path.exists():
        logger.warning(f"create output directory: '{results_path}'")
        results_path.mkdir(parents=True, exist_ok=True)

    problem_info = ""
    if problem:
        # FIXME: problem not initialized on multi-core and no simulator is assigned.
        # This should happen automatically, to ensure correct behavior
        problem.initialize(
            fitting_type=fitting_type,
            weighting_local=weighting_local,
            residual_type=residual_type,
        )
        problem.set_simulator(simulator=SimulatorSerial())
        problem.variable_step_size = variable_step_size
        problem.absolute_tolerance = absolute_tolerance
        problem.relative_tolerance = relative_tolerance

        problem_info = problem.report(
            path=None,
            print_output=False,
        )

    # write report
    result_info = opt_result.report(
        path=None,
        print_output=True,
    )
    info = problem_info + result_info
    with open(results_path / "00_report.txt", "w") as f_report:
        f_report.write(info)

    opt_result.to_json(path=results_path / "01_optimization_result.json")
    opt_result.to_tsv(path=results_path / "01_optimization_result.tsv")

    if opt_result.size > 1:
        opt_result.plot_waterfall(
            path=results_path / "02_waterfall.svg", show_plots=show_plots
        )
    opt_result.plot_traces(path=results_path / "02_traces.svg", show_plots=show_plots)

    # plot top fit
    if problem:
        xopt = opt_result.xopt
        optimization_analyzer = OptimizationAnalysis(optimization_problem=problem)

        df_costs = optimization_analyzer.plot_costs(
            x=xopt, path=results_path / "03_cost_improvement.svg", show_plots=show_plots
        )
        df_costs.to_csv(results_path / "03_cost_improvement.tsv", sep="\t", index=False)

        optimization_analyzer.plot_fits(
            x=xopt, path=results_path / "05_fits.svg", show_plots=show_plots
        )
        optimization_analyzer.plot_residuals(
            x=xopt, output_path=results_path, show_plots=show_plots
        )

    if opt_result.size > 1:
        opt_result.plot_correlation(
            path=results_path / "04_parameter_correlation", show_plots=show_plots
        )

    # TODO: overall HTML report for simple overview
