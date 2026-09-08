"""HCTZ parameter fitting.

Run a parameter fit of the HCTZ model against the data of the studies:

    python -m examples.hctz.fitting.fitting --runs=4 --cores=2 --seed=1234 \
        --method=LSQ --strategy=ALL --subset=PK --name=PK_LSQ_ALL

The results, figures and the HTML report are written into `results/fit` in the
working directory.
"""

import argparse
import itertools
import logging
from enum import StrEnum
from pathlib import Path
from typing import Any

from examples.hctz import DATA_PATH, HCTZ_PATH
from examples.hctz.fitting.fit_experiments import f_fitexp_pk, f_fitexp_pkiv
from examples.hctz.fitting.parameters import parameters_pk
from sbmlsim import log
from sbmlsim.console import console
from sbmlsim.fit import FitExperiment, FitParameter
from sbmlsim.fit.analysis import OptimizationAnalysis
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    LossFunctionType,
    OptimizationAlgorithmType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization
from sbmlsim.fit.sampling import SamplingType

logger = logging.getLogger(__name__)

# the settings are shared by the optimization and its analysis
fit_kwargs: dict[str, Any] = {
    # optimization settings
    "residual": ResidualType.NORMALIZED,
    "loss_function": LossFunctionType.LINEAR,
    "weighting_curves": [
        WeightingCurvesType.MAPPING,  # user defined weights
        WeightingCurvesType.POINTS,  # number of points
    ],
    # mappings without errors are weighted with CV=0.5
    "weighting_points": WeightingPointsType.ERROR_WEIGHTING,
    # additional integrator settings
    "variable_step_size": True,
    "relative_tolerance": 1e-6,
    "absolute_tolerance": 1e-6,
}


class OptimizationStrategy(StrEnum):
    """Strategy for fitting.

    Either fit all experiments together or individually.
    """

    ALL = "ALL"  # one parameter set for all experiments
    SINGLE = "SINGLE"  # one parameter set per experiment


class FitMethod(StrEnum):
    """Method for fitting."""

    LSQ = "LSQ"
    DE = "DE"


class FitExperimentSubset(StrEnum):
    """Subset of fit experiments for fitting."""

    PK = "PK"  # all pharmacokinetics data
    PKIV = "PKIV"  # pharmacokinetics data of the iv studies


FIT_EXPERIMENTS = {
    FitExperimentSubset.PK: f_fitexp_pk,
    FitExperimentSubset.PKIV: f_fitexp_pkiv,
}
FIT_PARAMETERS = {
    FitExperimentSubset.PK: parameters_pk,
    FitExperimentSubset.PKIV: parameters_pk,
}


def create_optimization_problem(
    fit_experiments: list[FitExperiment], opid: str, parameters: list[FitParameter]
) -> OptimizationProblem:
    """Create the optimization problem for the given experiments and parameters."""
    return OptimizationProblem(
        opid=opid,
        fit_experiments=fit_experiments,
        fit_parameters=parameters,
        base_path=HCTZ_PATH,
        data_path=DATA_PATH,
    )


def fitlsq(
    op: OptimizationProblem, seed: int, **kwargs
) -> tuple[OptimizationResult, OptimizationProblem]:
    """Local least square fitting."""
    opt_res = run_optimization(
        problem=op,
        seed=seed,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        # parameters for least square optimization
        sampling=SamplingType.LOGUNIFORM_LHS,
        diff_step=0.05,
        **kwargs,
    )
    return opt_res, op


def fitde(
    op: OptimizationProblem, seed: int, **kwargs
) -> tuple[OptimizationResult, OptimizationProblem]:
    """Global differential evolution fitting."""
    opt_res = run_optimization(
        problem=op,
        seed=seed,
        algorithm=OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION,
        **kwargs,
    )
    return opt_res, op


def fit_hctz(
    optimization_strategy: OptimizationStrategy,
    fit_method: FitMethod,
    fit_experiments: list[FitExperiment],
    parameters: list[FitParameter],
    n_cores: int,
    n_optimizations: int,
    seed: int,
) -> dict[str, tuple[OptimizationResult, OptimizationProblem]]:
    """Run the optimization for the given experiments and parameters.

    Returns:
        The optimization result and problem by optimization id.
    """

    def fit_op(
        op: OptimizationProblem,
    ) -> tuple[OptimizationResult, OptimizationProblem]:
        """Run a single optimization problem."""
        f_fit = fitlsq if fit_method == FitMethod.LSQ else fitde
        return f_fit(op, seed=seed, size=n_optimizations, n_cores=n_cores, **fit_kwargs)

    results: dict[str, tuple[OptimizationResult, OptimizationProblem]] = {}
    if optimization_strategy == OptimizationStrategy.SINGLE:
        # fit all experiments individually
        for fit_exp in fit_experiments:
            opid = fit_exp.experiment_class.__name__
            results[opid] = fit_op(
                create_optimization_problem(
                    fit_experiments=[fit_exp], opid=opid, parameters=parameters
                )
            )
    else:
        # fit all experiments together
        results["all"] = fit_op(
            create_optimization_problem(
                fit_experiments=fit_experiments, opid="all", parameters=parameters
            )
        )

    return results


def get_fit_experiments(
    fit_subset: FitExperimentSubset, study_ids: list[str] | None = None
) -> list[FitExperiment]:
    """Create the subset of fit experiments for the given subset and studies."""
    fitexp_dict = FIT_EXPERIMENTS[fit_subset]()

    if study_ids:
        fit_experiments = [fitexp_dict[sid] for sid in study_ids]
    else:
        fit_experiments = list(fitexp_dict.values())

    # reduce list of lists
    return list(itertools.chain(*fit_experiments))


def get_fit_parameters(fit_subset: FitExperimentSubset) -> list[FitParameter]:
    """Create the subset of fit parameters for the given subset."""
    return FIT_PARAMETERS[fit_subset]


def op_hctz(fit_subset: FitExperimentSubset) -> OptimizationProblem:
    """Get the uninitialized optimization problem of a subset of the data.

    Args:
        fit_subset: subset of the fit experiments and parameters.

    Returns:
        The optimization problem, `run_optimization` initializes and runs it.
    """
    return create_optimization_problem(
        fit_experiments=get_fit_experiments(fit_subset=fit_subset),
        opid=f"hctz_{fit_subset.value.lower()}",
        parameters=get_fit_parameters(fit_subset=fit_subset),
    )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse the arguments of the fitting script."""
    parser = argparse.ArgumentParser(
        prog="fit_hctz", description="Parameter fitting of the HCTZ model."
    )
    parser.add_argument(
        "-c", "--cores", type=int, default=1, help="number of cores for the fitting"
    )
    parser.add_argument(
        "-r", "--runs", type=int, default=4, help="number of optimization runs"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=1234, help="seed of the optimization"
    )
    parser.add_argument("-n", "--name", default="hctz", help="name of the optimization")
    parser.add_argument(
        "-m",
        "--method",
        type=FitMethod,
        choices=list(FitMethod),
        default=FitMethod.LSQ,
        help="optimization method",
    )
    parser.add_argument(
        "-t",
        "--strategy",
        type=OptimizationStrategy,
        choices=list(OptimizationStrategy),
        default=OptimizationStrategy.ALL,
        help="optimization strategy",
    )
    parser.add_argument(
        "-x",
        "--subset",
        type=FitExperimentSubset,
        choices=list(FitExperimentSubset),
        default=FitExperimentSubset.PK,
        help="subset of the fit experiments",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("results") / "fit",
        help="directory for the results of the fit",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    """Run the parameter fitting of the HCTZ model."""
    options = parse_args(args)
    log.enable_rich_logging()

    console.rule(":wrench: FIT HCTZ :wrench:", align="left", style="white")
    for key in ["cores", "runs", "seed", "name", "method", "strategy", "subset"]:
        console.print(f"{key:<20}: {getattr(options, key)}")

    output_dir: Path = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule("Parameters", align="left", style="white")
    parameters = get_fit_parameters(fit_subset=options.subset)
    console.print(FitParameter.parameters_to_df(parameters))

    fit_experiments = get_fit_experiments(fit_subset=options.subset)
    results = fit_hctz(
        fit_experiments=fit_experiments,
        parameters=parameters,
        optimization_strategy=options.strategy,
        fit_method=options.method,
        n_cores=options.cores,
        n_optimizations=options.runs,
        seed=options.seed,
    )

    for opt_result, op in results.values():
        opt_analysis = OptimizationAnalysis(
            opt_result=opt_result,
            op=op,
            output_name=options.name,
            output_dir=output_dir,
            show_plots=False,
            show_titles=False,
            **fit_kwargs,
        )
        opt_analysis.run()


if __name__ == "__main__":
    main()
