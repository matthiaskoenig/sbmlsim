"""
Defines the parameter fitting problems
"""
from pathlib import Path
from typing import Tuple

from sbmlsim.examples.experiments.midazolam import MIDAZOLAM_PATH
from sbmlsim.examples.experiments.midazolam.fitting_problems import (
    op_mandema1992,
    op_mid1oh_iv,
)
from sbmlsim.fit.analysis import OptimizationResult
from sbmlsim.fit.fit import process_optimization_result, run_optimization
from sbmlsim.fit.optimization import (
    FittingType,
    OptimizationProblem,
    OptimizerType,
    ResidualType,
    SamplingType,
    WeightingLocalType,
)


def fit_lsq(
    problem_factory, **fit_kwargs
) -> Tuple[OptimizationResult, OptimizationProblem]:
    """Local least square fitting."""
    problem: OptimizationProblem = problem_factory()
    print(problem)
    opt_res = run_optimization(
        problem=problem,
        size=5,
        seed=1236,
        optimizer=OptimizerType.LEAST_SQUARE,
        **fit_kwargs
    )
    return opt_res, problem


def fit_de(
    problem_factory, **fit_kwargs
) -> Tuple[OptimizationResult, OptimizationProblem]:
    """Global differential evolution fitting."""
    problem = problem_factory()
    opt_res = run_optimization(problem=problem, size=1, seed=1234, **fit_kwargs)
    return opt_res, problem


if __name__ == "__main__":
    fit_kwargs = {
        "fitting_type": FittingType.ABSOLUTE_VALUES,
        "residual_type": ResidualType.ABSOLUTE_NORMED_RESIDUALS,
        "weighting_local": WeightingLocalType.ABSOLUTE_ONE_OVER_WEIGHTING,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    }

    output_path = Path(__file__).parent / "results_fit"
    opt_res_lq, problem = fit_lsq(problem_factory=op_mid1oh_iv, **fit_kwargs)
    process_optimization_result(
        opt_res_lq, problem=problem, output_path=output_path, **fit_kwargs
    )

    opt_res_de, problem = fit_de(problem_factory=op_mid1oh_iv, **fit_kwargs)
    process_optimization_result(
        opt_res_de, problem=problem, output_path=output_path, **fit_kwargs
    )
