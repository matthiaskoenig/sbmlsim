"""
Defines the parameter fitting problems
"""
from typing import Tuple
from sbmlsim.fit.fit import run_optimization, analyze_optimization
from sbmlsim.fit.optimization import SamplingType, OptimizerType, \
    WeightingLocalType, ResidualType, OptimizationProblem
from sbmlsim.fit.analysis import OptimizationResult

from sbmlsim.examples.experiments.midazolam.fitting_problems import op_mid1oh_iv, op_mandema1992
from sbmlsim.examples.experiments.midazolam import MIDAZOLAM_PATH
RESULTS_PATH = MIDAZOLAM_PATH / "results"


def fit_lsq(problem_factory) -> Tuple[OptimizationResult, OptimizationProblem]:
    """Local least square fitting."""
    problem = problem_factory()
    opt_res = run_optimization(
        problem=problem, size=20, seed=1236,
        optimizer=OptimizerType.LEAST_SQUARE,
        weighting_local=WeightingLocalType.ABSOLUTE_ONE_OVER_WEIGHTING,
        residual_type=ResidualType.ABSOLUTE_NORMED_RESIDUALS,
    )
    return opt_res, problem


def fit_de(problem_factory) -> Tuple[OptimizationResult, OptimizationProblem]:
    """Global differential evolution fitting."""
    problem = problem_factory()
    opt_res = run_optimization(
        problem=problem, size=1, seed=1234,
        optimizer=OptimizerType.DIFFERENTIAL_EVOLUTION,
        weighting_local=WeightingLocalType.ABSOLUTE_ONE_OVER_WEIGHTING,
        residual_type=ResidualType.ABSOLUTE_NORMED_RESIDUALS,
    )
    return opt_res, problem


if __name__ == "__main__":
    opt_res_lq, problem = fit_lsq(problem_factory=op_mid1oh_iv)
    analyze_optimization(opt_res_lq, problem=problem)

    opt_res_de, problem = fit_de(problem_factory=op_mid1oh_iv)
    analyze_optimization(opt_res_de, problem=problem)
