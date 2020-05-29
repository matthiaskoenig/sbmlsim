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


def fitlq_mid1ohiv() -> Tuple[OptimizationResult, OptimizationProblem]:
    """Local least square fitting."""
    problem = op_mid1oh_iv()
    opt_res = run_optimization(
        problem=problem, size=20, seed=1236,
        optimizer=OptimizerType.LEAST_SQUARE,
        weighting_local=WeightingLocalType.ONE_OVER_WEIGHTING,
        weighting_global=ResidualType.NO_WEIGHTING,
        # parameters for least square optimization
        sampling=SamplingType.LOGUNIFORM_LHS,
        diff_step=0.05
    )
    return opt_res, problem


def fitde_mid1ohiv() -> Tuple[OptimizationResult, OptimizationProblem]:
    """Global differential evolution fitting."""
    problem = op_mid1oh_iv()
    opt_res = run_optimization(
        problem=problem, size=1, seed=1234,
        optimizer=OptimizerType.DIFFERENTIAL_EVOLUTION,
        weighting_local=WeightingLocalType.ONE_OVER_WEIGHTING,
        weighting_global=ResidualType.NO_WEIGHTING,
    )
    return opt_res, problem


if __name__ == "__main__":
    opt_res_lq, problem = fitlq_mid1ohiv()
    analyze_optimization(opt_res_lq, problem=problem)

    opt_res_de, problem = fitde_mid1ohiv()
    analyze_optimization(opt_res_de, problem=problem)
