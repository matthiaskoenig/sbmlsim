from typing import Tuple

from sbmlsim.fit.mpfit import run_optimization_parallel
from sbmlsim.fit.fit import analyze_optimization, OptimizationResult
from sbmlsim.fit.optimization import OptimizationProblem, SamplingType, \
    OptimizerType, WeightingLocalType, WeightingGlobalType
from sbmlsim.examples.experiments.midazolam import MIDAZOLAM_PATH
from sbmlsim.examples.experiments.midazolam.fitting_problems import op_mid1oh_iv, op_mandema1992

RESULTS_PATH = MIDAZOLAM_PATH / "results"


def fitlq_mid1ohiv() -> Tuple[OptimizationResult, OptimizationProblem]:
    """Local least square fitting."""
    problem = op_mid1oh_iv()
    opt_res = run_optimization_parallel(
        problem=problem, size=50, seed=1236, n_cores=10,
        optimizer=OptimizerType.LEAST_SQUARE,
        weighting_local=WeightingLocalType.NO_WEIGHTING,
        weighting_global=WeightingGlobalType.NO_WEIGHTING,

        # parameters for least square optimization
        sampling=SamplingType.LOGUNIFORM_LHS,
        diff_step=0.05
    )
    return opt_res, problem


def fitde_mid1ohiv() -> Tuple[OptimizationResult, OptimizationProblem]:
    """Global differential evolution fitting."""
    problem = op_mid1oh_iv()
    opt_res = run_optimization_parallel(
        problem=problem, size=10, seed=1234, n_cores=10,
        optimizer=OptimizerType.DIFFERENTIAL_EVOLUTION,
        weighting_local=WeightingLocalType.NO_WEIGHTING,
        weighting_global=WeightingGlobalType.NO_WEIGHTING
    )
    return opt_res, problem


if __name__ == "__main__":
    fit_path = MIDAZOLAM_PATH / "results_fit"
    fit_path_lq = fit_path / "mid1ohiv" / "lq"
    fit_path_de = fit_path / "mid1ohiv" / "de"
    for p in [fit_path_de, fit_path_lq]:
        if not p.exists():
            p.mkdir(parents=True)

    # TODO: save problem (serializable part & results)
    # json_str = opt_res_lq.to_json()
    # print(json_str)
    # opt_res2 = OptimizationResult.from_json(json_str)
    # analyze_optimization(opt_res2, problem=problem)

    opt_res_lq, problem = fitlq_mid1ohiv()
    analyze_optimization(opt_res_lq, problem=problem,
                         output_path=fit_path_lq)

    opt_res_de, problem = fitde_mid1ohiv()
    analyze_optimization(opt_res_de, problem=problem,
                         output_path=fit_path_de)
