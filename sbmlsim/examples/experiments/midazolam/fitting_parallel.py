from typing import Tuple

from sbmlsim.fit.mpfit import run_optimization_parallel
from sbmlsim.fit.fit import analyze_optimization, OptimizationResult
from sbmlsim.fit.optimization import OptimizationProblem, SamplingType, \
    OptimizerType, WeightingLocalType, ResidualType
from sbmlsim.examples.experiments.midazolam import MIDAZOLAM_PATH
from sbmlsim.examples.experiments.midazolam.fitting_problems import op_mid1oh_iv, op_mandema1992, op_kupferschmidt1995

RESULTS_PATH = MIDAZOLAM_PATH / "results"


def fit_lsq(problem_factory,
            weighting_local: WeightingLocalType,
            residual_type: ResidualType) -> Tuple[OptimizationResult, OptimizationProblem]:
    """Local least square fitting."""
    problem = problem_factory()
    opt_res = run_optimization_parallel(
        problem=problem, size=50, seed=1236, n_cores=10,
        optimizer=OptimizerType.LEAST_SQUARE,
        weighting_local=weighting_local,
        residual_type=residual_type,

        # parameters for least square optimization
        sampling=SamplingType.LOGUNIFORM_LHS,
        diff_step=0.05
    )
    return opt_res, problem


def fit_de(problem_factory,
           weighting_local: WeightingLocalType,
           residual_type: ResidualType
           ) -> Tuple[OptimizationResult, OptimizationProblem]:
    """Global differential evolution fitting."""
    problem = problem_factory()
    opt_res = run_optimization_parallel(
        problem=problem, size=10, seed=1234, n_cores=10,
        optimizer=OptimizerType.DIFFERENTIAL_EVOLUTION,
        weighting_local=weighting_local,
        residual_type=residual_type,
    )
    return opt_res, problem


if __name__ == "__main__":

    # fit_id = "mid1oh_iv"
    fit_id = "kupferschmidt1995"

    fit_path = MIDAZOLAM_PATH / "results_fit"
    fit_path_lsq = fit_path / fit_id / "lsq"
    fit_path_de = fit_path / fit_id / "de"
    for p in [fit_path_de, fit_path_lsq]:
        if not p.exists():
            p.mkdir(parents=True)

    # TODO: save problem (serializable part & results)
    # json_str = opt_res_lq.to_json()
    # print(json_str)
    # opt_res2 = OptimizationResult.from_json(json_str)
    # analyze_optimization(opt_res2, problem=problem)

    if fit_id == "mid1oh_iv":
        problem_factory = op_mid1oh_iv
    elif fit_id == "kupferschmidt1995":
        problem_factory = op_kupferschmidt1995

    fit_kwargs = {
        "weighting_local": WeightingLocalType.ABSOLUTE_ONE_OVER_WEIGHTING,
        "residual_type": ResidualType.ABSOLUTE_NORMED_RESIDUALS
    }

    if 0:
        opt_res_lsq, problem = fit_lsq(problem_factory, **fit_kwargs)
        analyze_optimization(opt_res_lsq, problem=problem,
                             output_path=fit_path_lsq, **fit_kwargs)

    if 1:

        opt_res_de, problem = fit_de(problem_factory, **fit_kwargs)
        analyze_optimization(opt_res_de, problem=problem,
                         output_path=fit_path_de, **fit_kwargs)
