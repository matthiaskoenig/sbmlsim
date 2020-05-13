from sbmlsim.fit.mpfit import run_optimization_parallel
from sbmlsim.fit.fit import analyze_optimization

from sbmlsim.fit.optimization import SamplingType, OptimizerType, WeightingType
from sbmlsim.examples.experiments.midazolam import MIDAZOLAM_PATH
from sbmlsim.examples.experiments.midazolam.fitting_problems import op_mid1oh_iv, op_mandema1992

RESULTS_PATH = MIDAZOLAM_PATH / "results"


def fitlq_mid1ohiv():
    """Local least square fitting."""
    return run_optimization_parallel(
        problem=op_mid1oh_iv(), size=50, seed=1236,
        optimizer=OptimizerType.LEAST_SQUARE,
        weighting=WeightingType.NO_WEIGHTING,
        # parameters for least square optimization
        sampling=SamplingType.LOGUNIFORM_LHS,
        diff_step=0.05
    )


def fitde_mid1ohiv():
    """Global differential evolution fitting."""
    return run_optimization_parallel(
            problem=op_mid1oh_iv(), size=10, seed=1234,
            optimizer=OptimizerType.DIFFERENTIAL_EVOLUTION,
            weighting=WeightingType.NO_WEIGHTING,
        )


if __name__ == "__main__":
    opt_res_lq = fitlq_mid1ohiv()
    analyze_optimization(opt_res_lq)

    opt_res_de = fitde_mid1ohiv()
    analyze_optimization(opt_res_de)
