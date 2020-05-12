from sbmlsim.fit.mpfit import run_optimization_parallel
from sbmlsim.fit.fit import analyze_optimization

from sbmlsim.fit.optimization import SamplingType, OptimizerType, WeightingType
from sbmlsim.examples.experiments.midazolam import MIDAZOLAM_PATH
from sbmlsim.examples.experiments.midazolam.fitting_problems import op_mid1oh_iv, op_mandema1992

RESULTS_PATH = MIDAZOLAM_PATH / "results"


if __name__ == "__main__":
    if False:
        opt_result = run_optimization_parallel(
            n_cores=10,
            problem=op_mid1oh_iv, size=20,
            verbose=False,
            optimizer=OptimizerType.LEAST_SQUARE,
            sampling=SamplingType.LOGUNIFORM_LHS,
            weighting=WeightingType.NO_WEIGHTING,
            diff_step=0.05,
        )
    else:
        opt_result = run_optimization_parallel(
            problem=op_mid1oh_iv, size=20, seed=1234,
            optimizer=OptimizerType.DIFFERENTIAL_EVOLUTION,
            weighting=WeightingType.NO_WEIGHTING,
        )
    analyze_optimization(opt_result)
