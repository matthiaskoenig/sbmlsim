"""
Defines the parameter fitting problems
"""
from sbmlsim.fit.fit import run_optimization, analyze_optimization
from sbmlsim.fit.optimization import SamplingType, OptimizerType, WeightingType

from sbmlsim.examples.experiments.midazolam.fitting_problems import op_mid1oh_iv, op_mandema1992
from sbmlsim.examples.experiments.midazolam import MIDAZOLAM_PATH
RESULTS_PATH = MIDAZOLAM_PATH / "results"


if __name__ == "__main__":
    if True:
        opt_result = run_optimization(
            op_mid1oh_iv, size=50, seed=1236,
            optimizer=OptimizerType.LEAST_SQUARE,
            sampling=SamplingType.LOGUNIFORM_LHS,
            weighting=WeightingType.ONE_OVER_WEIGHTING,
            diff_step=0.05
        )
    else:
        opt_result = run_optimization(
            op_mid1oh_iv, size=1, seed=1234,
            optimizer=OptimizerType.DIFFERENTIAL_EVOLUTION
        )

    analyze_optimization(opt_result)
