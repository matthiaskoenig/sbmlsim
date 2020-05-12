import multiprocessing

from sbmlsim.fit.mpfit import run_optimization_parallel
from sbmlsim.fit.fit import analyze_optimization

from sbmlsim.examples.experiments.midazolam import MIDAZOLAM_PATH
from sbmlsim.examples.experiments.midazolam.fitting_problems import op_mid1oh_iv, op_mandema1992

RESULTS_PATH = MIDAZOLAM_PATH / "results"


if __name__ == "__main__":
    opt_result = run_optimization_parallel(problem=op_mid1oh_iv, size=10, n_cores=3)
    analyze_optimization(opt_result)
