"""
Defines the parameter fitting problems
"""
from sbmlsim.fit import FitExperiment, FitParameter
from sbmlsim.fit.fit import run_optimization
from sbmlsim.fit.optimization import OptimizationProblem, SamplingType, OptimizerType
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.examples.experiments.midazolam.experiments.mandema1992 import Mandema1992

from sbmlsim.examples.experiments.midazolam import MIDAZOLAM_PATH
RESULTS_PATH = MIDAZOLAM_PATH / "results"
DATA_PATH = MIDAZOLAM_PATH / "data"


exp_kwargs = {
    "base_path": MIDAZOLAM_PATH,
    "data_path": DATA_PATH,
}

op_mid1oh_iv = OptimizationProblem(
    opid="mid1oh_iv",
    fit_experiments=[
        FitExperiment(experiment=Mandema1992, mappings=["fm4"])
    ],
    fit_parameters=[
        # distribution parameters
        FitParameter(parameter_id="ftissue_mid1oh", start_value=1.0,
                     lower_bound=1, upper_bound=1E5,
                     unit="liter/min"),
        FitParameter(parameter_id="fup_mid1oh", start_value=0.1,
                     lower_bound=0.01, upper_bound=0.3,
                     unit="dimensionless"),
        # mid1oh kinetics
        FitParameter(parameter_id="KI__MID1OHEX_Vmax", start_value=100,
                     lower_bound=1E-1, upper_bound=1E4,
                     unit="mmole/min"),
    ],
    **exp_kwargs
)


if __name__ == "__main__":
    if True:
        opt_res1 = run_optimization(
            op_mid1oh_iv, size=50, seed=1236,
            output_path=RESULTS_PATH,

           optimizer=OptimizerType.LEAST_SQUARE,
           sampling=SamplingType.LOGUNIFORM_LHS,
           diff_step=0.05,
           # jac='3-point', gtol=1e-10, xtol=1e-12,
        )


    if False:
        run_optimization(op_mid1oh_iv, size=3, seed=1234,
                       output_path=RESULTS_PATH,
                       optimizer=OptimizerType.DIFFERENTIAL_EVOLUTION,
        )
