"""
Defines the parameter fitting problems
"""
from sbmlsim.fit import FitExperiment, FitParameter, run_optimization
from sbmlsim.fit.optimization import OptimizationProblem, SamplingType, OptimizerType
from sbmlsim.simulator import SimulatorSerial

from sbmlsim.examples.experiments.midazolam.experiments.mandema1992 import Mandema1992

from sbmlsim.examples.experiments.midazolam import MIDAZOLAM_PATH
RESULTS_PATH = MIDAZOLAM_PATH / "results"
DATA_PATH = MIDAZOLAM_PATH / "data"
simulator = SimulatorSerial()

exp_kwargs = {
    "simulator": simulator,
    "base_path": MIDAZOLAM_PATH,
    "data_path": DATA_PATH,
}

op_mandema1992 = OptimizationProblem(
    opid="mandema1992",
    fit_experiments=[
        # FitExperiment(experiment=Mandema1992, mappings=["fm1"]),
        FitExperiment(experiment=Mandema1992, mappings=["fm1", "fm3", "fm4"]),
    ],
    fit_parameters=[
        # liver
        FitParameter(parameter_id="LI__MIDIM_Vmax", start_value=0.1,
                     lower_bound=1E-3, upper_bound=1E6,
                     unit="mmole_per_min"),
        FitParameter(parameter_id="LI__MID1OHEX_Vmax", start_value=0.1,
                     lower_bound=1E-3, upper_bound=1E6,
                     unit="mmole_per_min"),
        FitParameter(parameter_id="LI__MIDOH_Vmax", start_value=100,
                     lower_bound=10, upper_bound=200, unit="mmole_per_min"),
        # kidneys
        FitParameter(parameter_id="KI__MID1OHEX_Vmax", start_value=100,
                     lower_bound=1E-1, upper_bound=1E4,
                     unit="mmole/min"),

        # distribution
        FitParameter(parameter_id="ftissue_mid", start_value=2000,
                      lower_bound=1, upper_bound=1E5,
                      unit="liter/min"),
        FitParameter(parameter_id="fup_mid", start_value=0.1,
                      lower_bound=0.05, upper_bound=0.3,
                      unit="dimensionless"),
        # distribution parameters
        FitParameter(parameter_id="ftissue_mid1oh", start_value=1.0,
                     lower_bound=1, upper_bound=1E5,
                     unit="liter/min"),
        FitParameter(parameter_id="fup_mid1oh", start_value=0.1,
                     lower_bound=0.01, upper_bound=0.3,
                     unit="dimensionless"),
    ],
    **exp_kwargs
)


if __name__ == "__main__":
    if False:
        opt_res = run_optimization(op_mandema1992, size=50, seed=2345,
                   output_path=RESULTS_PATH,
                   optimizer=OptimizerType.LEAST_SQUARE,
                   sampling=SamplingType.LOGUNIFORM_LHS,
                   diff_step=0.05,
                   # jac='2-point',
                   # gtol=1e-10, xtol=1e-12,
        )

    if True:
        run_optimization(op_mandema1992, size=1, seed=1234,
                   output_path=RESULTS_PATH,
                   optimizer=OptimizerType.DIFFERENTIAL_EVOLUTION,
        )
