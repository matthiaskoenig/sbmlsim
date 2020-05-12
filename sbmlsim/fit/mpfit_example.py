import multiprocessing

from sbmlsim.fit.fit import analyze_optimization
from sbmlsim.fit import FitExperiment, FitParameter
from sbmlsim.fit import mpfit
from sbmlsim.examples.experiments.midazolam.experiments.mandema1992 import Mandema1992

from sbmlsim.examples.experiments.midazolam import MIDAZOLAM_PATH


RESULTS_PATH = MIDAZOLAM_PATH / "results"
DATA_PATH = MIDAZOLAM_PATH / "data"


op_mid1oh_iv = {
    "opid": "mid1oh_iv",
    "base_path": MIDAZOLAM_PATH,
    "data_path": DATA_PATH,
    "fit_experiments": [
            FitExperiment(experiment=Mandema1992, mappings=["fm4"])
        ],
    "fit_parameters": [
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
        ]
}

op_mandema1992 = {
    "opid": "mandema1992",
    "base_path": MIDAZOLAM_PATH,
    "data_path": DATA_PATH,
    "fit_experiments": [
        # FitExperiment(experiment=Mandema1992, mappings=["fm1"]),
        FitExperiment(experiment=Mandema1992, mappings=["fm1", "fm3", "fm4"]),
    ],
    "fit_parameters": [
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
}


def run_mpfit():
    n_cores = multiprocessing.cpu_count()
    opt_res = mpfit.fit_parallel(n_cores=n_cores, size=10, op_dict=op_mid1oh_iv)

    # opt_res.report()
    analyze_optimization(opt_res)


if __name__ == "__main__":
    run_mpfit()
