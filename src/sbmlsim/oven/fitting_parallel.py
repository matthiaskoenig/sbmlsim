"""
Defines the parameter fitting problems
"""
from sbmlsim.examples.experiments.midazolam import MIDAZOLAM_PATH
from sbmlsim.examples.experiments.midazolam.experiments.mandema1992 import Mandema1992
from sbmlsim.fit import FitExperiment, FitParameter, run_optimization
from sbmlsim.fit.optimization import OptimizationProblem, OptimizerType, SamplingType
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulator.simulation_ray import SimulatorParallel


RESULTS_PATH = MIDAZOLAM_PATH / "results"
DATA_PATH = MIDAZOLAM_PATH / "data"


op_kwargs = {
    "opid": "mid1oh_iv",
    "base_path": MIDAZOLAM_PATH,
    "data_path": DATA_PATH,
    "fit_experiments": [FitExperiment(experiment=Mandema1992, mappings=["fm4"])],
    "fit_parameters": [
        # distribution parameters
        FitParameter(
            parameter_id="ftissue_mid1oh",
            start_value=1.0,
            lower_bound=1,
            upper_bound=1e5,
            unit="liter/min",
        ),
        FitParameter(
            parameter_id="fup_mid1oh",
            start_value=0.1,
            lower_bound=0.01,
            upper_bound=0.3,
            unit="dimensionless",
        ),
        # mid1oh kinetics
        FitParameter(
            parameter_id="KI__MID1OHEX_Vmax",
            start_value=100,
            lower_bound=1e-1,
            upper_bound=1e4,
            unit="mmole/min",
        ),
    ],
}


if __name__ == "__main__":

    workers = 5
    ops = []
    for k in range(workers):
        simulator = SimulatorParallel(actor_count=1)
        op = OptimizationProblem(simulator=simulator, **op_kwargs)
        ops.append(op)

    for op in ops:
        # FIXME: set seed here
        run_optimization(
            op,
            size=5,
            output_path=RESULTS_PATH,
            plot_results=False,
            optimizer=OptimizerType.LEAST_SQUARE,
            sampling=SamplingType.LOGUNIFORM_LHS,
            diff_step=0.05,
        )

    # combining simulation results
    """
    from sbmlsim.fit.analysis import OptimizationResult
    opt_res = OptimizationResult.combine([opt_res1, opt_res2])
    opt_res.report()
    """
