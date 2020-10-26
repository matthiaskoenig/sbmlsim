"""
Run some example experiments.
"""
from pathlib import Path

from sbmlsim.examples.experiments.midazolam.experiments.kupferschmidt1995 import (
    Kupferschmidt1995,
)
from sbmlsim.examples.experiments.midazolam.experiments.mandema1992 import Mandema1992
from sbmlsim.examples.experiments.midazolam.experiments.model_change import (
    MidazolamModelChangeExperiment,
)
from sbmlsim.experiment import ExperimentRunner
from sbmlsim.report.experiment_report import ExperimentReport, ReportResults
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulator.simulation_ray import SimulatorParallel
from sbmlsim.utils import timeit


# from sbmlsim.examples.experiments.midazolam.experiments.pkscan import PKScanExperiment


@timeit
def midazolam_experiment():
    BASE_PATH = Path(__file__).parent
    runner = ExperimentRunner(
        # [Mandema1992, Kupferschmidt1995],
        [MidazolamModelChangeExperiment],
        # [PKScanExperiment],
        # simulator=SimulatorSerial(),
        simulator=SimulatorParallel(),
        base_path=BASE_PATH,
        data_path=BASE_PATH / "data",
    )
    results = runner.run_experiments(
        output_path=BASE_PATH / "results", show_figures=True
    )
    report_results = ReportResults()
    for exp_result in results:
        report_results.add_experiment_result(exp_result=exp_result)

    report = ExperimentReport(report_results)
    report.create_report(output_path=BASE_PATH / "results")


if __name__ == "__main__":
    midazolam_experiment()
