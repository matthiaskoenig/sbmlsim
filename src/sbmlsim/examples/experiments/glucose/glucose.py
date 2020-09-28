"""
Run some example experiments.
"""
from pathlib import Path

from sbmlsim.examples.experiments.glucose.experiments.dose_response import (
    DoseResponseExperiment,
)
from sbmlsim.experiment import ExperimentReport, ExperimentRunner
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.utils import timeit


@timeit
def glucose_experiment():
    BASE_PATH = Path(__file__).parent
    runner = ExperimentRunner(
        [DoseResponseExperiment],
        simulator=SimulatorSerial(),
        base_path=BASE_PATH,
        data_path=BASE_PATH / "data",
    )
    results = runner.run_experiments(
        output_path=BASE_PATH / "results", show_figures=False
    )
    report = ExperimentReport(results)
    report.create_report(output_path=BASE_PATH / "results")


if __name__ == "__main__":
    glucose_experiment()
