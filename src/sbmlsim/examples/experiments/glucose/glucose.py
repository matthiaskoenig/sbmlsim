"""
Run some example experiments.
"""

from pathlib import Path

from sbmlsim.examples.experiments.glucose.experiments.dose_response import (
    DoseResponseExperiment,
)
from sbmlsim.experiment import ExperimentReport, ExperimentRunner
from sbmlsim.simulator import SimulatorSerial


def run_glucose_experiments(output_path: Path) -> None:
    """Run glucose experiments."""
    BASE_PATH = Path(__file__).parent
    runner = ExperimentRunner(
        [DoseResponseExperiment],
        simulator=SimulatorSerial(),
        base_path=BASE_PATH,
        data_path=BASE_PATH / "data",
    )
    results = runner.run_experiments(output_path=output_path, show_figures=False)
    report = ExperimentReport(results)
    report.create_report(output_path=BASE_PATH / "results")


if __name__ == "__main__":
    run_glucose_experiments(Path(__file__).parent / "results")
