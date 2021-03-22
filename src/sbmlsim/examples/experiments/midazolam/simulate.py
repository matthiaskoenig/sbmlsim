"""
Run some example experiments.
"""
from pathlib import Path

from sbmlsim.examples.experiments.midazolam.experiments.kupferschmidt1995 import (
    Kupferschmidt1995,
)
from sbmlsim.examples.experiments.midazolam.experiments.mandema1992 import Mandema1992
from sbmlsim.experiment import ExperimentRunner
from sbmlsim.report.experiment_report import ExperimentReport, ReportResults
from sbmlsim.simulator.simulation_ray import SimulatorParallel


def run_midazolam_experiments(output_path: Path) -> None:
    """Run midazolam simulation experiments."""
    base_path = Path(__file__).parent
    runner = ExperimentRunner(
        [
            Mandema1992,
            Kupferschmidt1995,
        ],
        simulator=SimulatorParallel(),
        base_path=base_path,
        data_path=base_path / "data",
    )
    results = runner.run_experiments(output_path=output_path, show_figures=True)
    report_results = ReportResults()
    for exp_result in results:
        report_results.add_experiment_result(exp_result=exp_result)

    report = ExperimentReport(report_results)
    report.create_report(output_path=base_path / "results")


if __name__ == "__main__":
    output_path = Path(__file__).parent / "results"
    run_midazolam_experiments(output_path)
