"""
Run COVID-19 model experiments.
"""
from pathlib import Path

from sbmlsim.examples.experiments.covid.experiments import (
    Bertozzi2020,
    Carcione2020,
    Cuadros2020,
)
from sbmlsim.experiment import ExperimentRunner
from sbmlsim.report.experiment_report import ExperimentReport, ReportResults
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulator.simulation_ray import SimulatorParallel


def covid_experiments(base_path: Path, results_path: Path):
    runner = ExperimentRunner(
        [
            Cuadros2020,
            # Bertozzi2020,
            Carcione2020,
        ],
        simulator=SimulatorParallel(),
        base_path=base_path,
        data_path=base_path,
    )
    results = runner.run_experiments(
        output_path=base_path / "results",
        show_figures=True,
        reduced_selections=False,
    )
    report_results = ReportResults()
    for exp_result in results:
        report_results.add_experiment_result(exp_result=exp_result)

    report = ExperimentReport(report_results)
    report.create_report(output_path=results_path)


if __name__ == "__main__":
    base_path = Path(__file__).parent
    results_path = base_path / "results"
    if not results_path.exists():
        results_path.mkdir(parents=True)

    covid_experiments(base_path=base_path, results_path=results_path)
