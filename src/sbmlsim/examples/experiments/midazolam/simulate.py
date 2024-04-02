"""
Run some example experiments.
"""
from pathlib import Path
from typing import Type

from sbmlsim.combine.sedml.parser import SEDMLSerializer
from sbmlsim.combine.sedml.runner import execute_sedml
from sbmlsim.examples.experiments.midazolam.experiments.kupferschmidt1995 import (
    Kupferschmidt1995,
)
from sbmlsim.examples.experiments.midazolam.experiments.mandema1992 import Mandema1992
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.report.experiment_report import ExperimentReport, ReportResults
from sbmlsim.simulator.simulation_serial import SimulatorSerial


def run_midazolam_experiments(output_path: Path) -> None:
    """Run midazolam simulation experiments."""
    base_path = Path(__file__).parent
    runner = ExperimentRunner(
        [
            Mandema1992,
            Kupferschmidt1995,
        ],
        simulator=SimulatorSerial(),
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
    # run_midazolam_experiments(output_path)

    exp_class: Type[SimulationExperiment]
    for exp_class in [Kupferschmidt1995]:  # [Mandema1992, Kupferschmidt1995]:
        # serialize to SED-ML/OMEX archive
        omex_path = Path(__file__).parent / "results" / f"{exp_class.__name__}.omex"
        serializer = SEDMLSerializer(
            exp_class=exp_class,
            working_dir=output_path / "omex",
            sedml_filename=f"{exp_class.__name__}_sedml.xml",
            omex_path=omex_path,
            data_path=Path(__file__).parent / "data",
        )

        # execute OMEX archive
        execute_sedml(
            path=omex_path,
            working_dir=output_path / "sbmlsim_omex",
            output_path=output_path / "sbmlsim_omex",
        )
