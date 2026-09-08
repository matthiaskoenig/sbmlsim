"""Reusable functionality to run the HCTZ simulation experiments."""

from pathlib import Path

from examples.hctz import DATA_PATH, HCTZ_PATH, MODEL_PATH
from sbmlsim import log
from sbmlsim.console import console
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.plot import Figure
from sbmlsim.report.experiment_report import ExperimentReport, ReportResults
from sbmlsim.simulator.simulation_serial import SimulatorSerial

Figure.legend_fontsize = 10


def run_experiments(
    experiment_classes: type[SimulationExperiment] | list[type[SimulationExperiment]],
    output_dir: Path | str,
) -> Path:
    """Execute the given simulation experiment(s).

    sbmlsim does not configure logging, so the messages of the package are shown
    on the rich console explicitly, as in every script.

    Args:
        experiment_classes: simulation experiment class or classes to run.
        output_dir: directory for the results, relative to the working directory.

    Returns:
        Path of the directory the results were written to.
    """
    log.enable_rich_logging()

    output_path = Path("results") / output_dir
    simulator = SimulatorSerial(model=MODEL_PATH)

    if not isinstance(experiment_classes, list):
        experiment_classes = [experiment_classes]

    runner = ExperimentRunner(
        experiment_classes=experiment_classes,
        data_path=DATA_PATH,
        base_path=HCTZ_PATH,
        simulator=simulator,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
    )
    results = runner.run_experiments(
        output_path=output_path,
        show_figures=False,
        save_results=False,
        figure_formats=["svg", "png"],
        reduced_selections=True,
    )

    report_results = ReportResults()
    for exp_result in results:
        report_results.add_experiment_result(exp_result=exp_result)

    # create HTML report
    report = ExperimentReport(report_results, metadata=None)
    report.create_report(output_path, report_type=ExperimentReport.ReportType.HTML)

    console.print(
        f"Successfully executed simulation experiments: file://{output_path.resolve()}",
        style="success",
    )
    return output_path
