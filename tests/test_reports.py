from sbmlsim.examples.experiments.glucose import BASE_PATH, DATA_PATH
from sbmlsim.examples.experiments.glucose.experiments.dose_response import (
    DoseResponseExperiment,
)
from sbmlsim.experiment import ExperimentReport, ExperimentRunner
from sbmlsim.simulator import SimulatorSerial


def test_glucose_report(tmp_path):
    """Create model report for the glucose experiment.

    :param tmp_path:
    :return:
    """
    runner = ExperimentRunner(
        [DoseResponseExperiment],
        base_path=BASE_PATH,
        data_path=DATA_PATH,
        simulator=SimulatorSerial(),
    )
    results = runner.run_experiments(
        output_path=tmp_path / "results", show_figures=False
    )
    ExperimentReport(results).create_report(output_path=tmp_path / "results")
