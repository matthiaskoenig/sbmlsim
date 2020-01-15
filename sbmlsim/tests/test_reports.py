import pytest
from pathlib import Path
from sbmlsim.report import create_report


def test_glucose_report(tmp_path):
    """ Create model report for the glucose experiment.

    :param tmp_path:
    :return:
    """
    from sbmlsim.experiment import run_experiment
    from sbmlsim.examples.glucose.experiments.dose_response import DoseResponseExperiment
    from sbmlsim.examples.glucose import MODEL_PATH, DATA_PATH

    results = []
    info = run_experiment(
        DoseResponseExperiment,
        output_path=tmp_path / "results",
        model_path=MODEL_PATH / "liver_glucose.xml",
        data_path=DATA_PATH,
        show_figures=False
    )
    results.append(info)
    create_report(results, output_path=tmp_path)
