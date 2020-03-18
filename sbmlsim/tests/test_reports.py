import pytest

from sbmlsim.report import create_report


def test_glucose_report(tmp_path):
    """ Create model report for the glucose experiment.

    :param tmp_path:
    :return:
    """
    from sbmlsim.examples.glucose.experiments.dose_response import DoseResponseExperiment
    from sbmlsim.examples.glucose import BASE_PATH, DATA_PATH

    results = []
    exp = DoseResponseExperiment(
        base_path=BASE_PATH,
        data_path=DATA_PATH
    )
    info = exp.run(
        output_path=tmp_path / "results",
        show_figures=False
    )
    results.append(info)
    create_report(results, output_path=tmp_path)
