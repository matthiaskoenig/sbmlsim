"""
Run some example experiments.
"""
from pathlib import Path
from sbmlsim.experiment import run_experiment
from sbmlsim.report import create_report

from sbmlsim.examples.glucose.experiments.dose_response import DoseResponseExperiment
from sbmlsim.utils import timeit

@timeit
def glucose_experiment():
    BASE_PATH = Path(__file__).parent / "glucose"

    results = []
    info = run_experiment(
        DoseResponseExperiment,
        output_path=BASE_PATH / "results",
        model_path=BASE_PATH / "model" / "liver_glucose.xml",
        data_path=BASE_PATH / "data",
        show_figures=False
    )
    results.append(info)
    create_report(results, output_path=BASE_PATH)


if __name__ == "__main__":

    glucose_experiment()
