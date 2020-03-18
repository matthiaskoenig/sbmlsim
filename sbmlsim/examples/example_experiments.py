"""
Run some example experiments.
"""
from pathlib import Path
from sbmlsim.report import create_report

from sbmlsim.examples.glucose.experiments.dose_response import DoseResponseExperiment
from sbmlsim.utils import timeit


@timeit
def glucose_experiment():
    BASE_PATH = Path(__file__).parent / "glucose"

    results = []
    exp = DoseResponseExperiment(
        base_path=BASE_PATH,
        data_path=BASE_PATH / "data",
    )
    info = exp.run(
        output_path=BASE_PATH / "results",
        show_figures=False
    )
    results.append(info)

    create_report(results, output_path=BASE_PATH)


if __name__ == "__main__":
    glucose_experiment()
