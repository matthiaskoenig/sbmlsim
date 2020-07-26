"""
Run some example experiments.
"""
from pathlib import Path
from sbmlsim.experiment import ExperimentReport, ExperimentRunner
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.utils import timeit

from sbmlsim.examples.experiments.midazolam.experiments.mandema1992 import Mandema1992
from sbmlsim.examples.experiments.midazolam.experiments.kupferschmidt1995 import Kupferschmidt1995


@timeit
def midazolam_experiment():
    BASE_PATH = Path(__file__).parent
    runner = ExperimentRunner(
        [Mandema1992, Kupferschmidt1995],
        simulator=SimulatorSerial(),
        base_path=BASE_PATH,
        data_path=BASE_PATH / "data",
    )
    results = runner.run_experiments(
        output_path=BASE_PATH / "results",
        show_figures=True
    )
    report = ExperimentReport(results)
    report.create_report(output_path=BASE_PATH / "results")


if __name__ == "__main__":
    midazolam_experiment()
