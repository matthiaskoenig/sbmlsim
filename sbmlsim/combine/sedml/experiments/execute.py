"""
Main entry point to run all simulation experiments with the glucose model.
"""
from typing import List, Dict
from pathlib import Path
import pandas as pd

from sbmlsim.experiment import SimulationExperiment, run_experiment, ExperimentResult
from sbmlsim.combine.sedml.experiments.repressilator import RepressilatorExperiment


def run_repressilator(output_path: Path, show_figures: bool = False) -> ExperimentResult:
    """ Run all experiments.

    :param output_path:
    :param experiments: list of experiments to run
    :param show_figures:
    :return:
    """
    base_path = Path(__file__).parent
    model_path = base_path / "repressilator.xml"
    data_path = base_path

    results = run_experiment(
        RepressilatorExperiment,
        output_path=output_path / RepressilatorExperiment.__name__,
        model_path=model_path,
        data_path=data_path,
        show_figures=show_figures)

    return results


if __name__ == "__main__":
    run_repressilator(output_path=Path("."))

