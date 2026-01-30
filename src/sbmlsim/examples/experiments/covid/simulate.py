"""
Run COVID-19 model experiments.
"""

from pathlib import Path

from sbmlsim.combine.sedml.parser import SEDMLSerializer
from sbmlsim.combine.sedml.runner import execute_sedml
from sbmlsim.examples.experiments.covid.experiments import (
    Bertozzi2020,
    Carcione2020,
    Cuadros2020,
)
from sbmlsim.experiment.runner import run_experiments


def run_covid_examples(output_path: Path) -> None:
    experiments = [
        Bertozzi2020,
        Cuadros2020,
        Carcione2020,
    ]
    run_experiments(
        experiments=experiments,
        output_path=output_path / "sbmlsim",
        data_path=output_path,
        base_path=output_path,
    )

    for experiment in experiments:
        exp_id = experiment.__name__
        # serialize to SED-ML/OMEX archive
        omex_path = output_path / f"{exp_id}.omex"
        SEDMLSerializer(
            exp_class=experiment,
            working_dir=output_path / "omex",
            sedml_filename=f"{exp_id}_sedml.xml",
            omex_path=omex_path,
        )

        # execute OMEX archive
        execute_sedml(
            path=omex_path,
            working_dir=output_path / "sbmlsim_omex",
            output_path=output_path / "sbmlsim_omex",
        )


if __name__ == "__main__":
    run_covid_examples(output_path=Path(__file__).parent / "results")
