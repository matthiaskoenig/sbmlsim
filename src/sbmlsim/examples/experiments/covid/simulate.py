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


if __name__ == "__main__":
    experiments = [
        Bertozzi2020,
        Cuadros2020,
        Carcione2020,
    ]
    base_path = Path(__file__).parent
    run_experiments(
        experiments=experiments,
        output_path=base_path / "results" / "sbmlsim",
        data_path=base_path,
        base_path=base_path,
        parallel=True,
    )

    for experiment in experiments:
        exp_id = experiment.__name__
        # serialize to SED-ML/OMEX archive
        omex_path = Path(__file__).parent / "results" / f"{exp_id}.omex"
        serializer = SEDMLSerializer(
            experiment=experiment,
            working_dir=Path(__file__).parent / "results" / "omex",
            sedml_filename=f"{exp_id}_sedml.xml",
            omex_path=omex_path,
        )

        # execute OMEX archive
        execute_sedml(
            path=omex_path,
            working_dir=Path(__file__).parent / "results" / "sbmlsim_omex",
            output_path=Path(__file__).parent / "results" / "sbmlsim_omex",
        )
