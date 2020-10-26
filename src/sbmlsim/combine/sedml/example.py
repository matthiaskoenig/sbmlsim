from pathlib import Path

from sbmlsim.combine.sedml.io import read_sedml
from sbmlsim.combine.sedml.parser import SEDMLParser
from sbmlsim.experiment import SimulationExperiment


if __name__ == "__main__":

    base_path = Path(__file__).parent
    working_dir = base_path / "experiments"
    sedml_path = working_dir / "repressilator_sedml.xml"
    doc, working_dir, input_type = read_sedml(
        source=str(sedml_path), working_dir=working_dir
    )

    sed_parser = SEDMLParser(doc, working_dir=working_dir)
    print(sed_parser.models)
    print(sed_parser.data_descriptions)

    # models ->

    print(sed_parser.exp_class)
    exp = sed_parser.exp_class()  # type: SimulationExperiment
    exp.initialize()

    print("exp_models", exp.models())

    # TODO:
    # execute simulation experiment

    # TODO: write to new SED-ML file
