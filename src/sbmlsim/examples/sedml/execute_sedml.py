from pathlib import Path

from pprint import pprint

from sbmlsim.combine.sedml.io import read_sedml
from sbmlsim.combine.sedml.parser import SEDMLParser
from sbmlsim.experiment import SimulationExperiment, ExperimentRunner
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulator.simulation_ray import SimulatorParallel

import xmltodict
import json


def sedmltojson(sedml_path: Path) -> None:
    """Convert SED-ML to JSON file."""
    with open(sedml_path, "r") as f_sedml:
        xml = f_sedml.read()

    my_dict = xmltodict.parse(xml)
    json_data = json.dumps(my_dict, indent=2)

    json_path = sedml_path.parent / f"{sedml_path.name}.json"
    with open(json_path, "w") as f_json:
        # print(json_data)
        f_json.write(json_data)


def execute_sedml(working_dir: Path, sedml_path: Path) -> None:

    # convert to json
    sedmltojson(sedml_path)

    doc, working_dir, input_type = read_sedml(
        source=str(sedml_path), working_dir=working_dir
    )

    sed_parser = SEDMLParser(doc, working_dir=working_dir)

    print("*" * 80)
    print("--- MODELS ---")
    pprint(sed_parser.models)

    print("\n--- DATA DESCRIPTIONS ---")
    pprint(sed_parser.data_descriptions)

    print("\n--- SIMULATIONS ---")
    pprint(sed_parser.simulations)

    print("\n--- TASKS ---")
    pprint(sed_parser.tasks)

    print("\n--- DATA ---")
    pprint(sed_parser.data)

    print("\n--- FIGURES ---")
    pprint(sed_parser.figures)

    print("*" * 80)
    # analyze simulation experiment
    # print(sed_parser.exp_class)
    exp = sed_parser.exp_class()  # type: SimulationExperiment
    exp.initialize()
    print(exp)

    # execute simulation experiment
    base_path = Path(__file__).parent
    data_path = base_path
    runner = ExperimentRunner(
        [sed_parser.exp_class],
        simulator=SimulatorSerial(),
        data_path=data_path,
        base_path=base_path,
    )
    _results = runner.run_experiments(
        output_path=base_path / "results", show_figures=True
    )

    # TODO: write experiment to SED-ML file
    # serialization of experiments


if __name__ == "__main__":
    base_path = Path(__file__).parent
    working_dir = base_path / "experiments"
    # sedml_path = working_dir / "repressilator_sedml.xml"
    # sedml_path = working_dir / "test_file_1.sedml"
    sedml_path = working_dir / "test_line_fill.sedml"

    execute_sedml(working_dir=working_dir, sedml_path=sedml_path)
