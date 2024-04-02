"""Module with helpers to execute SED-ML files and COMBINE archives."""

import json
import os
from pathlib import Path

import xmltodict

from sbmlsim.combine.sedml.io import SEDMLReader
from sbmlsim.combine.sedml.parser import SEDMLParser
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.simulator import SimulatorSerial


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


def execute_sedml(path: Path, working_dir: Path, output_path: Path) -> None:
    """Execute the given SED-ML in the working directory.

    :param path: path to SED-ML file or OMEX archive.
    :param working_dir: directory for execution and resources
    :return:
    """
    sedml_reader = SEDMLReader(source=path, working_dir=working_dir)
    print(sedml_reader)

    cwd = os.getcwd()
    os.chdir(sedml_reader.exec_dir)
    print(os.getcwd())

    sedml_parser = SEDMLParser(
        sed_doc=sedml_reader.sed_doc,
        exec_dir=sedml_reader.exec_dir,
        working_dir=working_dir,
        name=path.stem,
    )
    sedml_parser.print_info()

    # check created experiment
    exp: SimulationExperiment = sedml_parser.exp_class()
    exp.initialize()
    print(exp)

    # execute simulation experiment
    runner = ExperimentRunner(
        [sedml_parser.exp_class],
        simulator=SimulatorSerial(),
        data_path=sedml_reader.exec_dir,
        base_path=sedml_reader.exec_dir,
    )
    runner.run_experiments(
        output_path=output_path, show_figures=True, figure_formats=["svg", "png"]
    )

    # TODO: write experiment to SED-ML file
    # serialization of experiments

    os.chdir(cwd)
