from pathlib import Path

from pprint import pprint

from sbmlsim.combine.sedml.io import read_sedml
from sbmlsim.combine.sedml.parser import SEDMLParser
from sbmlsim.experiment import SimulationExperiment, ExperimentRunner
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulator.simulation_ray import SimulatorParallel

import xmltodict
import json

base_path = Path(__file__).parent


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


def execute_sedml(working_dir: Path, name: str, sedml_path: Path) -> None:
    """Execute the given SED-ML in the working directory."""
    # convert to json
    sedmltojson(sedml_path)

    sed_doc, errorlog, _, _ = read_sedml(
        source=str(sedml_path), working_dir=working_dir
    )

    if errorlog.getNumErrors() > 0:
        raise IOError("Errors in the SED-ML document.")

    sed_parser = SEDMLParser(sed_doc, working_dir=working_dir, name=name)
    sed_parser.print_info()

    # check created experiment
    exp: SimulationExperiment = sed_parser.exp_class()
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
        output_path=base_path / "results", show_figures=True,
        figure_formats=["svg", "png"]
    )

    # TODO: write experiment to SED-ML file
    # serialization of experiments


if __name__ == "__main__":
    # ----------------------
    # L1V4 Plotting
    # ----------------------
    working_dir = base_path / "l1v4_plotting"
    for name, sedml_file in [
        # ("markertype", "markertype.sedml"),
        # ("linetype", "linetype.sedml"),
        ("axis", "axis.sedml"),
        # ("repressilator_figure", "repressilator_figure.xml"),
        # ("repressilator_l1v3", "repressilator_l1v3.xml"),
        # ("repressilator_urn_l1v3", "repressilator_urn_l1v3.xml"),  # FIXME: resolve URN
        # ("TestFile1", "test_file_1.sedml"),
        # ("TestLineFill", "test_line_fill.sedml"),

        # ("StackedBar", "stacked_bar.sedml"),
        # ("HBarStacked", "test_3hbarstacked.sedml"),
        # ("Bar", "test_bar.sedml"),
        # ("Bar", "test_bar3stacked.sedml"),
        # ("StackedBar", "test_file.sedml"),
        # ("StackedBar", "test_hbar_stacked.sedml"),
        # ("StackedBar", "test_shaded_area.sedml"),
    ]:
        execute_sedml(
            working_dir=working_dir,
            name=name,
            sedml_path=working_dir / sedml_file
        )

    # ----------------------
    # L1V4 Parameter Fitting
    # ----------------------
    working_dir = base_path / "l1v4_parameter_fitting"
    for name, sedml_file in [
        # ("Elowitz_Nature2000", "Elowitz_Nature2000.xml"),
    ]:
        execute_sedml(
            working_dir=working_dir,
            name=name,
            sedml_path=working_dir / sedml_file
        )
