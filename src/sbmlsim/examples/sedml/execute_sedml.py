from pathlib import Path
from pprint import pprint

from sbmlsim.combine.sedml.io import read_sedml
from sbmlsim.combine.sedml.parser import SEDMLParser
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulator.simulation_ray import SimulatorParallel


if __name__ == "__main__":

    base_path = Path(__file__).parent
    working_dir = base_path / "experiments"
    sedml_path = working_dir / "repressilator_sedml.xml"

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
