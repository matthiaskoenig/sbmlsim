from pathlib import Path
import pandas as pd

from sbmlsim.diff import DataSetsComparison, get_json_files
from matplotlib import pyplot as plt
from sbmlsim.simulator.simulation_serial import SimulatorSerial as Simulator
from sbmlsim.tests.constants import DATA_PATH, MODEL_REPRESSILATOR
from sbmlsim.models import RoadrunnerSBMLModel
from sbmlsim.simulation.timecourse import TimecourseSim


def run_simulations(create_files=True):
    """ Run all the simulations.

    :return:
    """
    diff_path = Path(DATA_PATH) / "diff"
    simulator = Simulator(RoadrunnerSBMLModel(MODEL_REPRESSILATOR),
                          absolute_tolerance=1E-16,
                          relative_tolerance=1E-13)

    simulations = get_json_files(diff_path)
    for simulation_key, json_path in simulations.items():

        tsv_path = diff_path / "sbmlsim" / f"{simulation_key}.tsv"
        tcsim = TimecourseSim.from_json(json_path)
        # print(tcsim)
        result = simulator.timecourses([tcsim])
        if create_files:
            result.mean.to_csv(tsv_path, sep="\t", index=False)


def run_comparisons(create_files=True):
    """ Run comparison of test simulations.

    :return:
    """
    diff_path = Path(DATA_PATH) / "diff"

    simulation_keys = get_json_files(diff_path)
    print(simulation_keys)

    for simulation_key in simulation_keys.keys():
        # run the comparison
        df_dict = {}
        for simulator_key in ["sbmlsim", "jws"]:
            tsv_path = diff_path / simulator_key / f"{simulation_key}.tsv"
            df_dict[simulator_key] = pd.read_csv(tsv_path, sep="\t")

        # perform comparison
        dsc = DataSetsComparison(
            dfs_dict=df_dict,
            columns_filter=None,
            title=f"{simulation_key} (sbmlsim | jws)"
        )

        f = dsc.report()
        if create_files:
            fig_path = diff_path / f"{simulation_key}_diff.png"
            f.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.show()

            report_path = diff_path / f"{simulation_key}_diff.tsv"
            with open(report_path, "w") as f_report:
                f_report.write(dsc.report_str())


if __name__ == "__main__":

    run_simulations()
    run_comparisons()

