from pathlib import Path
import pandas as pd

from typing import List

from sbmlsim.diff import DataSetsComparison
from matplotlib import pyplot as plt
from sbmlsim.simulation_serial import SimulatorSerial as Simulator
from sbmlsim.tests.constants import DATA_PATH, MODEL_REPRESSILATOR

from sbmlsim.timecourse import TimecourseSim, Timecourse


def get_simulation_keys() -> List[str]:
    """Get all simulation definitions in the test directory."""
    diff_path = Path(DATA_PATH) / "diff"
    simulation_keys = []

    # get all json files in the folder
    files = [f for f in diff_path.glob('**/*') if f.is_file() and f.suffix == ".json"]
    keys = [f.name[:-5] for f in files]

    return sorted(keys)


def run_simulations():
    """ Run all the simulations.

    :return:
    """
    diff_path = Path(DATA_PATH) / "diff"
    # FIXME: do not hardcode the model, model should be provided relative to the json definition
    simulator = Simulator(MODEL_REPRESSILATOR,
                          absolute_tolerance=1E-16,
                          relative_tolerance=1E-13)

    simulation_keys = get_simulation_keys()
    for simulation_key in simulation_keys:
        json_path = diff_path / f"{simulation_key}.json"
        tsv_path = diff_path / "sbmlsim" / f"{simulation_key}.tsv"
        tcsim = TimecourseSim.from_json(json_path)
        # print(tcsim)
        result = simulator.timecourses([tcsim])
        result.mean.to_csv(tsv_path, sep="\t", index=False)


def run_comparisons():
    """ Run comparison of test simulations.

    :return:
    """
    diff_path = Path(DATA_PATH) / "diff"

    simulation_keys = get_simulation_keys()
    print(simulation_keys)

    for simulation_key in simulation_keys:
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
        fig_path = diff_path / f"{simulation_key}_diff.png"
        f.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.show()

        report_path = diff_path / f"{simulation_key}_diff.tsv"
        with open(report_path, "w") as f_report:
            f_report.write(dsc.report_str())


if __name__ == "__main__":

    run_simulations()
    run_comparisons()

