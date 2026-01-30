"""Simulation of examples."""

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from sbmlsim.comparison.diff import DataSetsComparison, get_files_by_extension
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import TimecourseSim
from sbmlsim.simulator import SimulatorSerial

DATA_DIR = Path(__file__).parent / "data"
diff_path = Path(__file__).parent / "diff"


def run_simulations(create_files: bool = True) -> None:
    """Run all the simulations."""

    simulator = SimulatorSerial.from_sbml(REPRESSILATOR_SBML)
    simulator.set_integrator_settings(
        absolute_tolerance=1e-16,
        relative_tolerance=1e-13,
    )
    simulations = get_files_by_extension(diff_path)
    for simulation_key, json_path in simulations.items():
        tsv_path = diff_path / "sbmlsim" / f"{simulation_key}.tsv"
        tcsim = TimecourseSim.from_json(json_path)
        # print(tcsim)
        xres = simulator.run_timecourse(tcsim)
        if create_files:
            df = xres.to_mean_dataframe()
            df.to_csv(tsv_path, sep="\t", index=False)


def run_comparisons(create_files=True):
    """Run comparison of tests simulations.

    :return:
    """
    diff_path = Path(DATA_DIR) / "diff"

    simulation_keys = get_files_by_extension(diff_path)
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
            title=f"{simulation_key} (sbmlsim | jws)",
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
