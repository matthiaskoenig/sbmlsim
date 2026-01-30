"""Example for model comparison.

This supports:
- roadrunner, COPASI & AMICI as solvers
- simulations of conditions as given by PETab conditions
- setting of absolute and relative tolerances
- variable timepoints (as occurring in typical parameter fitting simulations)

# FIXME: improve comparison plots
# - multiple comparison [3 way comparison]
# - show top differences curves

# add additional information for comparison: AMICI/COPASI
# FIXME: run all conditions and make comparison
"""

from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd

from sbmlsim.comparison.diff import DataSetsComparison
from sbmlsim.comparison.simulate_amici import SimulateAmiciSBML
from simulate_roadrunner import SimulateRoadrunnerSBML
from simulate import Condition, SimulateSBML
from pymetadata.console import console

if __name__ == "__main__":
    """Comparison of ICG model simulations."""

    base_path: Path = Path(__file__).parent

    # model
    # model_path = base_path / "resources" / "icg_events_sd.xml"
    # model_path = base_path / "resources" / "icg_sd.xml"
    model_path = base_path / "resources" / "icg_liver.xml"

    # flatten_sbml(
    #     sbml_path=base_path / "resources" / "icg_liver.xml",
    #     sbml_flat_path=base_path / "resources" / "icg_liver_flat.xml",
    # )
    # model_path = base_path / "resources" / "icg_liver_flat.xml"

    print(model_path)

    # results
    results_dir: Path = base_path / "results"

    # conditions
    # conditions_path = base_path / "resources" / "condition.tsv"
    conditions_path = base_path / "resources" / "condition_liver.tsv"

    conditions_list: list[Condition] = Condition.parse_conditions_from_file(
        conditions_path=conditions_path
    )
    conditions: dict[str, Condition] = {c.sid: c for c in conditions_list}

    # simulate condition with simulators
    # ----------------------------------------------------------------
    # timepoints = np.linspace(start=0, stop=100, num=51).tolist()
    timepoints = np.linspace(start=0, stop=10, num=51).tolist()
    # timepoints = np.linspace(0, 10, num=11).tolist()
    absolute_tolerance = 1e-12
    relative_tolerance = 1e-14
    # condition = conditions["infusion1"]
    # condition = conditions["bw80"]
    # condition = conditions["Andersen1999_task_icg_iv"]
    condition = conditions["icg1"]
    # ----------------------------------------------------------------

    print(f"{timepoints=}")
    # timepoints = [0, 1, 10, 20, 35]
    # print(f"{timepoints=}")

    # run comparison
    dfs: dict[str, pd.DataFrame] = {}
    simulator: Type[SimulateSBML]
    for key, simulator in {
        "roadrunner": SimulateRoadrunnerSBML,
        # "copasi": SimulateCopasiSBML,
        "amici": SimulateAmiciSBML,
    }.items():
        console.rule(title=key, align="left", style="white")

        simulator = simulator(
            sbml_path=model_path,
            results_dir=results_dir,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        df = simulator.simulate_condition(
            condition=condition,
            timepoints=timepoints,
        )
        console.print(df.columns)
        console.print(df)
        # console.print(df["Cve_icg"])
        dfs[key] = df

    # debugging plots
    from matplotlib import pyplot as plt
    # f, ax = plt.subplots(nrows=1, ncols=1)
    # df_roadrunner = dfs["roadrunner"]
    # df_amici = dfs["amici"]
    # sid = "LI__bil_ext"
    # for key, df in dfs.items():
    #     ax.plot(df.time, df[sid], label=key)
    # ax.set_xlabel("time")
    # ax.set_ylabel(sid)
    # ax.legend()

    # comparison
    console.rule(style="white")
    comparison = DataSetsComparison(dfs_dict=dfs)
    comparison.report()

    plt.show()
