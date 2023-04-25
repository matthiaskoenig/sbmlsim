"""Example for model comparison.

This supports:
- roadrunner, COPASI & AMICI as solvers
- simulations of conditions as given by PETab conditions
- setting of absolute and relative tolerances
- variable timepoints (as occurring in typical parameter fitting simulations)

# FIXME: improve comparison plots
# - better heatmap
# - multiple comparison [3 way comparison]
# - show top differences curves
# FIXME: fix concentration/amount issues between roadrunner & COPASI
# FIXME: support AMICI
# FIXME: run all conditions and make comparison

"""
from pathlib import Path
from typing import List, Dict, Type

import numpy as np
import pandas as pd

from sbmlsim.comparison.diff import DataSetsComparison
from sbmlsim.comparison.simulate_amici import SimulateAmiciSBML
from sbmlsim.comparison.simulate_copasi import SimulateCopasiSBML
from simulate_roadrunner import SimulateRoadrunnerSBML
from simulate import Condition, SimulateSBML
from sbmlutils.log import get_logger
from sbmlutils.console import console


if __name__ == "__main__":
    """Comparison of ICG model simulations."""

    base_path: Path = Path(__file__).parent

    # model
    # model_path = base_path / "resources" / "icg_events_sd.xml"
    model_path = base_path / "resources" / "icg_sd.xml"
    print(model_path)

    # results
    results_dir: Path = base_path / "results"

    # conditions
    conditions_path = base_path / "resources" / "condition.tsv"


    conditions: List[Condition] = Condition.parse_conditions_from_file(
        conditions_path=conditions_path
    )

    # simulate condition with simulators
    timepoints = np.linspace(start=0, stop=100, num=21).tolist()
    print(f"{timepoints=}")
    # timepoints = [0, 1, 10, 20, 35]
    # print(f"{timepoints=}")


    # run comparison
    dfs: Dict[str, pd.DataFrame] = {}
    simulator: Type[SimulateSBML]
    for key, simulator in {
        "roadrunner": SimulateRoadrunnerSBML,
        # "copasi": SimulateCopasiSBML,
        "amici": SimulateAmiciSBML,

    }.items():
        console.rule(title=key, align="left", style="white")

        simulator = simulator(
            sbml_path=model_path,
            conditions=conditions,
            results_dir=results_dir,
            absolute_tolerance=1E-12,
            relative_tolerance=1E-15,
        )
        df = simulator.simulate_condition(
            condition=conditions[0],
            timepoints=timepoints,
        )
        console.print(df.columns)
        console.print(df)
        # console.print(df["Cve_icg"])
        dfs[key] = df

    # comparison
    exit()
    console.rule(style="white")
    comparison = DataSetsComparison(
        dfs_dict=dfs
    )
    comparison.report()
    comparison.plot_diff()

    from matplotlib import pyplot as plt
    plt.show()
