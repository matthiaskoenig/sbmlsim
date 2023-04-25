"""Example for model comparison."""
from pathlib import Path
from typing import List, Dict

import pandas as pd

from sbmlsim.comparison.simulate_amici import SimulateAmiciSBML
from sbmlsim.comparison.simulate_copasi import SimulateCopasiSBML
from simulate_roadrunner import SimulateRoadrunnerSBML
from simulate import Condition, Timepoints, SimulateSBML
from sbmlutils.log import get_logger
from sbmlutils.console import console

# FIXME: support roadrunner

if __name__ == "__main__":

    base_path: Path = Path(__file__).parent
    model_path = base_path / "resources" / "icg_sd.xml"
    conditions_path = base_path / "resources" / "condition.tsv"
    results_dir: Path = base_path / "results"
    timepoints = Timepoints(start=0, end=5, steps=10)

    conditions: List[Condition] = Condition.parse_conditions_from_file(conditions_path=conditions_path)

    # simulate condition with simulators

    dfs: Dict[str, pd.DataFrame] = {}

    simulator: SimulateSBML
    for key, simulator in {
        "roadrunner": SimulateRoadrunnerSBML,
        "copasi": SimulateCopasiSBML,
        "amici": SimulateAmiciSBML,

    }.items():
        console.rule(title=key, align="left", style="white")

        simulator = simulator(
            sbml_path=model_path,
            conditions=conditions,
            results_dir=results_dir,
        )
        df = simulator.simulate_condition(
            condition=conditions[0],
            timepoints=timepoints,
        )
        console.print(df.columns)
        console.print(df)
        dfs[key] = df

