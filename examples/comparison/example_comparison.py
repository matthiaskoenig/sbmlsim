"""Comparison of the simulations of roadrunner with AMICI and COPASI.

Every condition of a PEtab condition table is simulated by every simulator at
the same time points and with the same tolerances, as in a parameter fit, and
the results of AMICI and COPASI are compared with the results of roadrunner.
AMICI and COPASI (via basico) are not dependencies of sbmlsim, a simulator which
is not installed is skipped.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from examples.comparison.simulate import Condition, SimulateSBML
from examples.comparison.simulate_roadrunner import SimulateRoadrunnerSBML
from sbmlsim.comparison.diff import DataSetsComparison
from sbmlsim.console import console

#: the simulator the others are compared with
REFERENCE = "roadrunner"


def simulator_classes() -> dict[str, type[SimulateSBML]]:
    """Get the simulators which are installed, the reference first."""
    classes: dict[str, type[SimulateSBML]] = {REFERENCE: SimulateRoadrunnerSBML}
    try:
        from examples.comparison.simulate_amici import SimulateAmiciSBML

        classes["amici"] = SimulateAmiciSBML
    except ImportError:
        console.print("AMICI is not installed, it is not compared.")
    try:
        from examples.comparison.simulate_copasi import SimulateCopasiSBML

        classes["copasi"] = SimulateCopasiSBML
    except ImportError:
        console.print("basico is not installed, COPASI is not compared.")
    return classes


def compare_simulators(
    model_path: Path,
    conditions_path: Path,
    timepoints: np.ndarray,
    results_dir: Path,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-14,
) -> dict[str, dict[str, DataSetsComparison]]:
    """Simulate every condition with every simulator and compare the results.

    Returns:
        The comparisons per condition and simulator, against the reference.
    """
    conditions: list[Condition] = Condition.parse_conditions_from_file(
        conditions_path=conditions_path
    )
    classes = simulator_classes()

    comparisons: dict[str, dict[str, DataSetsComparison]] = {}
    for condition in conditions:
        console.rule(title=condition.sid, align="left", style="white")
        # the simulators keep the changes of a condition, every condition is
        # simulated by new simulators
        dfs: dict[str, pd.DataFrame] = {
            key: simulator_class(
                sbml_path=model_path,
                results_dir=results_dir,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            ).simulate_condition(condition=condition, timepoints=timepoints)
            for key, simulator_class in classes.items()
        }
        comparisons[condition.sid] = {}
        for key, df in dfs.items():
            if key == REFERENCE:
                continue
            comparison = DataSetsComparison(
                dfs_dict={REFERENCE: dfs[REFERENCE], key: df},
                title=f"{condition.sid}: {REFERENCE} | {key}",
            )
            fig = comparison.report()
            fig.savefig(
                results_dir / f"comparison_{condition.sid}_{key}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            comparisons[condition.sid][key] = comparison

    return comparisons


if __name__ == "__main__":
    resources_path: Path = Path(__file__).parent / "resources"
    results_dir: Path = Path.cwd() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    compare_simulators(
        model_path=resources_path / "icg_liver.xml",
        conditions_path=resources_path / "condition_liver.tsv",
        timepoints=np.linspace(start=0, stop=10, num=51),
        results_dir=results_dir,
    )
