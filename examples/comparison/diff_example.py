"""Comparison of simulation results of the repressilator with JWS Online.

`DataSetsComparison` (`sbmlsim.comparison.diff`) compares the results of a
simulation between simulators: it matches the columns of the data frames or
datasets, computes the absolute and relative differences on the shared time
points and reports which of them are above the tolerances.

The example simulates the six timecourses of `diff/example_*.json` with
roadrunner and compares them against the results of the same simulations on
[JWS Online](https://jjj.bio.vu.nl) in `diff/jws/`. The report of every
comparison is written into the working directory as `<example>_diff.tsv` with
the figure `<example>_diff.png`.
"""

from pathlib import Path

import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from sbmlsim.comparison.diff import DataSetsComparison, get_files_by_extension
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator

#: the simulations and the reference results of JWS Online
DIFF_DIR: Path = Path(__file__).parent / "diff"


def simulate_examples() -> dict[str, xr.Dataset]:
    """Simulate the example timecourses with roadrunner.

    The tolerances of the integrator are tightened, the comparison is of the
    order of the tolerance of the looser of the two simulators.

    Returns:
        The results of the simulations, by the key of the example.
    """
    simulator = Simulator(absolute_tolerance=1e-16, relative_tolerance=1e-13)
    model = simulator.load(REPRESSILATOR_SBML)

    results: dict[str, xr.Dataset] = {}
    for key, json_path in sorted(get_files_by_extension(DIFF_DIR).items()):
        simulation = Simulation.from_json(json_path)
        results[key] = simulator.run(model, simulation).ds
    return results


def compare_examples(results: dict[str, xr.Dataset], output_path: Path) -> None:
    """Compare the simulations with the results of JWS Online.

    Args:
        results: results of the simulations, by the key of the example.
        output_path: directory the reports and figures are written into.
    """
    for key, ds_sbmlsim in results.items():
        df_jws = pd.read_csv(DIFF_DIR / "jws" / f"{key}.tsv", sep="\t")
        # JWS Online reports the time of a change twice, before and after it;
        # `sbmlsim` reports the state after the change
        df_jws = df_jws.drop_duplicates(
            subset=df_jws.columns[0], keep="last"
        ).reset_index(drop=True)

        comparison = DataSetsComparison(
            dfs_dict={"sbmlsim": ds_sbmlsim, "jws": df_jws},
            title=f"{key} (sbmlsim | jws)",
        )
        fig = comparison.report()
        fig.savefig(output_path / f"{key}_diff.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        report_path = output_path / f"{key}_diff.tsv"
        report_path.write_text(comparison.report_str(), encoding="utf-8")


def diff_example(output_path: Path) -> None:
    """Run the simulations and compare them with JWS Online.

    Args:
        output_path: directory the reports and figures are written into.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    compare_examples(results=simulate_examples(), output_path=output_path)


if __name__ == "__main__":
    diff_example(output_path=Path.cwd() / "results")
