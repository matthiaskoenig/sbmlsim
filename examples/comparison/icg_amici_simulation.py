"""Simulation of the ICG models with AMICI, compared with roadrunner.

AMICI is the engine `amici` of sbmlsim, the optional extra `amici`, `pip
install "sbmlsim[amici]"`. Every model is simulated by both engines at the
same output times and with tight tolerances, and the results are compared
column by column with `DataSetsComparison`.
"""

from pathlib import Path

import numpy as np

from sbmlsim.comparison.diff import DataSetsComparison
from sbmlsim.console import console
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulator import Simulator

#: the models, the second one gives the dose with an event
MODELS: list[Path] = [
    Path(__file__).parent / "resources" / "icg_sd.xml",
    Path(__file__).parent / "resources" / "icg_events_sd.xml",
]


def compare_engines(model_path: Path) -> DataSetsComparison:
    """Simulate a model with roadrunner and AMICI and compare the results."""
    # an intravenous dose of 10 mg of ICG after 1 min, simulated for 20 min.
    # The dose of `icg_events_sd` is the event `IVDOSE_icg > 0`: a change at
    # the start fires no event, and a pre-initialization of the dose fires it
    # in AMICI and not in roadrunner (#439), so the dose is a change after
    # the start, which fires the event in every engine
    simulation = Simulation(
        start=0,
        end=20,
        times=np.linspace(0, 20, num=21).tolist(),
        changes=[Change(1, {"IVDOSE_icg": 10})],
    )
    results = {}
    for engine in ("roadrunner", "amici"):
        simulator = Simulator(
            n_workers=1,
            engine=engine,
            absolute_tolerance=1e-12,
            relative_tolerance=1e-10,
        )
        model = simulator.load(AbstractModel(source=model_path))
        results[engine] = simulator.run(model, simulation)
    return DataSetsComparison(
        dfs_dict={engine: res.ds for engine, res in results.items()},
        title=model_path.stem,
    )


if __name__ == "__main__":
    for model_path in MODELS:
        console.rule(title=model_path.stem, align="left", style="white")
        comparison = compare_engines(model_path)
        console.print(f"equal: {comparison.is_equal()}")
        # the relative difference of a value near 0 says little, e.g. of a
        # rounding residue of opposite signs in the two engines
        largest = float(comparison.diff_abs.max().max())
        console.print(f"largest absolute difference: {largest:.1e}")
