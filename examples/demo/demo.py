"""
Example simulation experiment.

Various scans.
"""

from pathlib import Path
from typing import override

import numpy as np

from sbmlsim import Q
from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.resources import DEMO_SBML
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulation.sensitivity import ModelSensitivity
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task

#: concentrations of the demo model in the external (e) and the cell (c) compartment
SELECTIONS = ["[e__A]", "[e__B]", "[e__C]", "[c__A]", "[c__B]", "[c__C]"]


class DemoExperiment(SimulationExperiment):
    """Scans of the demo model over its initial values and parameters."""

    @override
    def models(self) -> dict[str, AbstractModel | Path]:
        """Define models."""
        return {"model": RoadrunnerSBMLModel(source=DEMO_SBML, ureg=self.ureg)}

    @override
    def simulations(self) -> dict[str, Simulation | Scan]:
        """Define scan simulation."""
        return {
            "scan_init": Scan(
                simulation=Simulation(
                    end=20,
                    steps=200,
                    preinit_changes={"[e__A]": Q(10, "mM")},
                    changes=[Change(10, {"[e__B]": Q(10, "mM")})],
                ),
                dimensions=[
                    Dimension(
                        "dim_init",
                        values={"[e__A]": Q(np.linspace(5, 15, num=11), "mM")},
                    ),
                    ModelSensitivity.create_difference_dimension(
                        model=self._models["model"],
                        difference=0.5,
                    ),
                ],
            )
        }

    @override
    def tasks(self) -> dict[str, Task]:
        """Define tasks."""
        return {
            f"task_{key}": Task(model="model", simulation=key)
            for key in self._simulations
        }

    @override
    def data(self) -> dict[str, Data]:
        """Define the data of the experiment."""
        self.add_selections_data(selections=["time", *SELECTIONS])
        return {}

    @override
    def figures(self) -> dict[str, Figure]:
        """Define figure outputs (plots)."""
        unit_time = "second"
        unit_data = "mM"
        task_id = "task_scan_init"

        fig1 = Figure(experiment=self, sid="Fig1", num_cols=2, num_rows=1)
        plots = fig1.create_plots(
            xaxis=Axis("time", unit=unit_time),
            yaxis=Axis("data", unit=unit_data),
            legend=True,
        )
        for plot in plots:
            for key in SELECTIONS:
                # a curve of a scan draws the first point of the scan
                plot.curve(
                    x=Data("time", task=task_id),
                    y=Data(key, task=task_id),
                    label=key,
                )
        plots[1].set_yaxis("data", unit=unit_data, scale="log")

        return {"fig1": fig1}


def run_demo_experiments(output_path: Path) -> None:
    """Run the example."""
    base_path = Path(__file__).parent

    runner = ExperimentRunner(
        DemoExperiment,
        simulator=Simulator(),
        data_path=base_path,
        base_path=base_path,
    )
    runner.run_experiments(
        output_path=output_path / "results",
        show_figures=False,
        reduced_selections=False,
    )


if __name__ == "__main__":
    run_demo_experiments(output_path=Path.cwd())
