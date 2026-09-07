"""
Example simulation experiment.

Various scans.
"""

from pathlib import Path

import numpy as np

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.resources import DEMO_SBML
from sbmlsim.simulation import (
    AbstractSim,
    Dimension,
    ScanSim,
    Timecourse,
    TimecourseSim,
)
from sbmlsim.simulation.sensitivity import ModelSensitivity
from sbmlsim.simulator.simulation_serial import SimulatorSerial
from sbmlsim.task import Task


class DemoExperiment(SimulationExperiment):
    """Simple repressilator experiment."""

    def models(self) -> dict[str, AbstractModel]:
        """Define models."""
        return {"model": RoadrunnerSBMLModel(source=DEMO_SBML, ureg=self.ureg)}

    def tasks(self) -> dict[str, Task]:
        """Define tasks."""
        return {
            f"task_{key}": Task(model="model", simulation=key)
            for key in self.simulations()
        }

    def simulations(self) -> dict[str, AbstractSim]:
        """Define simulations."""
        return {
            **self.sim_scans(),
        }

    def sim_scans(self) -> dict[str, AbstractSim]:
        Q_ = self.Q_
        scan_init = ScanSim(
            simulation=TimecourseSim(
                [
                    Timecourse(
                        start=0, end=10, steps=100, changes={"[e__A]": Q_(10, "mM")}
                    ),
                    Timecourse(
                        start=0, end=10, steps=100, changes={"[e__B]": Q_(10, "mM")}
                    ),
                ]
            ),
            dimensions=[
                Dimension(
                    "dim_init", changes={"[e__A]": Q_(np.linspace(5, 15, num=11), "mM")}
                ),
                ModelSensitivity.create_difference_dimension(
                    model=self._models["model"],
                    difference=0.5,
                ),
            ],
            mapping={"dim_init": 0, "dim_sens": 0},
        )

        return {
            "scan_init": scan_init,
        }

    def figures(self) -> dict[str, Figure]:
        # print(self._results.keys())
        # print(self._results["task_scan_init"])

        unit_time = "min"
        unit_data = "mM"

        selections = ["[e__A]", "[e__B]", "[e__C]", "[c__A]", "[c__B]", "[c__C]"]
        self.add_selections_data(selections=["time"] + selections)

        fig1 = Figure(experiment=self, sid="Fig1", num_cols=2, num_rows=1)
        plots = fig1.create_plots(
            xaxis=Axis("time", unit=unit_time),
            yaxis=Axis("data", unit=unit_data),
            legend=True,
        )
        for k in [0, 1]:
            for key in selections:
                task_id = "task_scan_init"

                # This should plot the individual curve(s), i.e. in a scan the
                # additional dimensions have to be iterated over
                plots[k].curve(
                    x=Data("time", task=task_id),
                    y=Data(key, task=task_id),
                    label=key,
                )
        plots[1].yaxis.scale = "log"

        return {
            fig1.sid: fig1,
        }


def run_demo_experiments(output_path: Path) -> None:
    """Run the example."""
    base_path = Path(__file__).parent
    data_path = base_path

    runner = ExperimentRunner(
        DemoExperiment,
        simulator=SimulatorSerial(),
        data_path=data_path,
        base_path=base_path,
    )
    _results = runner.run_experiments(
        output_path=output_path / "results", show_figures=True, reduced_selections=False
    )


if __name__ == "__main__":
    output_path = Path(".")
    run_demo_experiments(output_path=output_path)
