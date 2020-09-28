"""
Example simulation experiment.

Various scans.
"""
from pathlib import Path
from typing import Dict

import numpy as np

from sbmlsim.data import Data
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.simulation import (
    AbstractSim,
    Dimension,
    ScanSim,
    Timecourse,
    TimecourseSim,
)
from sbmlsim.simulation.sensititvity import ModelSensitivity, SensitivityType
from sbmlsim.simulator.simulation_ray import SimulatorParallel, SimulatorSerial
from sbmlsim.task import Task
from sbmlsim.test import MODEL_DEMO


class DemoExperiment(SimulationExperiment):
    """Simple repressilator experiment."""

    def models(self) -> Dict[str, AbstractModel]:
        return {"model": RoadrunnerSBMLModel(source=MODEL_DEMO, ureg=self.ureg)}

    def tasks(self) -> Dict[str, Task]:
        return {
            f"task_{key}": Task(model="model", simulation=key)
            for key in self.simulations()
        }

    def simulations(self) -> Dict[str, AbstractSim]:
        return {
            **self.sim_scans(),
        }

    def sim_scans(self) -> Dict[str, AbstractSim]:
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
                    "dim_init", changes={"[e__A]": Q_(np.linspace(5, 15, num=10), "mM")}
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

    def figures(self) -> Dict[str, Figure]:
        unit_time = "min"
        unit_data = "mM"

        fig1 = Figure(experiment=self, sid="Fig1", num_cols=2, num_rows=2)
        plots = fig1.create_plots(
            xaxis=Axis("time", unit=unit_time),
            yaxis=Axis("data", unit=unit_data),
            legend=True,
        )
        for k in [0, 2]:
            for key in ["[e__A]", "[e__B]", "[e__C]", "[c__A]", "[c__B]", "[c__C]"]:
                task_id = "task_scan_init"
                plots[k].curve(
                    x=Data(self, "time", task=task_id, unit=unit_time),
                    y=Data(self, key, task=task_id, unit=unit_data),
                    label=key,
                )
        plots[2].yaxis.scale = "log"

        return {
            fig1.sid: fig1,
        }


def run(output_path):
    """Run the example."""
    base_path = Path(__file__).parent
    data_path = base_path
    simulator = SimulatorParallel()

    exp = DemoExperiment(simulator=simulator, data_path=data_path, base_path=base_path)
    exp.run(output_path=output_path / "results", show_figures=True)


if __name__ == "__main__":
    output_path = Path(".")
    run(output_path=output_path)
