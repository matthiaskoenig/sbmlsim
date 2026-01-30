"""
Example simulation experiment.
"""

from pathlib import Path
from typing import Union

import numpy as np

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import (
    AbstractSim,
    Dimension,
    ScanSim,
    Timecourse,
    TimecourseSim,
)
from sbmlsim.simulator.simulation_serial import SimulatorSerial
from sbmlsim.task import Task


class RepressilatorScanExperiment(SimulationExperiment):
    """Simple repressilator experiment."""

    def models(self) -> dict[str, Union[Path, AbstractModel]]:
        return {
            "model1": REPRESSILATOR_SBML,
            "model2": AbstractModel(
                REPRESSILATOR_SBML, changes={"X": self.Q_(100, "dimensionless")}
            ),
        }

    def simulations(self) -> dict[str, AbstractSim]:
        return {
            **self.sim_scans(),
            # **self.sim_sensitivities(),
        }

    def tasks(self) -> dict[str, Task]:
        tasks = dict()
        for model in ["model1", "model2"]:
            for sim_key in self.simulations():
                tasks[f"task_{model}_{sim_key}"] = Task(model=model, simulation=sim_key)
        return tasks

    def sim_scans(self) -> dict[str, AbstractSim]:
        Q_ = self.Q_
        unit_data = "dimensionless"
        tc = TimecourseSim(
            [
                Timecourse(start=0, end=100, steps=2000),
                Timecourse(
                    start=0,
                    end=100,
                    steps=2000,
                    changes={"X": Q_(10, unit_data), "Y": Q_(20, unit_data)},
                ),
            ]
        )

        scan1d = ScanSim(
            simulation=tc,
            dimensions=[
                Dimension(
                    "dim1", changes={"X": Q_(np.linspace(0, 10, num=11), unit_data)}
                )
            ],
        )
        scan2d = ScanSim(
            simulation=tc,
            dimensions=[
                Dimension(
                    "dim1",
                    changes={"X": Q_(np.random.normal(5, 2, size=10), unit_data)},
                ),
                Dimension(
                    "dim2",
                    changes={"Y": Q_(np.random.normal(5, 2, size=10), unit_data)},
                ),
            ],
        )
        # scan3d = ScanSim(
        #     simulation=tc,
        #     dimensions=[
        #         Dimension(
        #             "dim1", changes={"X": Q_(np.linspace(0, 10, num=5), unit_data)}
        #         ),
        #         Dimension(
        #             "dim2", changes={"Y": Q_(np.linspace(0, 10, num=5), unit_data)}
        #         ),
        #         Dimension(
        #             "dim3", changes={"Z": Q_(np.linspace(0, 10, num=5), unit_data)}
        #         ),
        #     ],
        # )

        return {
            "tc": tc,
            "scan1d": scan1d,
            "scan2d": scan2d,
            # "scan3d": scan3d,
        }

    def data(self) -> dict[str, Data]:
        """Data used for plotting and analysis.
        Generates promises for results.

        :return:
        """
        data = []

        for model in ["model1", "model2"]:
            for selection in ["X", "Y", "Z"]:
                # accessed data
                data.append(Data(task=f"task_{model}_tc", index=selection))

        # Define functions (data generators)
        data.extend(
            [
                Data(
                    index="f1",
                    function="(sin(X)+Y+Z)/max(X)",
                    variables={
                        "X": Data(index="X", task="task_model1_tc"),
                        "Y": Data(index="Y", task="task_model1_tc"),
                        "Z": Data(index="Y", task="task_model1_tc"),
                    },
                    parameters={},
                ),
                Data(
                    index="f2",
                    function="Y/max(Y)",
                    variables={
                        "Y": Data(index="Y", task="task_model1_tc"),
                    },
                ),
            ]
        )

        # FIXME: arbitrary processing
        # [3] arbitrary processing (e.g. pharmacokinetic calculations)
        # Processing(variables) # arbitrary functions
        # Aggregation over

        return {d.sid: d for d in data}

    def figures(self) -> dict[str, Figure]:
        unit_time = "min"
        unit_data = "dimensionless"

        self.add_selections_data(
            selections=["time", "X", "Y"],
            task_ids=[f"task_{m}_tc" for m in ["model1", "model2"]],
        )

        fig1 = Figure(experiment=self, sid="Fig1", num_cols=1, num_rows=1)
        plots = fig1.create_plots(
            xaxis=Axis("time", unit=unit_time),
            yaxis=Axis("data", unit=unit_data),
            legend=True,
        )
        plots[0].set_title(f"{self.sid}_{fig1.sid}")
        for model in ["model1", "model2"]:
            task_id = f"task_{model}_tc"
            plots[0].curve(
                x=Data("time", task=task_id),
                y=Data("X", task=task_id),
                label="X sim",
                color="black",
            )
            plots[0].curve(
                x=Data("time", task=task_id),
                y=Data("Y", task=task_id),
                label="Y sim",
                color="blue",
            )

        fig2 = Figure(experiment=self, sid="Fig2", num_rows=2, num_cols=1)
        plots = fig2.create_plots(
            xaxis=Axis("data", unit=unit_data),
            yaxis=Axis("data", unit=unit_data),
            legend=True,
        )
        plots[0].curve(
            x=self._data["f1"],
            y=self._data["f2"],
            label="f2 ~ f1",
            color="black",
            marker="o",
            alpha=0.3,
        )
        plots[1].curve(
            x=self._data["f1"],
            y=self._data["f2"],
            label="f2 ~ f1",
            color="black",
            marker="o",
            alpha=0.3,
        )

        plots[0].xaxis.min = -1.0
        plots[0].xaxis.max = 2.0
        plots[0].xaxis.grid = True

        plots[1].xaxis.scale = "log"
        plots[1].yaxis.scale = "log"

        return {
            fig1.sid: fig1,
            fig2.sid: fig2,
        }


def run_repressilator_experiments(output_path: Path) -> Path:
    """Run the repressilator simulation experiments."""
    base_path = Path(__file__).parent
    data_path = base_path

    for simulator in [SimulatorSerial()]:
        runner = ExperimentRunner(
            [RepressilatorScanExperiment],
            simulator=simulator,
            data_path=data_path,
            base_path=base_path,
        )
        _results = runner.run_experiments(
            output_path=output_path / "results", show_figures=True
        )


if __name__ == "__main__":
    run_repressilator_experiments(Path(__file__).parent / "results")
