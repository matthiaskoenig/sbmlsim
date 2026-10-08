"""
Example simulation experiment.
"""

from pathlib import Path
from typing import override

import numpy as np

from sbmlsim import Q
from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task


class RepressilatorScanExperiment(SimulationExperiment):
    """Scans of the repressilator over the values of X and Y."""

    @override
    def models(self) -> dict[str, Path | AbstractModel]:
        """Define models."""
        return {
            "model1": REPRESSILATOR_SBML,
            "model2": AbstractModel(
                REPRESSILATOR_SBML, changes={"X": Q(100, "dimensionless")}
            ),
        }

    @override
    def simulations(self) -> dict[str, Simulation | Scan]:
        """Define the timecourse and the scans of it."""
        unit_data = "dimensionless"
        rng = np.random.default_rng(seed=1234)
        tc = Simulation(
            end=200,
            steps=4000,
            changes=[Change(100, {"X": Q(10, unit_data), "Y": Q(20, unit_data)})],
        )

        scan1d = Scan(
            simulation=tc,
            dimensions=[
                # the scan sets X at the start, the simulation sets it again at
                # the time 100
                Dimension(
                    "dim1",
                    values={"X": Q(np.linspace(0, 10, num=11), unit_data)},
                    at=0,
                )
            ],
        )
        scan2d = Scan(
            simulation=tc,
            dimensions=[
                Dimension(
                    "dim1",
                    values={"X": Q(rng.normal(5, 2, size=10), unit_data)},
                    at=0,
                ),
                Dimension(
                    "dim2",
                    values={"Y": Q(rng.normal(5, 2, size=10), unit_data)},
                ),
            ],
        )
        return {"tc": tc, "scan1d": scan1d, "scan2d": scan2d}

    @override
    def tasks(self) -> dict[str, Task]:
        """Define tasks, every simulation on every model."""
        return {
            f"task_{model}_{sim_key}": Task(model=model, simulation=sim_key)
            for model in ["model1", "model2"]
            for sim_key in self._simulations
        }

    @override
    def data(self) -> dict[str, Data]:
        """Define the data of the experiment."""
        # accessed data
        data = [
            Data(task=f"task_{model}_tc", index=selection)
            for model in ["model1", "model2"]
            for selection in ["time", "X", "Y", "Z"]
        ]

        # functions (calculated data)
        data.extend(
            [
                Data(
                    index="f1",
                    function="(sin(X)+Y+Z)/max(X)",
                    variables={
                        "X": Data(index="X", task="task_model1_tc"),
                        "Y": Data(index="Y", task="task_model1_tc"),
                        "Z": Data(index="Z", task="task_model1_tc"),
                    },
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
        return {d.sid: d for d in data}

    @override
    def figures(self) -> dict[str, Figure]:
        """Define figure outputs (plots)."""
        unit_time = "second"
        unit_data = "dimensionless"

        fig1 = Figure(experiment=self, sid="Fig1", num_cols=1, num_rows=1)
        plots = fig1.create_plots(
            xaxis=Axis("time", unit=unit_time),
            yaxis=Axis("data", unit=unit_data),
            legend=True,
        )
        plots[0].set_title(f"{self.sid}_{fig1.sid}")
        for model, linestyle in [("model1", "-"), ("model2", "--")]:
            task_id = f"task_{model}_tc"
            for sid, color in [("X", "black"), ("Y", "blue")]:
                plots[0].curve(
                    x=Data("time", task=task_id),
                    y=Data(sid, task=task_id),
                    label=f"{sid} {model}",
                    color=color,
                    linestyle=linestyle,
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

        plots[0].set_xaxis("data", unit=unit_data, min=-1.0, max=2.0, grid=True)
        plots[1].set_xaxis("data", unit=unit_data, scale="log")
        plots[1].set_yaxis("data", unit=unit_data, scale="log")

        return {"fig1": fig1, "fig2": fig2}


def run_repressilator_experiments(output_path: Path) -> None:
    """Run the repressilator simulation experiments."""
    base_path = Path(__file__).parent

    runner = ExperimentRunner(
        [RepressilatorScanExperiment],
        simulator=Simulator(),
        data_path=base_path,
        base_path=base_path,
    )
    runner.run_experiments(output_path=output_path / "results", show_figures=False)


if __name__ == "__main__":
    run_repressilator_experiments(Path.cwd())
