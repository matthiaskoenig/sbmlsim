"""
Example simulation experiment.
"""
from pathlib import Path
from typing import Dict, Union

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure, Plot
from sbmlsim.simulation import (
    AbstractSim,
    Timecourse,
    TimecourseSim,
)
from sbmlsim.simulator.simulation_ray import SimulatorParallel, SimulatorSerial
from sbmlsim.task import Task
from sbmlsim.test import MODEL_REPRESSILATOR


class RepressilatorExperiment(SimulationExperiment):
    """Simple repressilator experiment."""

    def models(self) -> Dict[str, Union[Path, AbstractModel]]:
        """Define models."""
        return {
            "model1": MODEL_REPRESSILATOR,
            "model2": AbstractModel(
                MODEL_REPRESSILATOR, changes={
                    "ps_0": self.Q_(1.3E-5, "dimensionless"),
                    "ps_a": self.Q_(0.013, "dimensionless")
                }
            ),
        }

    def simulations(self) -> Dict[str, AbstractSim]:
        """Define simulations."""
        tc = TimecourseSim(
            timecourses=Timecourse(start=0, end=1000, steps=1000),
            time_offset=0,
        )
        return {
            "tc": tc
        }

    def tasks(self) -> Dict[str, Task]:
        """Define tasks."""
        tasks = dict()
        for model in ["model1", "model2"]:
            tasks[f"task_{model}_tc"] = Task(model=model, simulation="tc")
        return tasks

    def data(self) -> Dict[str, Data]:
        """Define data generators."""
        # direct access via id
        data = []
        for model in ["model1", "model2"]:
            for selection in ["time", "PX", "PY", "PZ"]:
                data.append(
                    Data(task=f"task_{model}_tc", index=selection)
                )

        # functions (calculated data generators)
        # FIXME: necessary to store units in the xres
        for sid in ["PX", "PY", "PZ"]:
            data.append(
                Data(
                    index=f"f_{sid}_normalized",
                    function=f"{sid}/max({sid})",
                    variables={
                        f"{sid}": Data(index=f'{sid}', task="task_model1_tc"),
                    },
                    parameters={}
                )
            )

        data_dict = {d.sid: d for d in data}
        from pprint import pprint
        pprint(data_dict)
        return data_dict

    def figures(self) -> Dict[str, Figure]:
        """Define figure outputs (plots)."""
        fig = Figure(experiment=self, sid="Repressilator example", num_cols=2, num_rows=2)

        # FIXME: add helper to easily create figure layouts with plots
        p0 = fig.add_subplot(Plot(sid="plot0", name="Timecourse"), row=1, col=1)
        p1 = fig.add_subplot(Plot(sid="plot1", name="Preprocessing"), row=1, col=2)
        p2 = fig.add_subplot(Plot(sid="plot2", name="Postprocessing"), row=2, col=1, col_span=2)

        p0.set_title(f"Timecourse")
        p0.set_xaxis("time", unit="second")
        p0.set_yaxis("data", unit="dimensionless")
        p1.set_title(f"Preprocessing")
        p1.set_xaxis("time", unit="second")
        p1.set_yaxis("data", unit="dimensionless")
        colors = ["tab:red", "tab:green", "tab:blue"]
        for k, sid in enumerate(["PX", "PY", "PZ"]):
            p0.curve(
                x=Data("time", task=f"task_model1_tc"),
                y=Data(f"{sid}", task=f"task_model1_tc"),
                label=f"{sid}",
                color=colors[k]
            )
            p1.curve(
                x=Data("time", task=f"task_model2_tc"),
                y=Data(f"{sid}", task=f"task_model2_tc"),
                label=f"{sid}",
                color=colors[k],
                linewidth=2.0,
            )

        p2.set_title(f"Postprocessing")
        p2.set_xaxis("data", unit="dimensionless")
        p2.set_yaxis("data", unit="dimensionless")

        colors2 = ["tab:orange", "tab:brown", "tab:purple"]
        for k, (sidx, sidy) in enumerate([("PX", "PZ"), ("PZ", "PY"), ('PY', 'PX')]):
            p2.curve(
                x=self._data[f"f_{sidx}_normalized"],
                y=self._data[f"f_{sidy}_normalized"],
                label=f"{sidy}/max({sidy}) ~ {sidx}/max({sidx})",
                color=colors2[k],
                linewidth=2.0,
            )
        return {
            fig.sid: fig,
        }

    def reports(self) -> Dict[str, Dict[str, Data]]:
        """Define reports.

        HashMap of DataGenerators.
        FIXME: separate class for these objects.
        """
        pass


def run_repressilator_experiments(output_path: Path) -> Path:
    """Run the repressilator simulation experiments."""
    base_path = Path(__file__).parent
    data_path = base_path

    runner = ExperimentRunner(
        [RepressilatorExperiment],
        simulator=SimulatorParallel(),
        data_path=data_path,
        base_path=base_path,
    )
    _results = runner.run_experiments(
        output_path=output_path / "results", show_figures=True
    )


if __name__ == "__main__":
    run_repressilator_experiments(Path(__file__).parent / "results")
