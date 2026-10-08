"""
Example simulation experiment.
"""

from pathlib import Path
from typing import override

from sbmlsim import Q
from sbmlsim.data import Data
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.experiment.runner import run_experiments
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Figure, Plot
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Simulation
from sbmlsim.task import Task


class RepressilatorExperiment(SimulationExperiment):
    """Simple repressilator experiment."""

    @override
    def models(self) -> dict[str, Path | AbstractModel]:
        """Define models."""
        return {
            "model1": REPRESSILATOR_SBML,
            "model2": AbstractModel(
                REPRESSILATOR_SBML,
                changes={
                    "ps_0": Q(1.3e-5, "dimensionless"),
                    "ps_a": Q(0.013, "dimensionless"),
                },
            ),
        }

    @override
    def simulations(self) -> dict[str, Simulation]:
        """Define simulations."""
        return {"tc": Simulation(end=1000, steps=1000)}

    @override
    def tasks(self) -> dict[str, Task]:
        """Define tasks."""
        return {
            f"task_{model}_tc": Task(model=model, simulation="tc")
            for model in ["model1", "model2"]
        }

    @override
    def data(self) -> dict[str, Data]:
        """Define the data of the experiment."""
        # direct access via id
        data = [
            Data(task=f"task_{model}_tc", index=selection)
            for model in ["model1", "model2"]
            for selection in ["time", "PX", "PY", "PZ"]
        ]

        # functions (calculated data), which carry the units of
        # their variables
        for sid in ["PX", "PY", "PZ"]:
            data.append(
                Data(
                    index=f"f_{sid}_normalized",
                    function=f"{sid}/max({sid})",
                    variables={
                        sid: Data(index=f"{sid}", task="task_model1_tc"),
                    },
                )
            )

        return {d.sid: d for d in data}

    @override
    def figures(self) -> dict[str, Figure]:
        """Define figure outputs (plots)."""
        fig = Figure(
            experiment=self,
            sid="figure0",
            name="Repressilator",
            num_cols=2,
            num_rows=2,
            width=10,
            height=10,
        )
        p0 = fig.add_subplot(Plot(sid="plot0", name="Timecourse"), row=1, col=1)
        p1 = fig.add_subplot(Plot(sid="plot1", name="Changes"), row=1, col=2)
        p2 = fig.add_subplot(
            Plot(sid="plot2", name="Postprocessing"), row=2, col=1, col_span=2
        )

        p0.set_title("Timecourse")
        p0.set_xaxis("time", unit="second")
        p0.set_yaxis("data", unit="dimensionless")
        p1.set_title("Timecourse with model changes")
        p1.set_xaxis("time", unit="second")
        p1.set_yaxis("data", unit="dimensionless")
        colors = ["tab:red", "tab:green", "tab:blue"]
        for k, sid in enumerate(["PX", "PY", "PZ"]):
            p0.curve(
                x=Data("time", task="task_model1_tc"),
                y=Data(f"{sid}", task="task_model1_tc"),
                label=f"{sid}",
                color=colors[k],
            )
            p1.curve(
                x=Data("time", task="task_model2_tc"),
                y=Data(f"{sid}", task="task_model2_tc"),
                label=f"{sid}",
                color=colors[k],
                linewidth=2.0,
            )

        p2.set_title("Postprocessing")
        p2.set_xaxis("data", unit="dimensionless")
        p2.set_yaxis("data", unit="dimensionless")

        colors2 = ["tab:orange", "tab:brown", "tab:purple"]
        for k, (sidx, sidy) in enumerate([("PX", "PZ"), ("PZ", "PY"), ("PY", "PX")]):
            p2.curve(
                x=self._data[f"f_{sidx}_normalized"],
                y=self._data[f"f_{sidy}_normalized"],
                label=f"{sidy}/max({sidy}) ~ {sidx}/max({sidx})",
                color=colors2[k],
                linewidth=2.0,
            )
        return {"fig1": fig}


def run_repressilator_example(output_path: Path) -> None:
    """Run repressilator example."""
    # run sbmlsim experiment
    run_experiments(
        experiments=RepressilatorExperiment,
        output_path=output_path / "sbmlsim",
    )


if __name__ == "__main__":
    run_repressilator_example(output_path=Path.cwd() / "results")
