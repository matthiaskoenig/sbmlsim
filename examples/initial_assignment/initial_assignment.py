"""
Example simulation experiment.
"""

from pathlib import Path
from typing import override

from sbmlsim import Q
from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.plot.plotting import (
    ColorType,
    CurveType,
    Line,
    LineType,
    Marker,
    MarkerType,
    Style,
)
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task

#: model with an initial assignment, the initial amount of `A1` is twice the dose `D`
MODEL_PATH = Path(__file__).parent / "initial_assignment.xml"


class AssignmentExperiment(SimulationExperiment):
    """Testing initial assignments."""

    @override
    def models(self) -> dict[str, AbstractModel | Path]:
        """Define models, with and without a change of the dose."""
        return {
            "model": RoadrunnerSBMLModel(source=MODEL_PATH, ureg=self.ureg),
            "model_changes": RoadrunnerSBMLModel(
                source=MODEL_PATH,
                ureg=self.ureg,
                changes={"D": Q(2.0, "mmole")},
            ),
        }

    @override
    def simulations(self) -> dict[str, Simulation]:
        """Define simulations."""
        return {
            "sim1": Simulation(end=20, steps=200),
            # the change at the time 20 sets the dose, the initial assignment of
            # the model is not evaluated again
            "sim2": Simulation(
                end=30, steps=400, changes=[Change(20, {"D": Q(3.0, "mmole")})]
            ),
        }

    @override
    def tasks(self) -> dict[str, Task]:
        """Define tasks, every simulation on every model."""
        return {
            f"task_{model_key}_{sim_key}": Task(model=model_key, simulation=sim_key)
            for model_key in self._models
            for sim_key in self._simulations
        }

    @override
    def data(self) -> dict[str, Data]:
        """Define the data of the experiment."""
        self.add_selections_data(selections=["time", "A1", "[A1]", "D"])
        return {}

    @override
    def figures(self) -> dict[str, Figure]:
        """Define figure outputs (plots)."""
        unit_time = "min"
        unit_amount = "mmole"
        unit_concentration = "mM"

        fig1 = Figure(experiment=self, sid="Fig1", num_cols=3, num_rows=1)
        plots = fig1.create_plots(
            xaxis=Axis("time", unit=unit_time),
            legend=True,
        )
        plots[0].set_yaxis("amount", unit=unit_amount)
        plots[1].set_yaxis("concentration", unit=unit_concentration)
        plots[2].set_yaxis("D", unit=unit_amount)

        colors = ["black", "blue", "red"]
        for ks, sim_key in enumerate(self._simulations):
            for model_key in self._models:
                task_key = f"task_{model_key}_{sim_key}"

                style = Style(
                    line=Line(
                        color=ColorType(colors[ks]),
                        type=LineType.SOLID if model_key == "model" else LineType.DASH,
                    ),
                    marker=Marker(type=MarkerType.NONE),
                )
                for plot, yid in zip(plots, ["A1", "[A1]", "D"], strict=True):
                    plot.add_data(
                        task=task_key,
                        xid="time",
                        yid=yid,
                        label=f"{model_key} {sim_key}",
                        type=CurveType.POINTS,
                        style=style,
                    )
        return {"fig1": fig1}


def run(output_path: Path) -> None:
    """Run the example."""
    base_path = Path(__file__).parent

    runner = ExperimentRunner(
        AssignmentExperiment,
        simulator=Simulator(),
        base_path=base_path,
        data_path=base_path,
    )
    runner.run_experiments(
        output_path=output_path / "results",
        show_figures=False,
        reduced_selections=False,
    )


if __name__ == "__main__":
    run(output_path=Path.cwd())
