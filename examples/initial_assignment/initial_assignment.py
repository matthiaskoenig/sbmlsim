"""
Example simulation experiment.
"""

from pathlib import Path

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
from sbmlsim.simulation import AbstractSim, Timecourse, TimecourseSim
from sbmlsim.simulator.simulation_serial import SimulatorSerial
from sbmlsim.task import Task

base_path = Path(__file__).parent


class AssignmentExperiment(SimulationExperiment):
    """Testing initial assignments."""

    def models(self) -> dict[str, AbstractModel | Path]:
        return {
            "model": RoadrunnerSBMLModel(
                source=base_path / "initial_assignment.xml", ureg=self.ureg
            ),
            "model_changes": RoadrunnerSBMLModel(
                source=base_path / "initial_assignment.xml",
                ureg=self.ureg,
                changes={"D": self.Q_(2.0, "mmole")},
            ),
        }

    def tasks(self) -> dict[str, Task]:
        tasks = {}
        for model_key in self._models:
            for sim_key in self._simulations:
                tasks[f"task_{model_key}_{sim_key}"] = Task(
                    model=model_key, simulation=sim_key
                )
        return tasks

    def simulations(self) -> dict[str, AbstractSim]:
        Q_ = self.Q_
        tcs = {}
        tcs["sim1"] = TimecourseSim(
            [Timecourse(start=0, end=20, steps=200, changes={})]
        )
        tcs["sim2"] = TimecourseSim(
            [
                Timecourse(start=0, end=20, steps=200, changes={}),
                Timecourse(
                    start=0,
                    end=10,
                    steps=200,
                    changes={
                        "D": Q_(3.0, "mmole"),
                    },
                ),
            ]
        )

        return tcs

    def figures(self) -> dict[str, Figure]:
        unit_time = "min"
        unit_amount = "mmole"
        unit_concentration = "mM"

        self.add_selections_data(selections=["time", "A1", "[A1]", "D"])

        fig1 = Figure(experiment=self, sid="Fig1", num_cols=2, num_rows=2)
        plots = fig1.create_plots(
            xaxis=Axis("time", unit=unit_time),
            legend=True,
        )
        plots[0].set_yaxis("amount", unit=unit_amount)
        plots[1].set_yaxis("concentration", unit=unit_concentration)
        plots[2].set_yaxis("D", unit=unit_amount)

        colors = ["black", "blue", "red"]
        for ks, sim_key in enumerate(self._simulations.keys()):
            for _km, model_key in enumerate(self._models.keys()):
                task_key = f"task_{model_key}_{sim_key}"

                style = Style(
                    line=Line(
                        color=ColorType(colors[ks]),
                        type=LineType.SOLID if model_key == "model" else LineType.DASH,
                    ),
                    marker=Marker(type=MarkerType.NONE),
                )
                for plot, yid in zip(plots[:3], ["A1", "[A1]", "D"], strict=True):
                    plot.add_data(
                        task=task_key,
                        xid="time",
                        yid=yid,
                        label=f"{model_key} {sim_key}",
                        type=CurveType.POINTS,
                        style=style,
                    )
        return {"fig1": fig1}


def run(output_path):
    """Run the example."""
    base_path = Path(__file__).parent

    runner = ExperimentRunner(
        AssignmentExperiment,
        simulator=SimulatorSerial(),
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
