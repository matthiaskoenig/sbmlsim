"""
Example simulation experiment.
"""

from pathlib import Path

from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.simulation import AbstractSim, Timecourse, TimecourseSim
from sbmlsim.simulator.simulation_serial import SimulatorSerial
from sbmlsim.task import Task


base_path = Path(__file__).parent


class AssignmentExperiment(SimulationExperiment):
    """Testing initial assignments."""

    def models(self) -> dict[str, AbstractModel]:
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
        for model_key in self._models.keys():
            for sim_key in self._simulations.keys():
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
            for km, model_key in enumerate(self._models.keys()):
                task_key = f"task_{model_key}_{sim_key}"

                kwargs = {
                    "color": colors[ks],
                    "linestyle": "-" if model_key == "model" else "--",
                }
                plots[0].add_data(
                    task=task_key,
                    xid="time",
                    yid="A1",
                    label=f"{model_key} {sim_key}",
                    **kwargs,
                )
                plots[1].add_data(
                    task=task_key,
                    xid="time",
                    yid="[A1]",
                    label=f"{model_key} {sim_key}",
                    **kwargs,
                )
                plots[2].add_data(
                    task=task_key,
                    xid="time",
                    yid="D",
                    label=f"{model_key} {sim_key}",
                    **kwargs,
                )
        return {
            fig1.sid: fig1,
        }


def run(output_path):
    """Run the example."""
    base_path = Path(__file__).parent

    runner = ExperimentRunner(
        AssignmentExperiment,
        simulator=SimulatorSerial(),
        base_path=base_path,
        data_path=base_path,
    )
    runner.run_experiments(output_path=output_path / "results", show_figures=True)


if __name__ == "__main__":
    output_path = Path(".")
    run(output_path=output_path)
