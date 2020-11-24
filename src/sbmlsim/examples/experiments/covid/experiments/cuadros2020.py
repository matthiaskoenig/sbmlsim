from pathlib import Path
from typing import Dict

from sbmlsim.experiment import ExperimentDict, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.task import Task


class Cuadros2020(SimulationExperiment):
    def models(self) -> Dict[str, AbstractModel]:
        Q_ = self.Q_
        models = {
            "model": AbstractModel(
                source=Path(__file__).parent
                / ".."
                / "models"
                / "Cuadros2020"
                / "Cuadros2020.xml",
                language_type=AbstractModel.LanguageType.SBML,
                changes={},
            )
        }
        return ExperimentDict(models)

    def tasks(self) -> Dict[str, Task]:
        if self.simulations():
            return ExperimentDict(
                {
                    f"task_{key}": Task(model="model", simulation=key)
                    for key in self.simulations()
                }
            )

    def simulations(self) -> Dict[str, TimecourseSim]:
        Q_ = self.Q_

        tcsims = {}
        tcsims["sim1"] = TimecourseSim(
            [
                Timecourse(
                    start=0,
                    end=75,
                    steps=74,
                    changes={},
                )
            ]
        )
        return tcsims

    def figures(self) -> Dict[str, Figure]:
        return {**self.figure_plot_1()}

    def figure_plot_1(self):
        unit_time = "time"
        unit_y = "substance"

        fig_1 = Figure(self, sid="plot_1", name=f"{self.sid} (plot_1)")
        plots = fig_1.create_plots(Axis("time", unit=unit_time), legend=True)
        plots[0].set_yaxis("Total_cumulative_cases", unit_y)

        # simulation
        plots[0].add_data(
            task="task_sim1",
            xid="time",
            yid="Total_cumulative_cases",
            label="Total_cumulative_cases",
            color="black",
            linewidth=2,
        )

        fig_3 = Figure(self, sid="plot_3", name=f"{self.sid} (plot_3)")
        plots = fig_3.create_plots(Axis("time", unit=unit_time), legend=True)
        plots[0].set_yaxis("Total_deaths", unit_y)

        # simulation
        plots[0].add_data(
            task="task_sim1",
            xid="time",
            yid="Total_deaths",
            label="Total_deaths",
            color="black",
            linewidth=2,
        )

        return {"plot_1": fig_1, "plot_3": fig_3}
