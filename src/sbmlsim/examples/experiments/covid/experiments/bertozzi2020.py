from pathlib import Path
from typing import Dict

from sbmlsim.experiment import ExperimentDict, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.task import Task


class Bertozzi2020(SimulationExperiment):
    def models(self) -> Dict[str, AbstractModel]:
        Q_ = self.Q_
        models = {
            "model": AbstractModel(
                source=Path(__file__).parent
                / ".."
                / "models"
                / "Bertozzi2020"
                / "Bertozzi2020.xml",
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

        Ro_CA = 1.9544
        Io_CA = [3956, 39.56, 0.3956, 0.003956]

        tcsims = {}
        for k, io_ca in enumerate(Io_CA):
            tcsims[f"sim{k}"] = TimecourseSim(
                [
                    Timecourse(
                        start=0,
                        end=215,
                        steps=281,
                        changes={
                            "Ro_CA": Q_(Ro_CA, "dimensionless"),
                            "Io_CA": Q_(io_ca, "dimensionless"),
                        },
                    )
                ]
            )

        return tcsims

    def figures(self) -> Dict[str, Figure]:
        unit_time = "time"
        unit_y = "substance"

        fig_1 = Figure(self, sid="plot_1", name=f"{self.sid} (plot_1)")
        plots = fig_1.create_plots(Axis("time", unit=unit_time), legend=True)
        plots[0].set_yaxis("y", unit_y)

        # simulation
        task_id = "task_sim0"
        colors = ["black", "blue", "red", "green"]

        for k, skey in enumerate(["Infected", "Susceptible", "Recovered", "Peak_Time"]):
            color = colors[k]
            plots[0].add_data(
                task=task_id,
                xid="time",
                yid=skey,
                label=skey,
                color=color,
                linewidth=2,
            )

        return {"plot_1": fig_1}
