"""
Example simulation experiment.
"""
import numpy as np
from typing import Dict
from pathlib import Path

from sbmlsim.tests.constants import MODEL_REPRESSILATOR

from sbmlsim.experiment import SimulationExperiment
from sbmlsim.models import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.data import Data, DataSet
from sbmlsim.timecourse import AbstractSim, Timecourse, TimecourseSim
from sbmlsim.scan import ParameterScan
from sbmlsim.tasks import Task
from sbmlsim.plotting import Figure, Axis


class RepressilatorExperiment(SimulationExperiment):
    """Simple repressilator experiment."""
    def models(self) -> Dict[str, AbstractModel]:
        return {
            'model1': RoadrunnerSBMLModel(source=MODEL_REPRESSILATOR,
                                          ureg=self.ureg)
        }

    def tasks(self) -> Dict[str, Task]:
        return {
            f'task_{key}': Task(model='model1', simulation=key) for key in self.simulations()
        }

    def simulations(self) -> Dict[str, AbstractSim]:
        """
        Simulation time is in [s]
        :return:
        """
        Q_ = self.Q_
        unit_data = "dimensionless"
        # simple timecourse
        tc = TimecourseSim([
            Timecourse(start=0, end=600, steps=2000),
            Timecourse(start=0, end=600, steps=2000,
                       changes={"X": Q_(10, unit_data),
                                "Y": Q_(20, unit_data)
                                }),
        ])

        scan1d = ParameterScan(
            TimecourseSim([Timecourse(start=0, end=600, steps=2000)]),
            scan={
                'X': Q_(np.linspace(0, 10, num=11), unit_data),
            }
        )
        scan2d = ParameterScan(
            TimecourseSim([Timecourse(start=0, end=600, steps=2000)]),
            scan={
                'X': Q_(np.linspace(0, 10, num=11), unit_data),
                'Y': Q_(np.linspace(0, 10, num=11), unit_data),
            }
        )
        scan3d = ParameterScan(
            TimecourseSim([Timecourse(start=0, end=600, steps=2000)]),
            scan={
                'X': Q_(np.linspace(0, 10, num=5), unit_data),
                'Y': Q_(np.linspace(0, 10, num=5), unit_data),
                'Z': Q_(np.linspace(0, 10, num=5), unit_data),
            }
        )

        return {
            "tc": tc,
            "scan1d": scan1d,
            "scan2d": scan2d,
            "scan3d": scan3d,
        }

    def datagenerators(self) -> None:
        """Data to plot and analyze.

        :return:
        """
        unit_time = "min"
        unit_data = "dimensionless"

        # Data accessed
        Data(self, task="task_tc", index="X", unit=unit_data)
        Data(self, task="task_tc", index="Y", unit=unit_data)
        Data(self, task="task_tc", index="Z", unit=unit_data)

        # Define functions (data generators)
        Data(
            self, index="f1", function="(sin(X)+Y+Z)/max(X)", unit="dimensionless",
            variables={
                "X": 'task_tc__X',
                "Y": 'task_tc__Y',
                "Z": 'task_tc__Z',
            })
        Data(
            self, index="f2", function="Y/max(Y)", unit="dimensionless",
            variables={
                "Y": 'task_tc__Y',
            }
        )
        # FIXME: arbitrary processing
        # [3] arbitrary processing (e.g. pharmacokinetic calculations)
        # Processing(variables) # arbitrary functions
        # Aggregation over

    def figures(self) -> Dict[str, Figure]:
        unit_time = "min"
        unit_data = "dimensionless"

        fig1 = Figure(experiment=self,
                     sid="Fig1", num_cols=1, num_rows=1)
        plots = fig1.create_plots(
            xaxis=Axis("time", unit=unit_time),
            yaxis=Axis("data", unit=unit_data),
            legend=True
        )
        plots[0].set_title(f"{self.sid}_{fig1.sid}")
        plots[0].curve(
            x=Data(self, "time", task="task_tc", unit=unit_time),
            y=Data(self, "X", task="task_tc", unit=unit_data),
            label="X sim", color="black"
        )
        plots[0].curve(
            x=Data(self, "time", task="task_tc", unit=unit_time),
            y=Data(self, "Y", task="task_tc", unit=unit_data),
            label="Y sim", color="blue"
        )

        fig2 = Figure(experiment=self,
                     sid="Fig2", num_rows=2, num_cols=1)
        plots = fig2.create_plots(
            xaxis=Axis("data", unit=unit_data),
            yaxis=Axis("data", unit=unit_data),
            legend=True
        )
        plots[0].curve(x=self._data['f1'], y=self._data['f2'],
            label="f2 ~ f1", color="black", marker="o", alpha=0.3
        )
        plots[1].curve(
            x=self._data['f1'], y=self._data['f2'],
            label="f2 ~ f1", color="black", marker="o", alpha=0.3
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


def run(output_path):
    """Run the example."""
    base_path = Path(__file__).parent
    data_path = base_path

    exp = RepressilatorExperiment(
        data_path=data_path,
        base_path=base_path
    )
    results = exp.run(
        output_path=output_path / RepressilatorExperiment.__name__,
        show_figures=True
    )
    print(results)


if __name__ == "__main__":
    output_path = Path(".")
    run(output_path=output_path)




