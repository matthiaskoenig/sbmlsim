"""
Example simulation experiment.
"""

from typing import Dict
from pathlib import Path

from sbmlsim.tests.constants import MODEL_REPRESSILATOR

from sbmlsim.experiment import SimulationExperiment
from sbmlsim.models import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.data import Data, DataSet
from sbmlsim.processing.function import Function
from sbmlsim.timecourse import AbstractSim, Timecourse, TimecourseSim
from sbmlsim.tasks import Task
from sbmlsim.plotting import Figure, Axis


class RepressilatorExperiment(SimulationExperiment):
    """Simple repressilator experiment."""
    def models(self) -> Dict[str, AbstractModel]:
        return {
            'model1': RoadrunnerSBMLModel(source=MODEL_REPRESSILATOR,
                                          ureg=self.ureg)
        }

    def datasets(self) -> Dict[str, DataSet]:
        return {}

    def tasks(self) -> Dict[str, Task]:
        return {
            'task_tc': Task(model='model1', simulation='tc')
        }

    def simulations(self) -> Dict[str, AbstractSim]:
        """
        Simulation time is in [s]
        :return:
        """
        tcsim = TimecourseSim([
            Timecourse(start=0, end=600, steps=2000),
            Timecourse(start=0, end=600, steps=2000, changes={"X": 10, "Y": 20}),
        ])
        return {
            "tc": tcsim,
        }

    def functions(self) -> Dict[str, Function]:
        """ Calculate additional functions.

        :return:
        """
        # processing calculates new outputs given on a single task result
        # These are no aggregation functions
        unit_time = "min"
        unit_data = "dimensionless"

        # FIXME: units
        f1 = Function(
            index="f1",
            formula="(sin(X)+Y+Z)/max(X)",
            variables={
                "X": Data(self, "X", task="task_tc", unit=unit_data),
                "Y": Data(self, "Y", task="task_tc", unit=unit_data),
                "Z": Data(self, "Z", task="task_tc", unit=unit_data),
            }
        )
        f2 = Function(
            index="f2",
            formula="Y/max(Y)",
            variables={
                "Y": Data(self, "Y", task="task_tc", unit=unit_data),
            }
        )
        # [3] arbitrary processing (e.g. pharmacokinetic calculations)
        # Processing(variables) # arbitrary functions
        # Aggregation over
        return {
            "f1": f1,
            "f2": f2
        }

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
                     sid="Fig2", num_cols=1, num_rows=1)
        plots = fig2.create_plots(
            xaxis=Axis("data", unit=unit_data),
            yaxis=Axis("data", unit=unit_data),
            legend=True
        )
        plots[0].curve(
            x=Data(self, "f1", function="f1"),
            y=Data(self, "f2", function="f2"),
            label="f2 ~ f1", color="black", marker="o", alpha=0.3
        )

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




