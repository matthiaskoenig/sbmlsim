"""
Example simulation experiment.
"""

from typing import Dict
from pathlib import Path

from sbmlsim.tests.constants import MODEL_REPRESSILATOR

from sbmlsim.experiment import SimulationExperiment
from sbmlsim.models import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.data import Data, DataSet
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
            Timecourse(start=0, end=600, steps=1000),
            Timecourse(start=0, end=600, steps=1000, changes={"X": 10, "Y": 20}),
        ])
        return {
            "tc": tcsim,
        }

    def processing(self):
        """ Calculate additional functions.

        :return:
        """
        # processing calculates new outputs given on a single task result
        # These are no aggregation functions
        unit_time = "min"
        unit_data = "dimensionless"

        # [1] direct access of variables (in task or dataset)
        task_tc_time = Data(self, "time", task="task_tc", unit=unit_time),
        task_tc_X = Data(self, "X", task="task_tc", unit=unit_time),
        task_tc_X = Data(self, "Y", task="task_tc", unit=unit_time),

        # [2] functional relationships expressable as MathML
        Function(variables) = X/Y
        Function = X/(X+Y+Z)
        # or aggregation functions
        Function = X/max(X)

        # [3] arbitrary processing (e.g. pharmacokinetic calculations)
        Processing(variables) # arbitrary functions

        # Aggregation over 



    def figures(self) -> Dict[str, Figure]:
        unit_time = "min"
        unit_data = "dimensionless"

        fig = Figure(experiment=self,
                     sid="Fig1", num_cols=1, num_rows=1)
        plots = fig.create_plots(
            xaxis=Axis("time", unit=unit_time),
            yaxis=Axis("data", unit=unit_data),
            legend=True
        )
        plots[0].set_title(f"{self.sid}_{fig.sid}")
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
        return {fig.sid: fig}


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




