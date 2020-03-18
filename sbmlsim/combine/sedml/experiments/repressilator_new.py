from typing import Dict, List, Iterable
from pathlib import Path
import pandas as pd

from sbmlsim.tests.constants import MODEL_REPRESSILATOR

from sbmlsim.experiment import SimulationExperiment
from sbmlsim.data import Data, DataSet
from sbmlsim.timecourse import Timecourse, TimecourseSim
from sbmlsim.plotting import Figure, Axis

from sbmlsim.experiment import ExperimentResult
from sbmlsim.models import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.tasks import Task


class RepressilatorExperiment(SimulationExperiment):

    def models(self) -> Dict[str, AbstractModel]:
        return {
            'model1': RoadrunnerSBMLModel(source=MODEL_REPRESSILATOR,
                                          ureg=self.ureg)
        }

    def datasets(self) -> Dict[str, DataSet]:
        return {}

    def tasks(self) -> Dict[str, Task]:
        Q_ = self.Q_
        tcsim = TimecourseSim([
            Timecourse(start=0, end=100, steps=100),
            Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 20}),
        ])
        return {
            'task1': Task(model='model1', simulation=tcsim)
        }

    def figures(self) -> Dict[str, Figure]:
        unit_time = "s"
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
            x=Data(self, "time", simulation="task1", unit=unit_time),
            y=Data(self, "X", simulation="task1", unit=unit_data),
            label="X sim", color="black"
        )
        plots[0].curve(
            x=Data(self, "time", simulation="task1", unit=unit_time),
            y=Data(self, "Y", simulation="task1", unit=unit_data),
            label="Y sim", color="blue"
        )
        return {fig.sid: fig}


if __name__ == "__main__":

    base_path = Path(__file__).parent
    data_path = base_path
    output_path = Path(".")

    exp = RepressilatorExperiment(
        data_path=data_path,
        base_path=base_path
    )
    results = exp.run(
        output_path=output_path / RepressilatorExperiment.__name__,
        show_figures=True
    )
    print(results)
