"""The simulation experiments of the toy model and their groups."""

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, override

import pandas as pd

from sbmlsim import Q
from sbmlsim.data import Data, DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitData, FitMapping
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Curve, Figure, Plot, SubPlot
from sbmlsim.simulation import Simulation
from sbmlsim.task import Task
from tests.project_toy import PROJECT


class ToyExperiment(SimulationExperiment):
    """The base of the toy experiments: the decay of A and its figure."""

    #: the variant of the model the experiment simulates
    variant: ClassVar[str | None] = None
    #: the rate of the decay [1/min], `None` for the rate of the model
    rate: ClassVar[float | None] = None

    @override
    def models(self) -> dict[str, AbstractModel | Path]:
        return {"model": AbstractModel(PROJECT.model_path(self.variant))}

    @override
    def simulations(self) -> dict[str, Simulation]:
        changes = None if self.rate is None else {"k": Q(self.rate, "1/min")}
        return {"decay": Simulation(end=10, steps=50, preinit_changes=changes)}

    @override
    def tasks(self) -> dict[str, Task]:
        return {"task_decay": Task(model="model", simulation="decay")}

    @override
    def figures(self) -> dict[str, Figure]:
        plot = Plot(
            sid="decay", xaxis=Axis("time", unit="min"), yaxis=Axis("A", unit="mM")
        )
        plot.curves.append(
            Curve(
                x=Data(index="time", task="task_decay"),
                y=Data(index="[A]", task="task_decay"),
                name="A",
            )
        )
        return {
            "fig_decay": Figure(
                experiment=self,
                sid="fig_decay",
                num_rows=1,
                num_cols=1,
                subplots=[SubPlot(plot=plot, row=1, col=1)],
            )
        }


class ToyDecay(ToyExperiment):
    """The decay at the rate of the model, with data of A."""

    @override
    def datasets(self) -> dict[str, DataSet]:
        df = pd.DataFrame(
            {
                "time": [0.0, 2.0, 4.0, 6.0, 8.0],
                "time_unit": "min",
                "A": [1.0, 0.67, 0.45, 0.30, 0.20],
                "A_unit": "mM",
            }
        )
        return {"data_decay": DataSet.from_df(df)}

    @override
    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm_decay_A": FitMapping(
                self,
                reference=FitData(self, dataset="data_decay", xid="time", yid="A"),
                observable=FitData(self, task="task_decay", xid="time", yid="[A]"),
            )
        }


class ToyDecaySlow(ToyExperiment):
    """The decay of the slow variant of the model."""

    variant = "slow"


class ToyDecayHalf(ToyExperiment):
    """The decay at half of the rate of the model."""

    rate = 0.1


EXPERIMENTS: Mapping[str, list[type[SimulationExperiment]]] = {
    "studies": [ToyDecay],
    "variants": [ToyDecaySlow, ToyDecayHalf],
}
"""The groups of the experiments, `all` is every experiment of the groups."""
