"""Two experiments of the decay of A with the mappings of every kind of evidence."""

from pathlib import Path
from typing import override

import pandas as pd

from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitData, FitMapping
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Simulation
from sbmlsim.task import Task
from tests.credibility_toy import MODEL_PATH
from tests.credibility_toy.metadata import Health, Route, ToyMetaData

TIMES = [0.0, 2.0, 4.0, 6.0, 8.0]
DECAY = [1.0, 0.67, 0.45, 0.30, 0.20]


def _data(values: list[float]) -> DataSet:
    """Get the data of A over the times, the means of four subjects."""
    df = pd.DataFrame(
        {
            "time": TIMES,
            "time_unit": "min",
            "A": values,
            "A_unit": "mM",
            "count": 4,
        }
    )
    return DataSet.from_df(df)


class CredExperiment(SimulationExperiment):
    """The base of the experiments: the decay, the data and the mappings."""

    @override
    def datasets(self) -> dict[str, DataSet]:
        return {
            "data_decay": _data(DECAY),
            "data_limit": _data([3 * v for v in DECAY]),
            "data_zero": _data([0.0] * len(DECAY)),
        }

    @override
    def models(self) -> dict[str, AbstractModel | Path]:
        return {"model": AbstractModel(MODEL_PATH)}

    @override
    def simulations(self) -> dict[str, Simulation]:
        return {"decay": Simulation(end=10, steps=50)}

    @override
    def tasks(self) -> dict[str, Task]:
        return {"task_decay": Task(model="model", simulation="decay")}

    def mapping(
        self, dataset: str, route: Route, health: Health = Health.HEALTHY
    ) -> FitMapping:
        """Get the mapping of a dataset of A to the simulation."""
        return FitMapping(
            self,
            reference=FitData(
                self, dataset=dataset, xid="time", yid="A", count="count"
            ),
            observable=FitData(self, task="task_decay", xid="time", yid="[A]"),
            metadata=ToyMetaData(route=route, health=health),
        )


class CredA(CredExperiment):
    """The calibration, validation and outlier mappings."""

    @override
    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm_train": self.mapping("data_decay", Route.IV),
            "fm_dose": self.mapping("data_decay", Route.IV),
            "fm_renal": self.mapping("data_decay", Route.IV, Health.RENAL_IMPAIRMENT),
            "fm_limit": self.mapping("data_limit", Route.IV),
            "fm_subject": self.mapping("data_decay", Route.IV),
            "fm_outlier": self.mapping("data_decay", Route.IV),
        }


class CredB(CredExperiment):
    """A mapping with the key of the study A and a mapping of zeros."""

    @override
    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm_dose": self.mapping("data_decay", Route.IV),
            "fm_zero": self.mapping("data_zero", Route.PO),
        }


EXPERIMENTS: dict[str, list[type[SimulationExperiment]]] = {"studies": [CredA, CredB]}
