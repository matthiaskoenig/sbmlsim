"""A hybrid problem which is defined in python, without PEtab.

The experiment simulates the model of Lotka and Volterra and compares the
species with data. It is a module of its own, because the workers of a
parallel fit import the experiment class of a problem.
"""

from pathlib import Path
from typing import ClassVar

import pandas as pd

from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitData, FitMapping, FitMappingCollection
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Simulation
from sbmlsim.task import Task
from tests.sciml.hybrid import MODEL_PATH

#: the species of the model at the times `1` to `10`, simulated with the
#: parameters of the model
DATA: dict[str, list[float]] = {
    "prey": [
        0.1996,
        0.4843,
        1.6064,
        5.4941,
        3.0782,
        0.1952,
        0.2965,
        0.9041,
        3.1091,
        8.8516,
    ],
    "predator": [
        0.9224,
        0.1947,
        0.0676,
        0.1419,
        6.6987,
        1.9956,
        0.3923,
        0.0994,
        0.0683,
        1.2238,
    ],
}

#: the simulations of the experiment, which are the conditions of the inputs
SIMULATIONS = ("e1", "e2")


class LotkaVolterra(SimulationExperiment):
    """The model of Lotka and Volterra, simulated twice and compared with data.

    Attributes:
        model_path: the model the experiment simulates, which a test replaces
            by the model with the networks.
    """

    model_path: ClassVar[Path] = MODEL_PATH

    def models(self) -> dict[str, AbstractModel | Path]:
        return {
            "lv": AbstractModel(
                source=self.model_path,
                language_type=AbstractModel.LanguageType.SBML,
            )
        }

    def datasets(self) -> dict[str, DataSet]:
        return {
            sid: DataSet.from_df(
                pd.DataFrame(
                    {
                        "time": [float(k) for k in range(1, 11)],
                        "time_unit": "second",
                        "value": values,
                        "value_unit": "dimensionless",
                    }
                ),
                ureg=self.ureg,
            )
            for sid, values in DATA.items()
        }

    def simulations(self) -> dict[str, Simulation]:
        return {sid: Simulation(start=0.0, end=10.0, steps=100) for sid in SIMULATIONS}

    def tasks(self) -> dict[str, Task]:
        return {f"task_{sid}": Task(model="lv", simulation=sid) for sid in SIMULATIONS}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            f"{species}_{sid}": FitMapping(
                self,
                reference=FitData(self, dataset=species, xid="time", yid="value"),
                observable=FitData(self, task=f"task_{sid}", xid="time", yid=species),
            )
            for sid in SIMULATIONS
            for species in DATA
        }


def collections(
    experiment: type[SimulationExperiment] = LotkaVolterra,
) -> list[FitMappingCollection]:
    """Get the fit mappings of the experiment, one collection per simulation.

    Args:
        experiment: the experiment class.

    Returns:
        The collections.
    """
    return [
        FitMappingCollection(
            experiment=experiment,
            sid=sid,
            mappings=[f"{species}_{sid}" for species in DATA],
        )
        for sid in SIMULATIONS
    ]
