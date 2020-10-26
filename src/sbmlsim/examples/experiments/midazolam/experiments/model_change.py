from typing import Dict, List

from sbmlsim.data import DataSet, load_pkdb_dataframes_by_substance
from sbmlsim.experiment import ExperimentDict
from sbmlsim.fit import FitData, FitMapping
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.task import Task

from ...midazolam import MODEL_PATH
from . import MidazolamSimulationExperiment


class MidazolamModelChangeExperiment(MidazolamSimulationExperiment):
    def models(self) -> Dict[str, AbstractModel]:
        Q_ = self.Q_
        return ExperimentDict(
            {
                "model": MODEL_PATH,
                "model_with_changes": AbstractModel(
                    source=MODEL_PATH,
                    changes={"[Cve_mid]": Q_(10, "nM"), "PODOSE_mid": Q_(0.1, "g")},
                ),
            }
        )

    def simulations(self) -> Dict[str, TimecourseSim]:
        Q_ = self.Q_
        bodyweight = Q_(75, "kg")
        tcsims = {}
        tcsims["sim1"] = TimecourseSim(
            [
                Timecourse(
                    start=0,
                    end=100,
                    steps=600,
                    changes={
                        **self.default_changes(),
                        "Ri_mid": Q_(0.1, "mg/kg") * bodyweight / Q_(15, "min"),
                        "BW": bodyweight,
                    },
                ),
            ]
        )

        return tcsims

    def tasks(self) -> Dict[str, Task]:
        tasks = {}
        for sim_key in self.simulations():
            for model_key in self.models():
                tasks[f"task_{model_key}_{sim_key}"] = Task(
                    model=model_key, simulation=sim_key
                )
        return ExperimentDict(tasks)
