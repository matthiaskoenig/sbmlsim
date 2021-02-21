from typing import Dict, Tuple
from collections import namedtuple

from sbmlsim.experiment import SimulationExperiment, ExperimentDict
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import TimecourseSim
from sbmlsim.simulation.sensititvity import ModelSensitivity
from sbmlsim.task import Task

from ...midazolam import MODEL_PATH


MolecularWeights = namedtuple("MolecularWeights", "mid mid1oh")


def exclude_parameters_midazolam(pid: str) -> bool:
    """Filter for excluding parameter ids in sensitivity."""
    if pid.startswith("Mr_"):
        return True
    if pid.startswith("conversion_"):
        return True
    if pid.startswith("F_"):
        return True
    if pid.startswith("BP_"):
        return True

    return False


class MidazolamSimulationExperiment(SimulationExperiment):
    """Base class for all GlucoseSimulationExperiments. """

    def models(self) -> Dict[str, AbstractModel]:
        Q_ = self.Q_
        models = {
            "model": AbstractModel(
                source=MODEL_PATH,
                language_type=AbstractModel.LanguageType.SBML,
                changes={
                    "KI__MID1OHEX_Vmax": Q_(14.259652024532818, "mmole/min"),
                    "KI__MID1OHEX_Km": Q_(0.7051197538875393, "mM"),
                    "ftissue_mid1oh": Q_(99.23248555491428, "liter/min"),
                    "fup_mid1oh": Q_(0.19507488419734886, "dimensionless"),
                },
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
        else:
            return {}

    def simulations(self, simulations=None) -> Dict[str, TimecourseSim]:
        if simulations is None:
            return simulations

        # injecting additional scan dimension for timecourse simulation
        for sim_key, sim in simulations.copy().items():
            if isinstance(sim, TimecourseSim):
                scan = ModelSensitivity.difference_sensitivity_scan(
                    model=self._models["model"],
                    simulation=sim,
                    difference=0.5,
                    exclude_filter=exclude_parameters_midazolam,
                )
                simulations[f"{sim_key}_sensitivity"] = scan

        # print("Simulation keys:", simulations.keys())
        return simulations

    @property
    def Mr(self):
        return MolecularWeights(
            mid=self.Q_(325.768, "g/mole"),
            mid1oh=self.Q_(341.8, "g/mole"),
        )
