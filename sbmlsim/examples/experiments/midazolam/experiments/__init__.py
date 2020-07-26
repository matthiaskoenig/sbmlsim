from typing import Dict, Tuple
from collections import namedtuple

from sbmlsim.experiment import SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.task import Task

from ...midazolam import MODEL_PATH


MolecularWeights = namedtuple('MolecularWeights', 'mid mid1oh')


class MidazolamSimulationExperiment(SimulationExperiment):
    """Base class for all GlucoseSimulationExperiments. """
    def models(self) -> Dict[str, AbstractModel]:
        return {
            "model": MODEL_PATH
        }

    def tasks(self) -> Dict[str, Task]:
        if self.simulations():
            return {
                f"task_{key}": Task(model="model", simulation=key) for key in self.simulations()
            }

    @property
    def Mr(self):
        return MolecularWeights(
            mid=self.Q_(325.768, 'g/mole'),
            mid1oh=self.Q_(341.8, 'g/mole'),
        )

    def default_changes(self: SimulationExperiment) -> Dict:
        """Default changes to simulations."""
        Q_ = self.Q_

        changes = {
            'KI__MID1OHEX_Vmax': Q_(14.259652024532818, 'mmole/min'),
            'KI__MID1OHEX_Km': Q_(0.7051197538875393, 'mM'),
            'ftissue_mid1oh': Q_(99.23248555491428, 'liter/min'),
            'fup_mid1oh': Q_(0.19507488419734886, 'dimensionless'),
        }
        return changes

