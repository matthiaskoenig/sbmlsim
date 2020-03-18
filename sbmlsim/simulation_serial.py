"""
Serial simulator.
"""
import logging
from typing import List

from sbmlsim.simulation import SimulatorAbstract, SimulatorWorker, set_integrator_settings
from sbmlsim.result import Result
from sbmlsim.timecourse import TimecourseSim
from sbmlsim.models import AbstractModel

logger = logging.getLogger(__name__)


class SimulatorSerial(SimulatorAbstract, SimulatorWorker):
    def __init__(self, model: AbstractModel, **kwargs):
        """ Serial simulator.

        :param model: Path to model
        :param selections: Selections to set
        :param kwargs: integrator arguments
        """
        self.r = model._model
        self.udict = model.udict
        self.ureg = model.ureg
        set_integrator_settings(self.r, **kwargs)

    def timecourses(self, simulations: List[TimecourseSim]) -> Result:
        """ Run many timecourses."""
        if isinstance(simulations, TimecourseSim):
            simulations = [simulations]

        if len(simulations) > 1:
            logger.warning("Use of SimulatorSerial to run multiple timecourses. "
                           "Use SimulatorParallel instead.")
        dfs = [self.timecourse(sim) for sim in simulations]
        return Result(dfs, self.udict, self.ureg)
