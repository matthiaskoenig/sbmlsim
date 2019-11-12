"""
Serial simulator.
"""
import logging
from typing import List

from sbmlsim.simulation import SimulatorAbstract, SimulatorWorker, set_integrator_settings
from sbmlsim.model import load_model
from sbmlsim.result import Result
from sbmlsim.timecourse import TimecourseSim
from sbmlsim.units import Units

logger = logging.getLogger(__name__)


class SimulatorSerial(SimulatorAbstract, SimulatorWorker):
    def __init__(self, path, selections: List[str] = None, **kwargs):
        """

        :param path: Path to model
        :param selections: Selections to set
        :param kwargs: integrator arguments
        """
        if path:
            self.r = load_model(path=path, selections=selections)
            set_integrator_settings(self.r, **kwargs)
            self.units = Units.get_units_from_sbml(model_path=path)

        else:
            self.r = None
            self.units = None
            logger.warning("Simulator without model instance created!")

    def timecourses(self, simulations: List[TimecourseSim]) -> Result:
        """ Run many timecourses."""
        if isinstance(simulations, TimecourseSim):
            simulations = [simulations]

        if len(simulations) > 1:
            logger.warning("Use of SimulatorSerial to run multiple timecourses. "
                           "Use SimulatorParallel instead.")
        dfs = [self.timecourse(sim) for sim in simulations]
        return Result(dfs, self.units)
