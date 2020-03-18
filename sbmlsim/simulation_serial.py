"""
Serial simulator.
"""
import logging
from typing import List
from pathlib import Path

from sbmlsim.models import RoadrunnerSBMLModel
from sbmlsim.simulation import SimulatorAbstract, SimulatorWorker, set_integrator_settings
from sbmlsim.result import Result
from sbmlsim.timecourse import TimecourseSim
from sbmlsim.units import Units

logger = logging.getLogger(__name__)


class SimulatorSerial(SimulatorAbstract, SimulatorWorker):
    def __init__(self, path: Path, selections: List[str] = None, **kwargs):
        """

        :param path: Path to model
        :param selections: Selections to set
        :param kwargs: integrator arguments
        """
        if path:
            # FIXME: store the abstract model class
            model = RoadrunnerSBMLModel(source=path, selections=selections)
            self.r = model._model
            set_integrator_settings(self.r, **kwargs)
            # TODO: use global ureg
            self.udict, self.ureg = Units.get_units_from_sbml(model_path=path)
        else:
            self.r = None
            self.udict = None
            self.ureg = None
            logger.warning("Simulator without model instance created!")

    def timecourses(self, simulations: List[TimecourseSim]) -> Result:
        """ Run many timecourses."""
        if isinstance(simulations, TimecourseSim):
            simulations = [simulations]

        if len(simulations) > 1:
            logger.warning("Use of SimulatorSerial to run multiple timecourses. "
                           "Use SimulatorParallel instead.")
        dfs = [self.timecourse(sim) for sim in simulations]
        return Result(dfs, self.udict, self.ureg)
