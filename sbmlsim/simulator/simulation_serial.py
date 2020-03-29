"""
Serial simulator.
"""
import logging
from typing import List
import pandas as pd

from sbmlsim.simulator.simulation import SimulatorAbstract, SimulatorWorker, set_integrator_settings
from sbmlsim.result import Result
from sbmlsim.simulation import TimecourseSim, ScanSim
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel

logger = logging.getLogger(__name__)


class SimulatorSerial(SimulatorAbstract, SimulatorWorker):
    def __init__(self, model: AbstractModel, **kwargs):
        """ Serial simulator.

        :param model: Path to model
        :param selections: Selections to set
        :param kwargs: integrator arguments
        """
        if isinstance(model, AbstractModel):
            self.r = model._model
            self.udict = model.udict
            self.ureg = model.ureg
        else:
            # handle path, urn, ...
            m = RoadrunnerSBMLModel(source=model)
            self.r = m._model
            self.udict = m.udict
            self.ureg = m.ureg

        set_integrator_settings(self.r, **kwargs)

    def _run_timecourses(self, simulations: List[TimecourseSim]) -> List[pd.DataFrame]:
        """ Run many timecourses."""
        if isinstance(simulations, TimecourseSim):
            simulations = [simulations]

        if len(simulations) > 1:
            logger.warning("Use of SimulatorSerial to run multiple timecourses. "
                           "Use SimulatorParallel instead.")
        return [self.timecourse(sim) for sim in simulations]

    def run_scan(self, scan: ScanSim) -> Result:
        """ Run a scan simulation."""
        # Create all possible combinations of the scan
        indices, simulations = scan.to_simulations()

        # Based on the indices the result structure must be created
        return Result.from_dfs(
            dfs=self._run_timecourses(simulations),
            scan=scan,
            udict=self.udict
        )