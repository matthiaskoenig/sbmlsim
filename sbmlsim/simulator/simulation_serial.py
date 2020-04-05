"""
Serial simulator.
"""
import logging
from typing import List
import pandas as pd

from sbmlsim.simulator.simulation import SimulatorAbstract, SimulatorWorker, set_integrator_settings
from sbmlsim.result import XResult
from sbmlsim.simulation import TimecourseSim, ScanSim
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.utils import timeit

logger = logging.getLogger(__name__)


class SimulatorSerial(SimulatorAbstract, SimulatorWorker):
    def __init__(self, model: AbstractModel, **kwargs):
        """ Serial simulator.

        :param model: Path to model
        :param selections: Selections to set
        :param kwargs: integrator arguments
        """
        if isinstance(model, AbstractModel):
            self.model = model
        else:
            # handle path, urn, ...
            self.model = RoadrunnerSBMLModel(source=model, **kwargs)

    @property
    def r(self):
        return self.model._model

    @property
    def ureg(self):
        return self.model.ureg

    @property
    def udict(self):
        return self.model.udict

    @timeit
    def run_timecourse(self, simulation: TimecourseSim) -> XResult:
        """ Run single timecourse."""
        if not isinstance(simulation, TimecourseSim):
            raise ValueError(f"'run_timecourse' requires TimecourseSim, but "
                             f"'{type(simulation)}'")
        scan = ScanSim(simulation=simulation)
        return self.run_scan(scan)

    @timeit
    def run_scan(self, scan: ScanSim) -> XResult:
        """ Run a scan simulation."""
        # normalize the scan
        scan.normalize(udict=self.udict, ureg=self.ureg)

        # Create all possible combinations of the scan
        indices, simulations = scan.to_simulations()

        if len(simulations) > 1:
            logger.warning("Use of SimulatorSerial to run multiple timecourses. "
                           "Use SimulatorParallel instead.")
        dfs = self._timecourses(simulations)

        # Based on the indices the result structure must be created
        return XResult.from_dfs(
            dfs=dfs,
            scan=scan,
            udict=self.udict,
            ureg=self.ureg
        )

    def _timecourses(self, simulations: List[TimecourseSim]) -> List[pd.DataFrame]:
        return [self._timecourse(sim) for sim in simulations]