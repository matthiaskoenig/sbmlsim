"""Serial simulator."""
from typing import List, Iterator

import pandas as pd
from pint import Quantity
from sbmlutils import log

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulator.model_rr import roadrunner
from sbmlsim.result import XResult
from sbmlsim.simulation import ScanSim, TimecourseSim
from sbmlsim.simulator.simulation_worker_rr import SimulationWorkerRR
from sbmlsim.units import UnitsInformation


logger = log.get_logger(__name__)

from abc import ABC, abstractmethod

# FIXME:


class SimulatorRR(ABC):
    @abstractmethod
    def set_model(self, model):
        """Set model."""
        pass

    @abstractmethod
    def set_timecourse_selections(self, selections: Iterator[str]):
        """Set timecourse selections."""
        pass

    def set_integrator_settings(self, **kwargs):
        """Set integrator settings."""
        pass

    @abstractmethod
    def _timecourses(self, simulations: List[TimecourseSim]) -> List[pd.DataFrame]:
        """Run timecourses."""
        pass

    def run_timecourse(self, simulation: TimecourseSim) -> XResult:
        """Run single timecourse."""
        if not isinstance(simulation, TimecourseSim):
            raise ValueError(
                f"'run_timecourse' requires TimecourseSim, but " f"'{type(simulation)}'"
            )
        scan = ScanSim(simulation=simulation)
        return self.run_scan(scan)

    def run_scan(self, scan: ScanSim) -> XResult:
        """Run a scan simulation."""
        # normalize the scan (simulation and dimensions)
        scan.normalize(uinfo=self.uinfo)

        # create all possible combinations of the scan
        indices, simulations = scan.to_simulations()

        # simulate (uses respective function of simulator)
        dfs = self._timecourses(simulations)

        # based on the indices the result structure must be created
        return XResult.from_dfs(dfs=dfs, scan=scan, uinfo=self.uinfo)


class SimulatorSerialRR(SimulatorRR):
    """Serial simulator using a single core.

    A single simulator can run many different models.
    See the parallel simulator to run simulations on multiple
    cores.
    """

    def __init__(self):
        """Initialize serial simulator with single worker."""

        # Create simulator once
        self.simulator = SimulationWorkerRR()

    def _timecourses(self, simulations: List[TimecourseSim]) -> List[pd.DataFrame]:
        if len(simulations) > 1:
            logger.warning(
                "Use of SimulatorSerial to run multiple timecourses. "
                "Use SimulatorParallel instead."
            )
        return [self._timecourse(sim) for sim in simulations]
