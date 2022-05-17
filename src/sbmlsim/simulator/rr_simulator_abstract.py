"""Classes for running simulations with SBML models."""
from typing import List, Iterator

import pandas as pd
from sbmlutils import log

from sbmlsim.simulation import TimecourseSim, ScanSim
from sbmlsim.result import XResult

# TODO: handle unit information
# from pint import Quantity
# from sbmlsim.units import UnitsInformation


logger = log.get_logger(__name__)


# FIXME: This can probably be all on the roadrunner model.

from abc import ABC, abstractmethod


# FIXME: default integrator settings
# # default settings
# self.integrator_settings = {
#     "absolute_tolerance": 1e-10,
#     "relative_tolerance": 1e-10,
# }
# self.integrator_settings.update(kwargs)

# set_model
# set_timecourse_selections
# set_integrator_settings


class SimulatorAbstractRR(ABC):
    """Abstract base class for roadrunner simulator."""
    @abstractmethod
    def set_model(self, model_state: str) -> None:
        """Set model from state."""
        pass

    @abstractmethod
    def set_timecourse_selections(self, selections: Iterator[str]) -> None:
        """Set timecourse selections."""
        pass

    @abstractmethod
    def set_integrator_settings(self, **kwargs) -> None:
        """Set integrator settings."""
        pass

    @abstractmethod
    def _run_timecourses(self, simulations: List[TimecourseSim]) -> List[pd.DataFrame]:
        """Execute timecourse simulations."""
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
        """Run scan simulation."""
        # normalize the scan (simulation and dimensions)
        # FIXME: units
        scan.normalize(uinfo=self.uinfo)

        # create all possible combinations of the scan
        indices, simulations = scan.to_simulations()

        # simulate (uses respective function of simulator)
        dfs = self._timecourses(simulations)

        # based on the indices the result structure must be created
        # FIXME: units
        # return XResult.from_dfs(dfs=dfs, scan=scan, uinfo=self.uinfo)

        return XResult.from_dfs(dfs=dfs, scan=scan)
