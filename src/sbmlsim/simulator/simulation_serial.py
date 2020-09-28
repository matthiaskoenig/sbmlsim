"""
Serial simulator.
"""
import logging
from typing import List

import pandas as pd

from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.result import XResult
from sbmlsim.simulation import ScanSim, TimecourseSim
from sbmlsim.simulator.simulation import SimulatorAbstract, SimulatorWorker
from sbmlsim.utils import timeit


logger = logging.getLogger(__name__)


class SimulatorSerial(SimulatorAbstract, SimulatorWorker):
    """
    Simulators can run many different models.
    Use the set_model method to set the model to execute.
    """

    def __init__(self, model=None, **kwargs):
        """Serial simulator.

        :param model: Path to model or model
        :param kwargs: integrator settings
        """
        self.integrator_settings = {
            "absolute_tolerance": 1e-14,
            "relative_tolerance": 1e-14,
        }
        self.integrator_settings.update(kwargs)
        self.set_model(model)

    def set_model(self, model):
        """Set model for simulator and updates the integrator settings

        This should handle caching and state saving.
        """
        if model is None:
            self.model = None
        else:
            if isinstance(model, AbstractModel):
                self.model = model
            else:
                # handle path, urn, ...
                self.model = RoadrunnerSBMLModel(
                    source=model, settings=self.integrator_settings
                )

            self.set_integrator_settings(**self.integrator_settings)

    def set_integrator_settings(self, **kwargs):
        if isinstance(self.model, RoadrunnerSBMLModel):
            RoadrunnerSBMLModel.set_integrator_settings(self.model.r, **kwargs)
        else:
            logger.warning(
                "Integrator settings can only be set on RoadrunnerSBMLModel."
            )

    def set_timecourse_selections(self, selections):
        RoadrunnerSBMLModel.set_timecourse_selections(self.r, selections=selections)

    @property
    def r(self):
        return self.model._model

    @property
    def ureg(self):
        return self.model.ureg

    @property
    def udict(self):
        return self.model.udict

    def run_timecourse(self, simulation: TimecourseSim) -> XResult:
        """ Run single timecourse."""
        if not isinstance(simulation, TimecourseSim):
            raise ValueError(
                f"'run_timecourse' requires TimecourseSim, but " f"'{type(simulation)}'"
            )
        scan = ScanSim(simulation=simulation)
        return self.run_scan(scan)

    def run_scan(self, scan: ScanSim) -> XResult:
        """ Run a scan simulation."""
        # normalize the scan (simulation and dimensions)
        scan.normalize(udict=self.udict, ureg=self.ureg)

        # create all possible combinations of the scan
        indices, simulations = scan.to_simulations()

        # simulate (uses respective function of simulator)
        dfs = self._timecourses(simulations)

        # based on the indices the result structure must be created
        return XResult.from_dfs(dfs=dfs, scan=scan, udict=self.udict, ureg=self.ureg)

    def _timecourses(self, simulations: List[TimecourseSim]) -> List[pd.DataFrame]:
        if len(simulations) > 1:
            logger.warning(
                "Use of SimulatorSerial to run multiple timecourses. "
                "Use SimulatorParallel instead."
            )
        return [self._timecourse(sim) for sim in simulations]
