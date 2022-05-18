"""Serial simulator.

Executing simulations with single roadrunner instance on a single core.
"""
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd
from sbmlutils import log

from sbmlsim.model.rr_model import roadrunner
from sbmlsim.simulation import TimecourseSim
from sbmlsim.simulator.rr_simulator_abstract import SimulatorAbstractRR
from sbmlsim.simulator.rr_worker import SimulationWorkerRR


logger = log.get_logger(__name__)


class SimulatorSerialRR(SimulatorAbstractRR):
    """Serial simulator using a single core."""

    @staticmethod
    def from_sbml(sbml_path: Path) -> "SimulatorSerialRR":
        """Set model from SBML."""
        rr: roadrunner.RoadRunner = roadrunner.RoadRunner(str(sbml_path))
        simulator = SimulatorSerialRR()
        # FIXME: implement global model cache
        simulator.set_model(rr.saveStateS())
        return simulator

    def __init__(self):
        """Initialize serial simulator with single worker."""
        self.worker = SimulationWorkerRR()

    def set_model(self, model_state: str) -> None:
        """Set model from state."""
        self.worker.set_model(model_state)

    def set_timecourse_selections(
        self, selections: Optional[Iterator[str]] = None
    ) -> None:
        """Set timecourse selections."""
        self.worker.set_timecourse_selections(selections=selections)

    def set_integrator_settings(self, **kwargs):
        """Set integrator settings."""
        self.worker.set_integrator_settings(**kwargs)

    def _run_timecourses(self, simulations: List[TimecourseSim]) -> List[pd.DataFrame]:
        """Execute timecourse simulations."""
        if len(simulations) > 1:
            logger.warning(
                "Use of SimulatorSerial to run multiple timecourses. "
                "Use SimulatorParallel instead."
            )
        return [self.worker._timecourse(sim) for sim in simulations]
