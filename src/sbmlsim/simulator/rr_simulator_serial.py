"""Serial simulator.

Executing simulations with single roadrunner instance on a single core.
"""
from typing import List, Iterator, Optional

import pandas as pd
from sbmlutils import log


from sbmlsim.simulation import TimecourseSim
from sbmlsim.simulator.rr_simulator_abstract import SimulatorAbstractRR
from sbmlsim.simulator.rr_worker import SimulationWorkerRR


logger = log.get_logger(__name__)


class SimulatorSerialRR(SimulatorAbstractRR):
    """Serial simulator using a single core."""

    def __init__(self, model_state: Optional[str] = None, actor_count: int = 1, **kwargs):
        """Initialize serial simulator with single worker."""
        self.simulator = SimulationWorkerRR()
        if actor_count != 1:
            raise ValueError("Only a single actor allowed.")
        self.actor_count: int = actor_count

        if model_state:
            self.set_model(model_state)

        # TODO: same for ray simulator


    def set_model(self, model_state: str) -> None:
        """Set model from state."""
        self.simulator.set_model(model_state)

    def set_timecourse_selections(self, selections: Iterator[str]):
        """Set timecourse selections."""
        self.simulator.set_timecourse_selections(selections=selections)

    def set_integrator_settings(self, **kwargs):
        """Set integrator settings."""
        self.simulator.set_integrator_settings(**kwargs)

    def _run_timecourses(self, simulations: List[TimecourseSim]) -> List[pd.DataFrame]:
        """Execute timecourse simulations."""
        if len(simulations) > 1:
            logger.warning(
                "Use of SimulatorSerial to run multiple timecourses. "
                "Use SimulatorParallel instead."
            )
        return [self._timecourse(sim) for sim in simulations]
