"""Serial simulator."""
from typing import List, Iterator

import pandas as pd
from sbmlutils import log


from sbmlsim.simulation import TimecourseSim
from sbmlsim.simulator.simulation_worker_rr import SimulatorRR, SimulationWorkerRR


logger = log.get_logger(__name__)


class SimulatorSerialRR(SimulatorRR):
    """Serial simulator using a single core.

    A single simulator can run many different models.
    See the parallel simulator to run simulations on multiple
    cores.
    """

    def __init__(self):
        """Initialize serial simulator with single worker."""
        self.simulator = SimulationWorkerRR()

    def set_model(self, model) -> None:
        """Set model."""
        self.simulator.set_model(model)

    def set_timecourse_selections(self, selections: Iterator[str]):
        """Set timecourse selections."""
        self.simulator.set_timecourse_selections(selections=selections)

    def set_integrator_settings(self, **kwargs):
        """Set integrator settings."""
        self.simulator.set_integrator_settings(**kwargs)

    def _timecourses(self, simulations: List[TimecourseSim]) -> List[pd.DataFrame]:
        if len(simulations) > 1:
            logger.warning(
                "Use of SimulatorSerial to run multiple timecourses. "
                "Use SimulatorParallel instead."
            )
        return [self._timecourse(sim) for sim in simulations]
