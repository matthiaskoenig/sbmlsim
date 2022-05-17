"""Parallel simulation using ray."""
import tempfile
from typing import Iterator, List

import numpy as np
import pandas as pd
import psutil
import ray
from sbmlutils import log


from sbmlsim.simulation import TimecourseSim
from sbmlsim.simulator.simulation_worker_rr import SimulatorRR, SimulationWorkerRR

logger = log.get_logger(__name__)
ray.init(ignore_reinit_error=True)


@ray.remote
class SimulatorActor(SimulationWorkerRR):
    """Ray actor to execute simulations.

    An actor is essentially a stateful worker
    """

    def work(self, simulations):
        """Run a bunch of simulations on a single worker."""
        results = []
        for tc_sim in simulations:
            results.append(self._timecourse(tc_sim))
        return results


class SimulatorRayRR(SimulatorRR):
    """Parallel simulator using multiple cores.


    """

    def __init__(self, model=None, **kwargs):
        """Initialize parallel simulator with multiple workers.

        :param model: model source or model
        :param actor_count: int,
        """
        if "actor_count" in kwargs:
            self.actor_count = kwargs.pop("actor_count")
        else:

            self.actor_count = max(self.cpu_count() - 1, 1)
        logger.info(f"Using '{self.actor_count}' cpu/core for parallel simulation.")

        logger.debug(f"Creating '{self.actor_count}' SimulationActors")
        self.simulators = [SimulatorActor.remote() for _ in range(self.actor_count)]

        super(SimulatorParallel, self).__init__(model=None, **kwargs)
        if model is not None:
            self.set_model(model=model)

    def set_model(self, model) -> None:
        """Set model."""
        for simulator in self.simulators:
            simulator.set_model(model)

    def set_timecourse_selections(self, selections: Iterator[str]):
        """Set timecourse selections."""
        for simulator in self.simulators:
            simulator.set_timecourse_selections.remote(selections)

    def set_integrator_settings(self, **kwargs):
        """Set integrator settings."""
        for simulator in self.simulators:
            self.simulator.set_integrator_settings(**kwargs)

    def _timecourses(self, simulations: List[TimecourseSim]) -> List[pd.DataFrame]:
        """Run all simulations with given model and collect the results.

        :param simulations: List[TimecourseSim]
        :return: Result
        """
        # Strip units for parallel simulations
        for sim in simulations:
            sim.strip_units()

        # Split simulations in chunks for actors
        # !simulation have to stay in same order to reconstruct dimensions!
        chunk_indices = np.array_split(np.arange(len(simulations)), self.actor_count)
        chunks = [[] for _ in range(self.actor_count)]
        for k, indices in enumerate(chunk_indices):
            for index in indices:
                chunks[k].append(simulations[index])

        tc_ids = []
        for k, simulator in enumerate(self.simulators):
            tcs_id = simulator.work.remote(chunks[k])
            tc_ids.append(tcs_id)

        results = ray.get(tc_ids)
        # flatten list of lists [[df, df], [df, df], ...]
        # indices = [k for sublist in chunks_indices for k in sublist]
        return [df for sublist in results for df in sublist]

    @staticmethod
    def _create_chunks(item, size: int):
        """Yield successive sized chunks from item."""
        for i in range(0, len(item), size):
            yield item[i : i + size]

    @staticmethod
    def cpu_count() -> int:
        """Get physical CPU count."""
        return psutil.cpu_count(logical=False)
