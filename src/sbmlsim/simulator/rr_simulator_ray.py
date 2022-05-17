"""Parallel simulation using ray."""
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
import psutil
import ray
from sbmlutils import log


from sbmlsim.simulation import TimecourseSim
from sbmlsim.simulator.rr_simulator_abstract import SimulatorAbstractRR
from sbmlsim.simulator.rr_worker import SimulationWorkerRR

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


class SimulatorRayRR(SimulatorAbstractRR):
    """Parallel simulator using multiple cores via ray."""

    def __init__(self, actor_count: Optional[int] = None):
        """Initialize parallel simulator with multiple workers.

        :param actor_count: int, number of actors (cores)
        """
        max_count = max(self.cpu_count() - 1, 1)
        if not actor_count:
            actor_count = max_count
        if actor_count > max_count:
            logger.warning(
                f"Actor count > maximal count '{actor_count} > {max_count}'"
            )
        self.actor_count: int = actor_count

        logger.info(f"Using '{actor_count}' cores for parallel simulation.")
        self.workers = [SimulatorActor.remote() for _ in range(self.actor_count)]

    def set_model(self, model_state: str) -> None:
        """Set model from state."""
        for worker in self.workers:
            worker.set_model.remote(model_state)

    def set_timecourse_selections(self, selections: Optional[Iterator[str]] = None) -> None:
        """Set timecourse selections."""
        for worker in self.workers:
            worker.set_timecourse_selections.remote(selections)

    def set_integrator_settings(self, **kwargs):
        """Set integrator settings."""
        for worker in self.workers:
            worker.set_integrator_settings.remote(**kwargs)

    def _run_timecourses(self, simulations: List[TimecourseSim]) -> List[pd.DataFrame]:
        """Execute timecourse simulations."""
        # Strip units for parallel simulations (this requires normalization of units!)
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
        for k, simulator in enumerate(self.workers):
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
