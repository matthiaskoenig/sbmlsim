import logging
import ray
import psutil
import roadrunner
from typing import List
import pandas as pd
import tempfile

from sbmlsim.model import RoadrunnerSBMLModel, AbstractModel
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulation import TimecourseSim
from sbmlsim.result import XResult
from sbmlsim.simulator.simulation import SimulatorWorker


logger = logging.getLogger(__name__)

# start ray
ray.init(ignore_reinit_error=True)


def cpu_count() -> int:
    """Get physical CPU count."""
    return psutil.cpu_count(logical=False)


@ray.remote
class SimulatorActor(SimulatorWorker):
    """Ray actor to execute simulations.

    An actor is essentially a stateful worker
    """
    def __init__(self, path_state):
        """State contains model, integrator settings and selections."""
        self.r = roadrunner.RoadRunner()  # type: roadrunner.RoadRunner
        self.r.loadState(path_state)

    def work(self, simulations):
        """Run a bunch of simulations on a single worker."""
        results = []
        for tc_sim in simulations:
            results.append(self._timecourse(tc_sim))
        return results


class SimulatorParallel(SimulatorSerial):
    """
    Parallel simulator
    """
    def __init__(self, model: AbstractModel, **kwargs):
        """ Initialize parallel simulator with multiple workers.

        :param path:
        :param selections: List[str],  selections to set, if None full selection is performed
        :param actor_count: int,
        """
        super(SimulatorParallel, self).__init__(model, **kwargs)
        self.actor_count = kwargs.get("actor_count", cpu_count()-1)
        logger.warning(f"Creating '{self.actor_count}' SimulationActors")

        f_tmp = tempfile.NamedTemporaryFile(suffix=".dat")
        self.r.saveState(f_tmp.name)

        self.simulators = [SimulatorActor.remote(f_tmp.name) for _ in range(self.actor_count)]

    def _timecourses(self, simulations: List[TimecourseSim]) -> List[pd.DataFrame]:
        """ Run all simulations with given model and collect the results.

        :param simulations: List[TimecourseSim]
        :return: Result
        """

        # Split simulations in chunks for actors
        chunks = [[] for _ in range(self.actor_count)]
        for k, tc_sim in enumerate(simulations):
            chunks[k % self.actor_count].append(tc_sim)

        tc_ids = []
        for k, simulator in enumerate(self.simulators):
            tcs_id = simulator.work.remote(chunks[k])
            tc_ids.append(tcs_id)

        results = ray.get(tc_ids)
        # flatten list of lists [[df, df], [df, df], ...]
        dfs = [df for sublist in results for df in sublist]
        return XResult(dfs)

    @staticmethod
    def _create_chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
