import ray

import roadrunner
import pandas as pd
import logging
from sbmlsim.model import load_model
from sbmlsim.timecourse import TimecourseSim, Timecourse
from sbmlsim.result import Result
from sbmlsim.simulation import SimulatorWorker, set_integrator_settings
from sbmlsim.units import Units


logger = logging.getLogger(__name__)

# start ray
ray.init(ignore_reinit_error=True)


@ray.remote
class SimulatorActor(SimulatorWorker):
    """Ray actor to execute simulations.
    An actor instance is specific for a given model.
    An actor is essentially a stateful worker

    # FIXME: currently no setting of integrator settings

    """
    def __init__(self, path, selections=None):
        self.r = load_model(path, selections)
        self.units = Units.get_units_from_sbml(model_path=path)
        # set_integrator_settings(self.r, **kwargs)

    def _timecourses(self, simulations):
        """Run a bunch of simulations on a single worker."""
        results = []
        for tc_sim in simulations:
            results.append(self.timecourse(tc_sim))
        return results


class SimulatorParallel(object):
    """
    Parallel simulator
    """
    def __init__(self, path, selections=None, actor_count=15):
        """ Initialize parallel simulator with multiple workers.

        :param path:
        :param selections: List[str],  selections to set, if None full selection is performed
        :param actor_count: int,
        """
        logger.warning(f"creating '{actor_count}' SimulationActors for: '{path}'")
        self.actor_count = actor_count

        # read SBML string once, to avoid IO blocking
        with open(path, "r") as f_sbml:
            sbml_str = f_sbml.read()

        self.simulators = [SimulatorActor.remote(sbml_str, selections) for _ in range(actor_count)]

    def timecourses(self, simulations):
        """ Run all simulations with given model and collect the results.

        :param simulations: List[TimecourseSim]
        :return: Result
        """
        if isinstance(simulations, TimecourseSim):
            simulations = [simulations]

        # Split simulations in chunks for actors
        chunks = [[] for _ in range(self.actor_count)]
        for k, tc_sim in enumerate(simulations):
            chunks[k % self.actor_count].append(tc_sim)

        tc_ids = []
        for k, simulator in enumerate(self.simulators):
            tcs_id = simulator._timecourses.remote(chunks[k])
            tc_ids.append(tcs_id)

        results = ray.get(tc_ids)
        # flatten list of lists [[df, df], [df, df], ...]
        dfs = [df for sublist in results for df in sublist]
        return Result(dfs)
        # return results

    @staticmethod
    def _create_chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
