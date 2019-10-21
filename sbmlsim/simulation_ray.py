import ray

import roadrunner
import pandas as pd
import libsbml
import logging
from sbmlsim.timecourse import TimecourseSim, Timecourse
from sbmlsim.result import Result
from typing import List

from sbmlsim.model import set_timecourse_selections
from sbmlsim.simulation import SimulatorAbstract, SimulatorWorker

# start ray
ray.init(ignore_reinit_error=True)


@ray.remote
class SimulatorActor(SimulatorWorker):
    """Ray actor to execute simulations.
    An actor instance is specific for a given model.
    An actor is essentially a stateful worker"""
    def __init__(self, path, selections: List[str] = None):
        self.r = roadrunner.RoadRunner(path)
        set_timecourse_selections(self.r, selections)

    def timecourses(self, simulations: list) -> list:
        """"""
        results = []
        for tc_sim in simulations:
            results.append(self.timecourse(tc_sim))
        return results


class SimulatorParallel(SimulatorAbstract):
    """
    Parallel simulator
    """
    def __init__(self, path, selections: List[str] = None, actor_count: int = 15):
        """ Initialize parallel simulator with multiple workers.

        :param path:
        :param selections: selections to set, if None full selection is performed
        :param actor_count:
        """
        logging.warning(f"creating '{actor_count}' SimulationActors for: '{path}'")
        self.actor_count = actor_count

        # read SBML string once, to avoid IO blocking
        with open(path, "r") as f_sbml:
            sbml_str = f_sbml.read()

        self.simulators = [SimulatorActor.remote(sbml_str, selections) for _ in range(actor_count)]

    def timecourses(self, simulations: List[TimecourseSim]) -> Result:
        """ Run all simulations with given model and collect the results.

        :param path:
        :param simulations:
        :param selections:
        :return:
        """
        if isinstance(simulations, TimecourseSim):
            simulations = [simulations]

        # Split simulations in chunks for actors
        chunks = [[] for _ in range(self.actor_count)]
        for k, tc_sim in enumerate(simulations):
            chunks[k % self.actor_count].append(tc_sim)

        tc_ids = []
        for k, simulator in enumerate(self.simulators):
            tcs_id = simulator.timecourses.remote(chunks[k])
            tc_ids.append(tcs_id)

        results = ray.get(tc_ids)
        # flatten list of lists [[df, df], [df, df], ...]
        dfs = [df for sublist in results for df in sublist]
        return Result(dfs)
        # return results

    @staticmethod
    def create_chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
