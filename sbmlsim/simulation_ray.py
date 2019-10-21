import ray

import roadrunner
import pandas as pd
import logging
from sbmlsim.timecourse import TimecourseSim, Timecourse
from sbmlsim.result import Result

# start ray
ray.init(ignore_reinit_error=True)


@ray.remote
class SimulatorActor(object):
    """Ray actor to execute simulations.
    An actor instance is specific for a given model.
    An actor is essentially a stateful worker"""
    def __init__(self, path, selections=None):
        self.r = roadrunner.RoadRunner(path)
        if not selections:
            r_model = self.r.model  # type: roadrunner.ExecutableModel

            self.r.timeCourseSelections = ["time"] \
                 + r_model.getFloatingSpeciesIds() \
                 + r_model.getBoundarySpeciesIds() \
                 + r_model.getGlobalParameterIds() \
                 + r_model.getReactionIds() \
                 + r_model.getCompartmentIds()
            self.r.timeCourseSelections += [f'[{key}]' for key in (
                    r_model.getFloatingSpeciesIds() + r_model.getBoundarySpeciesIds())]
        else:
            self.r.timeCourseSelections = selections

    def _timecourses(self, simulations):
        """"""
        results = []
        for tc_sim in simulations:
            results.append(self.timecourse(tc_sim))
        return results

    def timecourse(self, simulation: TimecourseSim) -> pd.DataFrame:
        """ Timecourse simulations based on timecourse_definition.

        :param r: Roadrunner model instance
        :param simulation: Simulation definition(s)
        :param reset_all: Reset model at the beginning
        :return:
        """
        if isinstance(simulation, Timecourse):
            simulation = TimecourseSim(timecourses=[simulation])

        if simulation.reset:
            self.r.resetToOrigin()

        # selections backup
        model_selections = self.r.timeCourseSelections
        if simulation.selections is not None:
            self.r.timeCourseSelections = simulation.selections

        frames = []
        t_offset = 0.0
        for tc in simulation.timecourses:

            # apply changes
            for key, value in tc.changes.items():
                self.r[key] = value

            # run simulation
            s = self.r.simulate(start=tc.start, end=tc.end, steps=tc.steps)
            df = pd.DataFrame(s, columns=s.colnames)
            df.time = df.time + t_offset
            frames.append(df)
            t_offset += tc.end

        # reset selections
        self.r.timeCourseSelections = model_selections

        return pd.concat(frames)


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
        logging.warning(f"creating '{actor_count}' SimulationActors for: '{path}'")
        self.actor_count = actor_count

        # read SBML string once, to avoid IO blocking
        with open(path, "r") as f_sbml:
            sbml_str = f_sbml.read()
        sbml_strings = [sbml_str] * actor_count

        self.simulators = [SimulatorActor.remote(sbml_strings[k], selections) for k in range(actor_count)]

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
