import ray

import roadrunner
import pandas as pd

from sbmlsim.timecourse import TimecourseSim, Timecourse

# start ray
ray.init()


@ray.remote
class SimulatorActor(object):
    """Ray actor to execute simulations.
    An actor instance is specific for a given model.
    An actor is essentially a stateful worker"""
    def __init__(self, path, selections: bool = True):
        self.r = roadrunner.RoadRunner(path)
        if selections:
            self._set_timecourse_selections()

    def _set_timecourse_selections(self, selections=None) -> None:
        """ Sets the full model selections. """
        if selections:
            self.r.timeCourseSelections = selections
        else:
            r_model = self.r.model  # type: roadrunner.ExecutableModel

            self.r.timeCourseSelections = ["time"] \
                     + r_model.getFloatingSpeciesIds() \
                     + r_model.getBoundarySpeciesIds() \
                     + r_model.getGlobalParameterIds() \
                     + r_model.getReactionIds() \
                     + r_model.getCompartmentIds()
            self.r.timeCourseSelections += [f'[{key}]' for key in (
                    r_model.getFloatingSpeciesIds() + r_model.getBoundarySpeciesIds())]

    def timecourses(self, simulations: list) -> list:
        """"""
        results = []
        for tc_sim in simulations:
            results.append(self.timecourse(tc_sim))
        return results


    def timecourse(self, sim: TimecourseSim) -> pd.DataFrame:
        """ Timecourse simulations based on timecourse_definition.

        :param sim: Simulation definition(s)
        :return:
        """
        if sim.reset:
            self.r.resetToOrigin()

        # selections backup
        model_selections = self.r.timeCourseSelections
        if sim.selections is not None:
            self.r.timeCourseSelections = sim.selections

        frames = []
        t_offset = 0.0
        for tc in sim.timecourses:

            # apply changes
            for key, value in tc.changes.items():
                self.r[key] = value

            # FIXME: model changes

            # run simulation
            s = self.r.simulate(start=tc.start, end=tc.end, steps=tc.steps)
            df = pd.DataFrame(s, columns=s.colnames)
            df.time = df.time + t_offset
            frames.append(df)
            t_offset += tc.end

        # reset selections
        self.r.timeCourseSelections = model_selections

        # self.s = pd.concat(frames)
        return pd.concat(frames)


class Simulator(object):
    """
    # TODO: cash the actors
    """
    def __init__(self, path, selections=None, actor_count=16):
        self.actor_count = actor_count
        self.simulators = [SimulatorActor.remote(path, selections) for _ in range(actor_count)]

    def timecourses(self, simulations):
        """ Run all simulations with given model and collect the results.

        :param path:
        :param simulations:
        :param selections:
        :return:
        """
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
        return [df for sublist in results for df in sublist]
        # return results

    @staticmethod
    def create_chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
