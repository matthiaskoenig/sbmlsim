import ray

import roadrunner
import numpy as np
import pandas as pd

from sbmlsim.simulation import TimecourseSimulation, Timecourse

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


    def timecourse(self, sim: TimecourseSimulation) -> pd.DataFrame:
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
    @staticmethod
    def timecourses(path, simulations, selections=None, actor_count=16):
        """ Run all simulations with given model and collect the results.

        :param path:
        :param simulations:
        :param selections:
        :return:
        """
        # Create simulators
        simulators = [SimulatorActor.remote(path, selections) for _ in range(actor_count)]


        # Split simulations in chunks for actors
        tc_ids = []
        chunks = list(Simulator.create_chunks(simulations, actor_count))
        chunks_count = len(chunks)
        for chunk in chunks:
            print(len(chunk))


        for k, simulator in enumerate(simulators):
            # FIXME: chunking is not working
            if k<(chunks_count-1):
                chunk = chunks[k]
                tc_id = simulator.timecourses.remote(chunk)
                tc_ids.append(tc_id)

        results = ray.get(tc_ids)
        # flatten list of lists [[df, df], [df, df], ...]
        return [df for sublist in results for df in sublist]
        # return results

    @staticmethod
    def create_chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

if __name__ == "__main__":

    if False:
        # [1] Create single actor process
        sa = SimulatorActor.remote("repressilator.xml")

        # run simulation
        tc_sim = TimecourseSimulation([
            Timecourse(start=0, end=100, steps=100),
            Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 20}),
        ])
        tc_id = sa.timecourse.remote(tc_sim)
        print("-" * 80)
        print(ray.get(tc_id))
        print("-" * 80)

    if False:
        # [2] Create ten Simulators.
        simulators = [SimulatorActor.remote("repressilator.xml") for _ in range(16)]
        # Run simulation on every simulator
        tc_ids = [s.timecourse.remote(tc_sim) for s in simulators]
        results = ray.get(tc_ids)
        assert results

    # [3] execute multiple simulations
    simulations = []
    tc_sim = TimecourseSimulation([
        Timecourse(start=0, end=100, steps=100),
        Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 20}),
    ])
    for _ in range(1000):
        simulations.append(tc_sim)

    results = Simulator.timecourses(path="repressilator.xml", simulations=simulations)
    print(len(results))


