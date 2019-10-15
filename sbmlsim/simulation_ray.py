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


if __name__ == "__main__":

    # Create actor process
    sa = SimulatorActor.remote("repressilator.xml")

    # run simulation
    tc_sim = TimecourseSimulation([
        Timecourse(start=0, end=100, steps=100),
        Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 20}),
    ])

    # run sin
    tc_id = sa.timecourse.remote(tc_sim)
    print("-" * 80)
    print(ray.get(tc_id))
    print("-" * 80)
    exit()


    # Create ten Simulators.
    simulators = [SimulatorActor.remote("repressilator.xml") for _ in range(16)]

    # Run simulation on every simulator
    sim_ids = [s.timecourse.remote(tc_sim) for s in simulators]
    results = ray.get([s.get_value.remote() for s in simulators])
    print(results)

