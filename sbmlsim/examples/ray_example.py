import ray

import libsbml
import roadrunner
import numpy as np
import pandas as pd
from typing import Union

from sbmlsim.simulation import TimecourseSimulation, Timecourse

# use string object to avoid parallel file io
model_path = "repressilator.xml"
doc = libsbml.readSBMLFromFile(model_path)  # type: libsbml.SBMLDocument
sbml_str = libsbml.writeSBMLToString(doc)

# get simulators (? can this be parallelized?)

# simulators = [roadrunner.RoadRunner(sbml_str) for k in range(cpus)]
# from datetime import datetime

# start ray
ray.init()

@ray.remote
def create_matrix(size):
    return np.random.normal(size=size)

@ray.remote
def multiply_matrices(x, y):
    return np.dot(x, y)

x_id = create_matrix.remote([1000, 1000])
y_id = create_matrix.remote([1000, 1000])
z_id = multiply_matrices.remote(x_id, y_id)

# Get the results.
z = ray.get(z_id)


print("-" * 80)
@ray.remote
class SimulatorActor(object):
    """An actor is essentially a stateful worker"""
    def __init__(self, model_path):
        self.r = roadrunner.RoadRunner(model_path)  # type: roadrunner.RoadRunner
        self.s = None

    # def load_model(self, model_path):
    #    self.r = roadrunner.RoadRunner(model_path)  # type: roadrunner.RoadRunner

    def run_simulation(self):
        s = self.r.simulate(start=0, end=100, steps=1001)
        self.s = pd.DataFrame(s, columns=s.colnames)

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

        self.s = pd.concat(frames)


    def get_value(self):
        return self.s


tc_sim = TimecourseSimulation([
        Timecourse(start=0, end=100, steps=100),
        Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 20}),
    ])


# Create an actor process.
sa = SimulatorActor.remote(model_path="repressilator.xml")
sa.run_simulation.remote()

sa.timecourse.remote(tc_sim)


# Check the actor's counter value.
print(ray.get(sa.get_value.remote()))
print("-" * 80)


# Create ten Simulators.
simulators = [SimulatorActor.remote(model_path="repressilator.xml") for _ in range(16)]


results = ray.get([s.run_simulation.remote() for s in simulators])
print(results)  # prints [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

results = ray.get([s.timecourse.remote(tc_sim) for s in simulators])
print(results)  # prints [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]



