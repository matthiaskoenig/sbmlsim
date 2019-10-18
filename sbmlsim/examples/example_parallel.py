"""
Parallel execution of timecourses
"""
from sbmlsim.simulation_ray import SimulatorActor
import ray
from sbmlsim.tests.constants import MODEL_REPRESSILATOR

from sbmlsim.timecourse import TimecourseSim, Timecourse


def example_single_actor():
    """ Creates a single stateful simulator actor and executes timecourse.
    Normally multiple actors are created which execute the simulation
    load together.

    :return:
    """
    # Create single actor process
    sa = SimulatorActor.remote()

    # run simulation
    tc_sim = TimecourseSim([
        Timecourse(start=0, end=100, steps=100),
        Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 20}),
    ])
    tc_id = sa.timecourse.remote(tc_sim)
    print("-" * 80)
    print(ray.get(tc_id))
    print("-" * 80)

def example_multiple_actors():
    # [2] Create ten Simulators.
    simulators = [SimulatorActor.remote("repressilator.xml") for _ in range(16)]
    # Run simulation on every simulator
    tc_ids = [s.timecourse.remote(tc_sim) for s in simulators]
    results = ray.get(tc_ids)
    assert results

def exam

# [3] execute multiple simulations
simulations = []
tc_sim_rep = TimecourseSimulation([
    Timecourse(start=0, end=100, steps=100),
    Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 20}),
])

tc_sim = TimecourseSimulation([
    Timecourse(start=0, end=100, steps=100,
               changes={
                   'IVDOSE_som': 0.0,  # [mg]
                   'PODOSE_som': 0.0,  # [mg]
                   'Ri_som': 10.0E-6,  # [mg/min]
               }),
])

for _ in range(1000):
    simulations.append(tc_sim)

import time

# model_path = "repressilator.xml"
model_path = "body19_livertoy_flat.xml"

simulator = Simulator(path=model_path, actor_count=16)

start_time = time.time()
results = simulator.timecourses(simulations=simulations)
print("--- %s seconds ---" % (time.time() - start_time))
print(len(results))

from sbmlsim.simulation import timecourses as timcourses_serial
from sbmlsim import load_model

r = load_model(model_path)

start_time = time.time()
timcourses_serial(r, simulations)
print("--- %s seconds ---" % (time.time() - start_time))

