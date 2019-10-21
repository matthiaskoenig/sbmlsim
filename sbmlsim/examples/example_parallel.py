"""
Parallel execution of timecourses
"""

import time
import roadrunner
import ray

from sbmlsim.timecourse import TimecourseSim, Timecourse
from sbmlsim.simulation_ray import SimulatorParallel, SimulatorActor
from sbmlsim.simulation_serial import SimulatorSerial

from sbmlsim.tests.constants import MODEL_REPRESSILATOR, MODEL_GLCWB


def example_single_actor():
    """ Creates a single stateful simulator actor and executes timecourse.

    Normally multiple actors are created which execute the simulation
    load together. Actors should not be created manually, use the Simulator
    classes for simulations.

    :return:
    """
    # Create single actor process
    sa = SimulatorActor.remote(MODEL_REPRESSILATOR)

    # run simulation
    tcsim = TimecourseSim([
        Timecourse(start=0, end=100, steps=100),
        Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 20}),
    ])
    tc_id = sa.timecourse.remote(tcsim)
    print("-" * 80)
    print(ray.get(tc_id))
    print("-" * 80)


def example_multiple_actors():
    """Multiple independent simulator actors.

    Actors should not be created manually, use the Simulator
    classes for simulations.
    """
    # create ten Simulators.
    simulators = [SimulatorActor.remote(MODEL_REPRESSILATOR) for _ in range(16)]

    # define timecourse
    tcsim = TimecourseSim([
        Timecourse(start=0, end=100, steps=100),
        Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 20}),
    ])

    # run simulation on simulators
    tc_ids = [s.timecourse.remote(tcsim) for s in simulators]
    # collect results
    results = ray.get(tc_ids)
    return results


def example_parallel_timecourse():
    """Execute multiple simulations with model in parallel."""
    tcsims = []

    tc_sim = TimecourseSim([
        Timecourse(start=0, end=100, steps=100,
                   changes={
                       'IVDOSE_som': 0.0,  # [mg]
                       'PODOSE_som': 0.0,  # [mg]
                       'Ri_som': 10.0E-6,  # [mg/min]
                   })
    ])

    # collect all simulation definitions (see also the ensemble functions)
    nsim = 500
    for _ in range(nsim):
        tcsims.append(tc_sim)

    def message(info, time):
        print(f"{info:<20}: {time:4.3f}")

    # load model once for caching (fair comparison)
    r = roadrunner.RoadRunner(MODEL_GLCWB)

    print("-" * 80)
    print(f"Run '{nsim}' simulations")
    print("-" * 80)

    simulator_defs = [
        {
            "key": "parallel",
            'simulator': SimulatorParallel,
            'kwargs': {'actor_count': 12}
        },
        {
            "key": "serial",
            'simulator': SimulatorSerial,
            'kwargs': {}
        },
    ]

    for info in simulator_defs:

        key = info['key']
        Simulator = info['simulator']
        kwargs = info['kwargs']

        # run simulation (with model reading)
        start_time = time.time()
        # create a simulator with 16 parallel actors
        simulator = Simulator(path=MODEL_GLCWB, **kwargs)
        load_time = time.time()-start_time
        message(f"{key} loading", load_time)

        start_time = time.time()
        results = simulator.timecourses(simulations=tcsims)
        sim_time = time.time()-start_time
        message(f"{key} total", load_time + sim_time)
        message(f"{key} simulation", sim_time)

        # run parallel simulation (without model reading)
        start_time = time.time()
        results = simulator.timecourses(simulations=tcsims)
        message(f"{key} repeat", time.time()-start_time)

        print("-" * 80)


if __name__ == "__main__":
    # example_single_actor()
    # example_multiple_actors()
    example_parallel_timecourse()
