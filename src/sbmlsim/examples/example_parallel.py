"""
Parallel execution of timecourses
"""
import tempfile
import time

import numpy as np
import roadrunner

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Dimension, ScanSim, Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulator.simulation_ray import (
    SimulatorActor,
    SimulatorParallel,
    cpu_count,
    ray,
)
from sbmlsim.test import MODEL_GLCWB, MODEL_REPRESSILATOR
from sbmlsim.units import Units


def example_single_actor():
    """Creates a single stateful simulator actor and executes timecourse.

    Actors should never created manually !
    Use the SimulatorParallel to run parallel simulations.

    :return:
    """
    # create state file
    r = roadrunner.RoadRunner(str(MODEL_REPRESSILATOR))
    RoadrunnerSBMLModel.set_timecourse_selections(r)
    udict, ureg = Units.get_units_from_sbml(str(MODEL_REPRESSILATOR))

    f_state = tempfile.NamedTemporaryFile(suffix=".dat")
    r.saveState(f_state.name)

    # Create single actor process
    sa = SimulatorActor.remote(f_state.name)

    # run simulation
    tcsim = TimecourseSim(
        [
            Timecourse(start=0, end=100, steps=100),
            Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 20}),
        ]
    )

    tcsim.normalize(udict=udict, ureg=ureg)
    tc_id = sa._timecourse.remote(tcsim)
    results = ray.get(tc_id)
    return results


def example_multiple_actors():
    """Multiple independent simulator actors.

    Actors should never be created manually, use the SimulatorParallel!
    """
    # create state file
    r = roadrunner.RoadRunner(str(MODEL_REPRESSILATOR))
    RoadrunnerSBMLModel.set_timecourse_selections(r)
    udict, ureg = Units.get_units_from_sbml(str(MODEL_REPRESSILATOR))

    f_state = tempfile.NamedTemporaryFile(suffix=".dat")
    r.saveState(f_state.name)

    # create ten Simulators.
    simulators = [SimulatorActor.remote(f_state.name) for _ in range(16)]

    # define timecourse
    tcsim = TimecourseSim(
        [
            Timecourse(start=0, end=100, steps=100),
            Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 20}),
        ]
    )
    tcsim.normalize(udict=udict, ureg=ureg)

    # run simulation on simulators
    tc_ids = [s._timecourse.remote(tcsim) for s in simulators]
    # collect results
    results = ray.get(tc_ids)
    return results


def example_parallel_timecourse(nsim=40, actor_count=15):
    """Execute multiple simulations with model in parallel.

    :param nsim: number of simulations
    :return:
    """
    # collect all simulation definitions (see also the ensemble functions)

    scan = ScanSim(
        simulation=TimecourseSim(
            [
                Timecourse(
                    start=0,
                    end=100,
                    steps=100,
                    changes={
                        "IVDOSE_som": 0.0,  # [mg]
                        "PODOSE_som": 0.0,  # [mg]
                        "Ri_som": 10.0e-6,  # [mg/min]
                    },
                )
            ]
        ),
        dimensions=[Dimension("dim1", index=np.arange(nsim))],
    )

    def message(info, time):
        print(f"{info:<10}: {time:4.3f}")

    # load model once for caching (fair comparison)
    r = roadrunner.RoadRunner(str(MODEL_GLCWB))
    # simulator definitions
    simulator_defs = [
        {
            "key": "parallel",
            "simulator": SimulatorParallel,
            "kwargs": {"actor_count": actor_count},
        },
        {"key": "serial", "simulator": SimulatorSerial, "kwargs": {}},
    ]
    print("-" * 80)
    print(f"Run '{nsim}' simulations")
    print("-" * 80)
    sim_info = []
    for sim_def in simulator_defs:
        key = sim_def["key"]
        print("***", key, "***")
        Simulator = sim_def["simulator"]
        kwargs = sim_def["kwargs"]

        # run simulation (with model reading)
        start_time = time.time()
        # create a simulator with 16 parallel actors
        simulator = Simulator(model=MODEL_GLCWB, **kwargs)
        load_time = time.time() - start_time
        message(f"load", load_time)

        start_time = time.time()
        results = simulator.run_scan(scan=scan)
        sim_time = time.time() - start_time
        total_time = load_time + sim_time
        message("simulate", sim_time)
        message("total", total_time)
        assert len(results.coords["dim1"]) == nsim

        # run parallel simulation (without model reading)
        start_time = time.time()
        results = simulator.run_scan(scan=scan)
        repeat_time = time.time() - start_time
        message(f"repeat", repeat_time)
        assert len(results.coords["dim1"]) == nsim

        actor_count = kwargs.get("actor_count", 1)
        times = {
            "load": load_time,
            "simulate": sim_time,
            "total": total_time,
            "repeat": repeat_time,
        }
        sim_info.extend(
            [
                {
                    "key": key,
                    "nsim": nsim,
                    "actor_count": actor_count,
                    "time_type": k,
                    "time": v,
                }
                for (k, v) in times.items()
            ]
        )

        print("-" * 80)

    return sim_info


if __name__ == "__main__":
    # example_single_actor()
    # example_multiple_actors()

    sim_info = example_parallel_timecourse(nsim=100, actor_count=15)
    ray.timeline(filename="timeline.json")
