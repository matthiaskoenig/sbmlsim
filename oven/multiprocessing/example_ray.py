"""Example parallel simulation with ray.

requirements
    roadrunner==2.2.1
    ray
    pandas
"""
import time
from pathlib import Path

import pandas as pd
import ray
import roadrunner
from roadrunner import Config


Config.setValue(Config.LLVM_BACKEND, Config.LLJIT)


# start ray
ray.init(ignore_reinit_error=True)


@ray.remote
class SimulatorActorPath(object):
    """Ray actor to execute simulations."""

    def __init__(self, r: roadrunner.RoadRunner):
        self.r: roadrunner.RoadRunner = r

    def simulate(self, size=1):
        """Simulate."""
        print("Start simulations")
        ts = time.time()
        results = []
        for _ in range(size):
            self.r.resetAll()
            s = self.r.simulate(0, 100, steps=100)
            # create numpy array (which can be pickled), better use shared memory
            df = pd.DataFrame(s, columns=s.colnames)
            results.append(df)
        te = time.time()
        print("Finished '{}' simulations: {:2.2f} ms".format(size, (te - ts) * 1000))
        return results


def ray_example():
    """Ray example."""
    actor_count: int = 3  # cores to run this on

    model_path = Path(__file__).parent / "icg_body_flat.xml"
    rr: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))

    simulators = [SimulatorActorPath.remote(rr) for _ in range(actor_count)]

    # run simulations
    sim_per_actor = 10
    tc_ids = []
    for simulator in simulators:
        tcs_id = simulator.simulate.remote(size=sim_per_actor)
        tc_ids.append(tcs_id)

    results = ray.get(tc_ids)
    return results


if __name__ == "__main__":
    ray_example()
