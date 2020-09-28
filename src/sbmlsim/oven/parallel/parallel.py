"""
Example parallel simulation

requirements
    roadrunner
    ray
    python-libsbml-experimental
    pandas
"""
import time

import libsbml
import pandas as pd
import ray
import roadrunner
from pint import UnitRegistry


# start ray
ray.init(ignore_reinit_error=True)


@ray.remote
class SimulatorActorState(object):
    """Ray actor to execute simulations."""

    def __init__(self, state_path, ureg):
        print("Start loading model")
        ts = time.time()
        self.r = roadrunner.RoadRunner()  # type: roadrunner.RoadRunner
        self.r.loadState(state_path)
        self.ureg = ureg
        te = time.time()
        print("Finished loading model: {:2.2f} ms".format((te - ts) * 1000))

    def simulate(self, size=1):
        print("Start simulations")
        ts = time.time()
        results = []
        for k in range(size):
            self.r.resetAll()
            s = self.r.simulate(0, 100, steps=100)
            # create numpy array (which can be pickled), better used shared memory
            df = pd.DataFrame(s, columns=s.colnames)
            results.append(df)
        te = time.time()
        print("Finished '{}' simulations: {:2.2f} ms".format(size, (te - ts) * 1000))
        return results


if __name__ == "__main__":

    actor_count = 10  # cores to run this on

    # sending the SBML to every core, every core needs to read the SBML
    # sending as string to avoid parallel IO by 30 cores
    path = "glucose_rbcmqreduced_flat.xml"
    # path = "glucose_rbcmqparasite_flat.xml"
    doc = libsbml.readSBMLFromFile(path)  # type: libsbml.SBMLDocument
    sbml_str = libsbml.writeSBMLToString(doc)

    r = roadrunner.RoadRunner(path)
    r.saveState("test.dat")
    del r
    ureg = UnitRegistry()

    def run_simulations(simulators):
        # run simulations
        sim_per_actor = 10
        tc_ids = []
        for k, simulator in enumerate(simulators):
            tcs_id = simulator.simulate.remote(size=sim_per_actor)
            tc_ids.append(tcs_id)

        results = ray.get(tc_ids)

    # Run parallel simulations
    simulators = [
        SimulatorActorState.remote("test.dat", ureg) for _ in range(actor_count)
    ]
    run_simulations(simulators)
