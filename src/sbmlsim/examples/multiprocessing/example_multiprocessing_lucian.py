import ray
import roadrunner

# start ray
ray.init(ignore_reinit_error=True)

modelstr = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model metaid="__main" id="__main">
    <listOfParameters>
      <parameter id="a" value="3" constant="true"/>
    </listOfParameters>
  </model>
</sbml>
"""

@ray.remote
class SimulatorActorPath(object):
    """Ray actor to execute simulations."""

    def __init__(self):
        self.r = roadrunner.RoadRunner(modelstr)

    def simulate(self):
        """Simulate."""
        print("Just read the value of 'a'")
        print(self.r.getValue("a"), "\n")

def ray_example():
    """Ray example."""
    actor_count: int = 1  # cores to run this on

    simulators = [SimulatorActorPath.remote() for _ in range(actor_count)]

    # run simulations
    # tc_ids = []
    for simulator in simulators:
        simulator.simulate.remote()


if __name__ == "__main__":
    ray_example()
