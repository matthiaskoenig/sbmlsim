import ray
ray.init()

import libsbml
import roadrunner
import pandas as pd


# use string object to avoid parallel file io
model_path = "repressilator.xml"
doc = libsbml.readSBMLFromFile(model_path)  # type: libsbml.SBMLDocument
sbml_str = libsbml.writeSBMLToString(doc)

# get simulators (? can this be parallelized?)
cpus = 10
simulators = [roadrunner.RoadRunner(sbml_str) for k in range(cpus)]
from datetime import datetime

#Python 3:
startTime = datetime.now()
r = roadrunner.RoadRunner(model_path)
print(datetime.now() - startTime)

startTime = datetime.now()
r = roadrunner.RoadRunner(model_path)
print(datetime.now() - startTime)



@ray.remote
def simulate(r, k):
    # loading

    r = roadrunner.RoadRunner(model_path)
    # simulating
    s = r.simulate(start=0, end=100, steps=100)
    df = pd.DataFrame(s, columns=s.colnames)
    return k

results = [simulate.remote(simulators[i], i) for i in range(4)]
print(ray.get(results))
