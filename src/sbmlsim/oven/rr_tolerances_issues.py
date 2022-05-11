import pandas as pd

from sbmlsim.model.model_roadrunner import roadrunner


tiny_sbml = "tiny_example_1.xml"
r = roadrunner.RoadRunner(tiny_sbml)
s = r.simulate(0, 100, steps=100)
df = pd.DataFrame(s, columns=s.colnames)
r.plot()

r = roadrunner.RoadRunner(tiny_sbml)
r.integrator.setValue("relative_tolerance", 1e-18)
r.integrator.setValue("absolute_tolerance", 1e-18)
s = r.simulate(0, 100, steps=100)
df = pd.DataFrame(s, columns=s.colnames)
r.plot()
