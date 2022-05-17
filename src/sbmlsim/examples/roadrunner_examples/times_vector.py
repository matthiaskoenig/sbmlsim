"""Simulate roadrunner with a times vector."""

from rich import print
from sbmlutils.resources import REPRESSILATOR_SBML

from sbmlsim.simulator.model_rr import roadrunner
from sbmlsim.utils import timeit


r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(REPRESSILATOR_SBML))
s1 = r.simulate(times=[0, 10.98, 50.12])
s2 = r.simulate(start=0, end=10, steps=50)

print(s1)
print("-" * 80)
print(s2)
print("-" * 80)

r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(REPRESSILATOR_SBML))
s1 = r.simulate(start=0, end=10, steps=50)
s2 = r.simulate(times=[0, 10.98, 50.12])

print(s1)
print("-" * 80)
print(s2)
print("-" * 80)


@timeit
def simulate_times(r: roadrunner.RoadRunner):
    r.resetToOrigin()
    r.simulate(times=[0, 10.98, 50.12])


@timeit
def simulate_steps(r):
    r.resetToOrigin()
    r.simulate(times=[0, 10.98, 50.12, 100.0])
