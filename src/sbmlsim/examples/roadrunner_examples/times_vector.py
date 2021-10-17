"""Simulate roadrunner with a times vector."""

from rich import print
import roadrunner
from sbmlsim.utils import timeit

from sbmlutils.test import REPRESSILATOR_SBML

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


def simulate_times():
    r.resetToOrigin()
    r.simulate(times=[0, 10.98, 50.12])

