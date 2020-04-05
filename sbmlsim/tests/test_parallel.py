import pytest

from sbmlsim.simulation.timecourse import Timecourse, TimecourseSim
from sbmlsim.simulator.simulation_ray import SimulatorParallel as Simulator
from sbmlsim.examples import example_parallel

from sbmlsim.tests.constants import MODEL_GLCWB


@pytest.mark.skip
def test_simulation_parallel():
    tcsims = []

    tc_sim = TimecourseSim([
        Timecourse(start=0, end=100, steps=100,
                   changes={
                       'IVDOSE_som': 0.0,  # [mg]
                       'PODOSE_som': 0.0,  # [mg]
                       'Ri_som': 10.0E-6,  # [mg/min]
                   })
    ])

    # collect multiple simulation definitions (see also the ensemble functions)
    nsim = 100
    for _ in range(nsim):
        tcsims.append(tc_sim)

    simulator = Simulator(path=MODEL_GLCWB, actor_count=15)
    results = simulator.timecourses(simulations=tcsims)

@pytest.mark.skip
def test_parallel_1():
    example_parallel.example_single_actor()


@pytest.mark.skip
def test_parallel_2():
    example_parallel.example_multiple_actors()


@pytest.mark.skip
def test_parallel_3():
    example_parallel.example_parallel_timecourse(nsim=20, actor_count=5)
