import pytest

from sbmlsim.timecourse import Timecourse, TimecourseSim
from sbmlsim.simulation_ray import SimulatorParallel as Simulator

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
