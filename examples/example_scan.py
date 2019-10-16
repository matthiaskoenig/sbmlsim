"""
Example shows basic model simulations and plotting.
"""
import os
import numpy as np
from matplotlib import pyplot as plt

from sbmlsim.model import load_model
from sbmlsim.simulation import timecourses, Timecourse, TimecourseSimulation
from sbmlsim.results import TimecourseResult
from sbmlsim.parametrization import ChangeSet
from sbmlsim.plotting_matlab import add_line
from sbmlsim.tests.settings import MODEL_REPRESSILATOR


def parameter_scan(parallel=False):
    """Perform a parameter scan"""

    # [2] value scan
    scan_changeset = ChangeSet.scan_changeset('n', values=np.linspace(start=2, stop=10, num=8))
    tc_sims = TimecourseSimulation(
        Timecourse(start=0, end=100, steps=100)
    ).ensemble(changeset=scan_changeset)

    for tc_sim in tc_sims:
        print(tc_sim)

    if not parallel:
        r = load_model(MODEL_REPRESSILATOR)
        results = timecourses(r, tc_sims)
    else:
        # pass
        # from sbmlsim.simulation_ray import

        # TODO: implement
        results = None

    return results


if __name__ == "__main__":
    parameter_scan(parallel=False)
    parameter_scan(parallel=True)

