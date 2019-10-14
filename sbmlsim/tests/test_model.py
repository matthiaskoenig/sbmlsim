"""
Test model module.
"""
from matplotlib import pyplot as plt

import sbmlsim
from sbmlsim import plotting
from sbmlsim.simulation import TimecourseSimulation
from sbmlsim.model import clamp_species
from sbmlsim.parametrization import ChangeSet

from sbmlsim.tests.settings import MODEL_REPRESSILATOR


def test_clamp_sid():
    r = sbmlsim.load_model(MODEL_REPRESSILATOR)
    tsim = TimecourseSimulation(tstart=0, tend=400, steps=400)
    results = sbmlsim.timecourse(r, tsim)

    # Perform clamping
    # TODO:
    clamp_species(r, sid="X", value=20)


    # create figure
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    plotting.add_line(ax=ax1, data=results,
                      xid='time', yid="X", label="X")
    plotting.add_line(ax=ax1, data=results,
                      xid='time', yid="Y", label="Y", color="darkblue")

    ax1.legend()
    plt.show()
