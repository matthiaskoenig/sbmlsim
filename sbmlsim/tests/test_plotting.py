"""
Testing plotting functionality.
"""
from matplotlib import pyplot as plt

import sbmlsim
from sbmlsim.simulation import TimecourseSimulation
from sbmlsim.plotting import add_line
from sbmlsim.parametrization import ChangeSet

from sbmlsim.tests.settings import MODEL_REPRESSILATOR


def test_plotting():
    r = sbmlsim.load_model(MODEL_REPRESSILATOR)
    changeset = ChangeSet.parameter_sensitivity_changeset(r, sensitivity=0.5)
    tsim = TimecourseSimulation(tstart=0, tend=400, steps=400, changeset=changeset)
    results = sbmlsim.timecourse(r, tsim)

    # create figure
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    add_line(ax=ax1, data=results,
             xid='time', yid="X", label="X")
    add_line(ax=ax1, data=results,
             xid='time', yid="Y", label="Y", color="darkblue")

    ax1.legend()
    plt.show()
