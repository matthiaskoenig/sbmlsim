"""
Testing plotting functionality.
"""
from matplotlib import pyplot as plt
from sbmlsim.plotting_matplotlib import add_line

import sbmlsim
from sbmlsim.simulation_serial import timecourses
from sbmlsim.timecourse import Timecourse, TimecourseSim
from sbmlsim.parametrization import ChangeSet
from sbmlsim.tests.constants import MODEL_REPRESSILATOR


def test_plotting():
    r = sbmlsim.load_model(MODEL_REPRESSILATOR)
    changeset = ChangeSet.parameter_sensitivity_changeset(r, sensitivity=0.5)
    tc_sims = TimecourseSim(
        Timecourse(start=0, end=400, steps=400),
    ).ensemble(changeset)
    result = timecourses(r, tc_sims)

    # create figure
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    add_line(ax=ax1, data=result,
             xid='time', yid="X", label="X")
    add_line(ax=ax1, data=result,
             xid='time', yid="Y", label="Y", color="darkblue")

    ax1.legend()
    plt.show()
