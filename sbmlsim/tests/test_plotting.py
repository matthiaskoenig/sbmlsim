"""
Testing plotting functionality.
"""
from matplotlib import pyplot as plt

import sbmlsim
from sbmlsim.simulation import TimecourseSimulation, Timecourse, timecourses
from sbmlsim.plotting_matlab import add_line
from sbmlsim.parametrization import ChangeSet

from sbmlsim.tests.settings import MODEL_REPRESSILATOR


def test_plotting():
    r = sbmlsim.load_model(MODEL_REPRESSILATOR)
    changeset = ChangeSet.parameter_sensitivity_changeset(r, sensitivity=0.5)
    tc_sims = TimecourseSimulation(
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
