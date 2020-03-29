"""
Testing plotting functionality.
"""
from matplotlib import pyplot as plt
from sbmlsim.plot.plotting_matplotlib import add_line

from sbmlsim.simulator.simulation_serial import SimulatorSerial as Simulator
from sbmlsim.simulation.timecourse import Timecourse, TimecourseSim
from sbmlsim.tests.constants import MODEL_REPRESSILATOR


def test_plotting():
    simulator = Simulator(MODEL_REPRESSILATOR)

    changeset = ChangeSet.parameter_sensitivity_changeset(simulator.r, sensitivity=0.5)
    tcsims = ensemble(TimecourseSim([
            Timecourse(start=0, end=400, steps=400),
        ]), changeset)
    result = simulator.run_timecourse(tcsims)

    # create figure
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    add_line(ax=ax1, data=result,
             xid='time', yid="X", label="X")
    add_line(ax=ax1, data=result,
             xid='time', yid="Y", label="Y", color="darkblue")

    ax1.legend()
    plt.show()
