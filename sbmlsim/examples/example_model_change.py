"""
Example of changing boundary conditions
"""

from sbmlsim.simulator import SimulatorSerial as Simulator
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.plot.plotting_matplotlib import add_line, plt

from sbmlsim.tests.constants import MODEL_REPRESSILATOR



def run_model_change_example():
    simulator = Simulator(MODEL_REPRESSILATOR)

    # setting a species as boundary condition
    tcsim = TimecourseSim([
            Timecourse(start=0, end=100, steps=100),
            Timecourse(start=0, end=50, steps=100, model_changes={"boundary_condition": {"X": True}}),
            Timecourse(start=0, end=100, steps=100, model_changes={"boundary_condition": {"X": False}}),
        ])
    xres = simulator.run_timecourse(tcsim)

    # create figure
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    add_line(ax1, xres, "time", "X", label="X")
    add_line(ax2, xres, "time", "Y", label="Y")
    add_line(ax3, xres, "time", "Z", label="Z")

    for ax in (ax1, ax2, ax3):
        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("concentration")
    plt.show()


if __name__ == "__main__":
    run_model_change_example()
