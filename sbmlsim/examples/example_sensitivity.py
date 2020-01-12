"""
Example shows basic model simulations and plotting.
"""
from sbmlsim.model import load_model
from sbmlsim.simulation_serial import SimulatorSerial as Simulator
from sbmlsim.timecourse import TimecourseSim, Timecourse, ensemble

from sbmlsim.parametrization import ChangeSet

from sbmlsim.plotting_matplotlib import add_line, plt
from sbmlsim.tests.constants import MODEL_REPRESSILATOR


def run_sensitivity():
    """ Parameter sensitivity simulations.

    :return:
    """
    simulator = Simulator(MODEL_REPRESSILATOR)

    # parameter sensitivity
    # FIXME: make work with parallel
    r = load_model(MODEL_REPRESSILATOR)
    changeset = ChangeSet.parameter_sensitivity_changeset(r)
    tc_sim = TimecourseSim([
            Timecourse(start=0, end=100, steps=100),
            Timecourse(start=0, end=200, steps=100, model_changes={"boundary_condition": {"X": True}}),
            Timecourse(start=0, end=100, steps=100, model_changes={"boundary_condition": {"X": False}}),
        ])
    tc_sims = ensemble(tc_sim, changeset=changeset)
    result = simulator.timecourses(tc_sims)

    # create figure
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    add_line(ax=ax1, data=result,
             xid='time', yid="X", label="X")
    add_line(ax=ax1, data=result,
             xid='time', yid="Y", label="Y", color="darkblue")
    add_line(ax=ax1, data=result,
             xid='time', yid="Z", label="Z", color="darkorange")

    ax1.legend()
    plt.show()


if __name__ == "__main__":
    run_sensitivity()
