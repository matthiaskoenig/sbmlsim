"""
Example for handling units in simulations and results.
"""
import os
from matplotlib import pyplot as plt

from sbmlsim.model import load_model
# from sbmlsim.simulation_ray import SimulatorParallel as Simulator
from sbmlsim.simulation_serial import SimulatorSerial as Simulator

from sbmlsim.timecourse import Timecourse, TimecourseSim
from sbmlsim.tests.constants import MODEL_DEMO
from sbmlsim.plotting_matplotlib import add_line

from pprint import pprint
from sbmlsim.timecourse import ensemble
from sbmlsim.parametrization import ChangeSet
from sbmlsim.model import load_model


def run_demo_example():
    """ Run various timecourses. """
    simulator = Simulator(MODEL_DEMO)
    # build quantities using the unit registry for the model
    Q_ = simulator.ureg.Quantity
    pprint(simulator.udict)

    # 1. simple timecourse simulation
    print("*** setting concentrations and amounts ***")
    tc_sim = TimecourseSim([
        Timecourse(start=0, end=10, steps=100,
                   changes={
                       "[e__A]": Q_(10, "mM"),
                       "[e__B]": Q_(1, "mmole/litre"),
                       "[e__C]": Q_(1, "mole/m**3"),
                       "c__A": Q_(1E-5, "mole"),
                       "c__B": Q_(10, "µmole"),
                       "Vmax_bA": Q_(300.0, "mole/min")
                   }
        )
    ])
    tc_sim.normalize(udict=simulator.udict, ureg=simulator.ureg)
    print(tc_sim)


    # FIXME: some problem with ensemble and unit conversion before normalization
    # r = load_model(MODEL_DEMO)
    #tc_sims = ensemble(tc_sim, ChangeSet.parameter_sensitivity_changeset(r, 0.2))
    tc_sims = [tc_sim]
    s = simulator.timecourses(tc_sims)

    # create figure
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    axes = (ax1, ax2, ax3, ax4)

    axes_units = [
        {"xunit": "s", "yunit": "mM"},
        {"xunit": "ms", "yunit": "mole/litre"},
        {"xunit": "min", "yunit": "nM"},
        {"xunit": "hr", "yunit": "pmole/cm**3"},
    ]

    for ax, ax_units in dict(zip(axes, axes_units)).items():

        xunit = ax_units["xunit"]
        yunit = ax_units["yunit"]

        for key in ["[e__A]", "[e__B]", "[e__C]", "[c__A]", "[c__B]", "[c__C]"]:
            add_line(ax, s, "time", key, xunit=xunit, yunit=yunit,
                           label=f"{key} [{yunit}]")

        ax.legend()
        ax.set_xlabel(f"time [{xunit}]")
        ax.set_ylabel(f"concentration [{yunit}]")

    plt.show()

if __name__ == "__main__":
    run_demo_example()
