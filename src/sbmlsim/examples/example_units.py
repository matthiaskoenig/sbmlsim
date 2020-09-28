"""
Example for handling units in simulations and results.
"""
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

from sbmlsim.plot.plotting_matplotlib import add_line
from sbmlsim.result import XResult
from sbmlsim.simulation import Dimension, ScanSim, Timecourse, TimecourseSim
from sbmlsim.simulator.simulation_serial import SimulatorSerial as Simulator
from sbmlsim.test import MODEL_DEMO


def run_demo_example():
    """ Run various timecourses. """
    simulator = Simulator(MODEL_DEMO)
    # build quantities using the unit registry for the model
    Q_ = simulator.ureg.Quantity
    pprint(simulator.udict)

    # 1. simple timecourse simulation
    print("*** setting concentrations and amounts ***")

    tc_scan = ScanSim(
        simulation=TimecourseSim(
            [
                Timecourse(
                    start=0,
                    end=10,
                    steps=100,
                    changes={
                        "[e__A]": Q_(10, "mM"),
                        "[e__B]": Q_(1, "mmole/litre"),
                        "[e__C]": Q_(1, "mole/m**3"),
                        "c__A": Q_(1e-5, "mole"),
                        "c__B": Q_(10, "µmole"),
                        "Vmax_bA": Q_(300.0, "mole/min"),
                    },
                )
            ]
        ),
        dimensions=[
            Dimension(
                "dim1",
                index=np.arange(20),
                changes={"[e__A]": Q_(np.linspace(5, 15, num=20), "mM")},
            )
        ],
    )

    # print(tc_sim)
    # xres = simulator.run_timecourse(tc_sim)  # type: XResult
    xres = simulator.run_scan(tc_scan)  # type: XResult
    # print(tc_sim)  # simulation has been unit normalized

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
            add_line(
                ax,
                xres,
                "time",
                key,
                xunit=xunit,
                yunit=yunit,
                label=f"{key} [{yunit}]",
            )
        ax.legend()
        ax.set_xlabel(f"time [{xunit}]")
        ax.set_ylabel(f"concentration [{yunit}]")

    plt.show()


if __name__ == "__main__":
    run_demo_example()
