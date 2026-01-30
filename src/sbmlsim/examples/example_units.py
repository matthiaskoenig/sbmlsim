"""
Example for handling units in simulations and results.
"""

import numpy as np
from matplotlib import pyplot as plt
from pymetadata.console import console

from sbmlsim.resources import DEMO_SBML
from sbmlsim.simulation import Dimension, ScanSim, Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.units import UnitsInformation
from sbmlsim.result import XResult


def run_demo_example():
    """Run various timecourses."""
    simulator = SimulatorSerial(DEMO_SBML)

    # units information
    uinfo = UnitsInformation.from_sbml(DEMO_SBML)
    Q_ = uinfo.Q_

    # 1. simple timecourse simulation
    print("*** setting concentrations and amounts ***")

    # FIXME: units of changes are not used for conversion
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
    xres: XResult = simulator.run_scan(tc_scan)
    xres.uinfo = uinfo

    console.log(xres)

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

    ax: plt.Axes
    for ax, ax_units in dict(zip(axes, axes_units)).items():
        xunit = ax_units["xunit"]
        yunit = ax_units["yunit"]

        for key in ["[e__A]", "[e__B]", "[e__C]", "[c__A]", "[c__B]", "[c__C]"]:
            ax.plot(
                Q_(xres["time"].values, xres.uinfo["time"]).to(xunit).m,
                Q_(xres[key].values, xres.uinfo[key]).to(yunit).m,
                label=f"{key} [{yunit}]",
            )
        ax.legend()
        ax.set_xlabel(f"time [{xunit}]")
        ax.set_ylabel(f"concentration [{yunit}]")

    plt.show()


if __name__ == "__main__":
    run_demo_example()
