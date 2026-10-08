"""
Example for handling units in simulations and results.
"""

import numpy as np
from matplotlib import pyplot as plt

from sbmlsim import Q
from sbmlsim.console import console
from sbmlsim.resources import DEMO_SBML
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator


def run_demo_example():
    """Run various timecourses."""
    # 1. simple timecourse simulation
    print("*** setting concentrations and amounts ***")

    # the quantities are converted into the units of the model
    tc_scan = Scan(
        simulation=Simulation(
            end=10,
            steps=100,
            preinit_changes={
                "[e__A]": Q(10, "mM"),
                "[e__B]": Q(1, "mmole/litre"),
                "[e__C]": Q(1, "mole/m**3"),
                "c__A": Q(1e-5, "mole"),
                "c__B": Q(10, "µmole"),
                "Vmax_bA": Q(300.0, "mole/min"),
            },
        ),
        dimensions=[
            Dimension(
                "dim1",
                values={"[e__A]": Q(np.linspace(5, 15, num=20), "mM")},
            )
        ],
    )

    # the result carries the units of the model, `res.units`
    res = Simulator().run(DEMO_SBML, tc_scan)
    console.log(res)

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
    for ax, ax_units in dict(zip(axes, axes_units, strict=False)).items():
        xunit = ax_units["xunit"]
        yunit = ax_units["yunit"]

        keys = ["[e__A]", "[e__B]", "[e__C]", "[c__A]", "[c__B]", "[c__C]"]
        for k, key in enumerate(keys):
            # the values are (dim1, time), one line per point of the scan
            lines = ax.plot(
                res.quantity("time").to(xunit).m,
                res.quantity(key).to(yunit).m.T,
                color=f"C{k}",
            )
            # the lines of a variable share its color and its entry of the legend
            lines[0].set_label(f"{key} [{yunit}]")
        ax.legend()
        ax.set_xlabel(f"time [{xunit}]")
        ax.set_ylabel(f"concentration [{yunit}]")

    fig.savefig("units.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_demo_example()
