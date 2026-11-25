"""
Example for handling units in simulations and results.
"""

import numpy as np

from sbmlsim import Q
from sbmlsim.console import console
from sbmlsim.plot import Chart, ChartAxis
from sbmlsim.plot.colors import CYCLE
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

    # create the figure, every axis shows the quantities in its own units
    chart = Chart(width=1200, height=300, columns=4)
    axes_units = [
        {"xunit": "s", "yunit": "mM"},
        {"xunit": "ms", "yunit": "mole/litre"},
        {"xunit": "min", "yunit": "nM"},
        {"xunit": "hr", "yunit": "pmole/cm**3"},
    ]
    keys = ["[e__A]", "[e__B]", "[e__C]", "[c__A]", "[c__B]", "[c__C]"]
    for column, ax_units in enumerate(axes_units):
        plot = chart.plot(
            0,
            column,
            x=ChartAxis(label="time", unit=ax_units["xunit"]),
            y=ChartAxis(label="concentration", unit=ax_units["yunit"]),
        )
        for k, key in enumerate(keys):
            # the values are (dim1, time), one line per point of the scan, the
            # lines of a variable share its color and its entry of the legend
            for point in range(res.ds.sizes["dim1"]):
                plot.curve(
                    res.q["time"],
                    res.isel(dim1=point).q[key],
                    color=CYCLE[k],
                    label=key if point == 0 else None,
                    legend=key,
                )

    chart.save("units.svg")


if __name__ == "__main__":
    run_demo_example()
