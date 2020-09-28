"""
Testing plotting functionality.
"""
import numpy as np

from sbmlsim.plot.plotting_matplotlib import add_line, plt
from sbmlsim.simulation import Dimension, ScanSim, Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial as Simulator
from sbmlsim.test import MODEL_REPRESSILATOR


def test_plotting():
    simulator = Simulator(MODEL_REPRESSILATOR)
    Q_ = simulator.ureg.Quantity
    scan1d = ScanSim(
        simulation=TimecourseSim(
            [
                Timecourse(start=0, end=100, steps=400),
                Timecourse(
                    start=0, end=60, steps=100, changes={"[X]": Q_(10, "dimensionless")}
                ),
                Timecourse(
                    start=0, end=60, steps=100, changes={"X": Q_(10, "dimensionless")}
                ),
            ]
        ),
        dimensions=[
            Dimension(
                "dim1",
                changes={
                    "Y": Q_(
                        np.random.normal(loc=10.0, scale=3.0, size=50), "dimensionless"
                    ),
                },
            )
        ],
    )
    xres = simulator.run_scan(scan1d)

    # create figure
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    add_line(
        ax=ax1,
        xres=xres,
        xunit="min",
        yunit="dimensionless",
        xid="time",
        yid="X",
        label="X",
        color="darkgreen",
    )
    add_line(
        ax=ax1,
        xres=xres,
        xunit="min",
        yunit="dimensionless",
        xid="time",
        yid="Y",
        label="Y",
        color="darkblue",
    )

    ax1.legend()
    plt.show()
