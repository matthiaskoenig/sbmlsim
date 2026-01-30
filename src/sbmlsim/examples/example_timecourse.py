"""Example showing basic timecourse simulations and plotting."""

from pymetadata.console import console

from sbmlsim.plot.serialization_matplotlib import plt
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerialRR
from sbmlsim.xresult import XResult


def run_timecourse_examples():
    """Run various timecourses."""
    simulator = SimulatorSerialRR.from_sbml(REPRESSILATOR_SBML)

    # 1. simple timecourse simulation
    console.rule(title="simple timecourse")
    tc_sim = TimecourseSim(Timecourse(start=0, end=100, steps=100))
    xr1: XResult = simulator.run_timecourse(tc_sim)
    console.print(tc_sim)

    # 2. timecourse with parameter changes
    console.rule(title="parameter change")
    tc_sim = TimecourseSim(
        Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 200})
    )
    xr2: XResult = simulator.run_timecourse(tc_sim)
    console.print(tc_sim)

    # 3. combined timecourses
    console.rule(title="combined timecourse")
    tc_sim = TimecourseSim(
        [
            Timecourse(start=0, end=100, steps=100),
            Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 20}),
        ]
    )
    xr3: XResult = simulator.run_timecourse(tc_sim)
    console.print(tc_sim)

    # create figure
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    ax1.set_title("simple timecourse")
    ax2.set_title("parameter change")
    ax3.set_title("combined timecourse")

    for xres, ax in [(xr1, ax1), (xr2, ax2), (xr3, ax3)]:
        console.print(xres)
        ax.plot(xres.time, xres["[X]"], label="[X]")
        ax.plot(xres.time, xres["[Y]"], label="[Y]")
        ax.plot(xres.time, xres["[Z]"], label="[Z]")

    for ax in (ax1, ax2, ax3):
        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("concentration")
    plt.show()


if __name__ == "__main__":
    run_timecourse_examples()
