"""Example showing basic simulations and plotting.

The figure is written to `timecourse.png` in the working directory.
"""

from matplotlib import pyplot as plt

from sbmlsim.console import console
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.result import XResult
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulator import SimulatorSerial


def run_timecourse_examples() -> None:
    """Run various simulations."""
    simulator = SimulatorSerial(model=REPRESSILATOR_SBML)

    # 1. simple simulation, the output are the steps of the integrator
    console.rule(title="simple simulation")
    simulation = Simulation(end=100)
    xr1: XResult = simulator.run_simulation(simulation)
    console.print(simulation)

    # 2. simulation with changes before the initialization
    console.rule(title="parameter change")
    simulation = Simulation(end=100, preinit_changes={"X": 10, "Y": 200})
    xr2: XResult = simulator.run_simulation(simulation)
    console.print(simulation)

    # 3. changes at a time, on an equidistant output grid
    console.rule(title="change at a time")
    simulation = Simulation(
        end=200, changes=[Change(100, {"X": 10, "Y": 20})], steps=200
    )
    xr3: XResult = simulator.run_simulation(simulation)
    console.print(simulation)

    # create figure
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    ax1.set_title("simple simulation")
    ax2.set_title("parameter change")
    ax3.set_title("change at a time")

    for xres, ax in [(xr1, ax1), (xr2, ax2), (xr3, ax3)]:
        console.print(xres)
        ax.plot(xres["time"], xres["[X]"], label="[X]")
        ax.plot(xres["time"], xres["[Y]"], label="[Y]")
        ax.plot(xres["time"], xres["[Z]"], label="[Z]")

    for ax in (ax1, ax2, ax3):
        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("concentration")
    fig.savefig("timecourse.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_timecourse_examples()
