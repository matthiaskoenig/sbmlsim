"""Example showing basic simulations and plotting.

The figure is written to `timecourse.png` in the working directory.
"""

from matplotlib import pyplot as plt

from sbmlsim.console import console
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.result import ScanResult
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulator import Simulator


def run_timecourse_examples() -> None:
    """Run various simulations."""
    simulator = Simulator()
    model = simulator.load(REPRESSILATOR_SBML)

    # 1. simple simulation, the output are the steps of the integrator
    console.rule(title="simple simulation")
    simulation = Simulation(end=100)
    res1: ScanResult = simulator.run(model, simulation)
    console.print(simulation)

    # 2. simulation with changes before the initialization
    console.rule(title="parameter change")
    simulation = Simulation(end=100, preinit_changes={"X": 10, "Y": 200})
    res2: ScanResult = simulator.run(model, simulation)
    console.print(simulation)

    # 3. changes at a time, on an equidistant output grid
    console.rule(title="change at a time")
    simulation = Simulation(
        end=200, changes=[Change(100, {"X": 10, "Y": 20})], steps=200
    )
    res3: ScanResult = simulator.run(model, simulation)
    console.print(simulation)

    # create figure
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    ax1.set_title("simple simulation")
    ax2.set_title("parameter change")
    ax3.set_title("change at a time")

    for res, ax in [(res1, ax1), (res2, ax2), (res3, ax3)]:
        console.print(res)
        ax.plot(res["time"], res["[X]"], label="[X]")
        ax.plot(res["time"], res["[Y]"], label="[Y]")
        ax.plot(res["time"], res["[Z]"], label="[Z]")

    for ax in (ax1, ax2, ax3):
        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("concentration")
    fig.savefig("timecourse.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_timecourse_examples()
