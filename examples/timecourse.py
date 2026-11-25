"""Example showing basic simulations and plotting.

The figure is written to `timecourse.svg` in the working directory.
"""

from sbmlsim.console import console
from sbmlsim.plot import Chart, ChartAxis
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

    # create the figure
    chart = Chart(width=1080, height=360, columns=3)
    cases = [
        ("simple simulation", res1),
        ("parameter change", res2),
        ("change at a time", res3),
    ]
    for column, (title, res) in enumerate(cases):
        console.print(res)
        plot = chart.plot(
            0,
            column,
            title=title,
            x=ChartAxis(label="time", unit=res.units["time"]),
            y=ChartAxis(label="concentration"),
        )
        for sid in ["[X]", "[Y]", "[Z]"]:
            plot.curve(res.q["time"], res.q[sid], label=sid)
    chart.save("timecourse.svg")


if __name__ == "__main__":
    run_timecourse_examples()
