"""Parameter sensitivity of the repressilator: lognormal draws and a local design.

The figures show the mean and the range of [X], [Y] and [Z] over the
simulations of a scan and the phase plane of [X] and [Y].
"""

from typing import Literal

from sbmlsim import sensitivity
from sbmlsim.plot import Chart, ChartAxis
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.result import ScanResult
from sbmlsim.simulation import Formula, Scan, Simulation, sampling
from sbmlsim.simulator import Simulator

#: the observables of the concentrations and their names in the figures
SPECIES = {"x": ("[X]", "#1f77b4"), "y": ("[Y]", "#d62728"), "z": ("[Z]", "#2ca02c")}


def plot_results(res: ScanResult, filename: str) -> None:
    """Plot the mean and the range of the simulations of a scan."""
    chart = Chart(width=720, height=720, rows=2, columns=2)
    summary = res.summary(statistics=["mean", "min", "max"])
    times = summary["time"].values
    mean = {sid: summary[sid].sel(statistic="mean").values for sid in SPECIES}

    scales: list[Literal["linear", "log"]] = ["linear", "log"]
    for row, scale in enumerate(scales):
        # the lines of a log axis have no point where a concentration is not positive
        concentration = chart.plot(
            row,
            0,
            x=ChartAxis(label="time", unit=res.units["time"]),
            y=ChartAxis(label="concentration", unit=res.units["x"], scale=scale),
        )
        for sid, (name, color) in SPECIES.items():
            lower = summary[sid].sel(statistic="min").values
            upper = summary[sid].sel(statistic="max").values
            concentration.band(
                times, lower, upper, median=mean[sid], color=color, label=name
            )

        phase = chart.plot(
            row,
            1,
            x=ChartAxis(label="[X]", unit=res.units["x"], scale=scale),
            y=ChartAxis(label="[Y]", unit=res.units["y"], scale=scale),
        )
        positive = (mean["x"] > 0) & (mean["y"] > 0) if scale == "log" else slice(None)
        phase.curve(mean["x"][positive], mean["y"][positive], color="#000000")

    chart.save(filename)


def run_sensitivity() -> None:
    """Parameter sensitivity simulations: a local design and lognormal draws."""
    simulator = Simulator()
    model = simulator.load(REPRESSILATOR_SBML)
    tcsim = Simulation(end=200, steps=2000)
    parameters = sampling.parameters_of(model)
    observables = [
        Formula("x", "[X]"),
        Formula("y", "[Y]"),
        Formula("z", "[Z]"),
        Formula("x_max", "max([X])"),
    ]

    # the parameters drawn from lognormal distributions around their references
    draws = sampling.random(
        {pid: sampling.LogNormal(cv=0.03) for pid in parameters},
        50,
        seed=1234,
        model=model,
    )
    res_distrib_scan = simulator.run(model, Scan(tcsim, [draws]), observables)

    # every parameter alone 10 % up and down
    local = sampling.local(parameters, delta=0.1, model=model)
    res_diff_scan = simulator.run(model, Scan(tcsim, [local]), observables)

    # the local indices of the maximum of X
    local_indices = sensitivity.local(res_diff_scan, observables=["x_max"])
    print(local_indices.index("normalized").to_pandas().round(3))

    # create figures
    plot_results(res_distrib_scan, "model_sensitivity_distribution.svg")
    plot_results(res_diff_scan, "model_sensitivity_difference.svg")


if __name__ == "__main__":
    run_sensitivity()
