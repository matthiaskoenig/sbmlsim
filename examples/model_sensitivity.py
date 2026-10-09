"""Parameter sensitivity of the repressilator: lognormal draws and a local design.

The figures show the mean and the range of [X], [Y] and [Z] over the
simulations of a scan and the phase plane of [X] and [Y].
"""

from matplotlib import pyplot as plt

from sbmlsim import sensitivity
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.result import ScanResult
from sbmlsim.simulation import Formula, Scan, Simulation, sampling
from sbmlsim.simulator import Simulator

#: the observables of the concentrations and their names in the figures
SPECIES = {"x": ("[X]", "tab:blue"), "y": ("[Y]", "tab:red"), "z": ("[Z]", "tab:green")}


def _label(name: str, unit: str) -> str:
    """Get an axis label with the unit if it has one."""
    return f"{name} [{unit}]" if unit and unit != "dimensionless" else name


def plot_results(res: ScanResult, filename: str) -> None:
    """Plot the mean and the range of the simulations of a scan."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    summary = res.summary(statistics=["mean", "min", "max"])
    times = summary["time"].values
    ax: plt.Axes
    for ax in (ax1, ax3):
        for sid, (name, color) in SPECIES.items():
            # range of the simulations
            ax.fill_between(
                times,
                summary[sid].sel(statistic="min").values,
                summary[sid].sel(statistic="max").values,
                color=color,
                alpha=0.3,
            )
            # mean line
            ax.plot(
                times,
                summary[sid].sel(statistic="mean").values,
                color=color,
                label=name,
            )
        ax.set_xlabel(_label("time", res.units["time"]))
        ax.set_ylabel(_label("concentration", res.units["x"]))
        ax.legend()

    for ax in (ax2, ax4):
        ax.plot(
            summary["x"].sel(statistic="mean").values,
            summary["y"].sel(statistic="mean").values,
            color="black",
        )
        ax.set_xlabel(_label("[X]", res.units["x"]))
        ax.set_ylabel(_label("[Y]", res.units["y"]))

    for ax in (ax3, ax4):
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


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
    plot_results(res_distrib_scan, "model_sensitivity_distribution.png")
    plot_results(res_diff_scan, "model_sensitivity_difference.png")


if __name__ == "__main__":
    run_sensitivity()
