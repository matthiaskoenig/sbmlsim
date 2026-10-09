"""
Example shows basic model simulations and plotting.
"""

from matplotlib import pyplot as plt

from sbmlsim import sensitivity
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.result import ScanResult
from sbmlsim.simulation import Formula, Scan, Simulation, sampling
from sbmlsim.simulator import Simulator


def plot_results(res: ScanResult, filename: str) -> None:
    """Plot the mean and the range of the simulations of a scan."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    axes = (ax1, ax2, ax3, ax4)

    summary = res.summary(statistics=["mean", "min", "max"])
    times = summary["time"].values
    ax: plt.Axes
    for ax in (ax1, ax3):
        for sid, color in [
            ("[X]", "tab:blue"),
            ("[Y]", "tab:red"),
            ("[Z]", "tab:green"),
        ]:
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
                times, summary[sid].sel(statistic="mean").values, color=color, label=sid
            )

    for ax in (ax2, ax4):
        ax.plot(
            summary["[X]"].sel(statistic="mean").values,
            summary["[Y]"].sel(statistic="mean").values,
            color="black",
            label="Y~X",
        )

    for ax in (ax3, ax4):
        ax.set_xscale("log")
        ax.set_yscale("log")

    for ax in (ax1, ax3):
        ax.set_xlabel("time [second]")
    for ax in (ax2, ax4):
        ax.set_xlabel("value [dimensionless]")

    for ax in axes:
        ax.set_ylabel("value [dimensionless]")
        ax.legend()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def run_sensitivity() -> None:
    """Parameter sensitivity simulations: a local design and lognormal draws."""
    simulator = Simulator()
    model = simulator.load(REPRESSILATOR_SBML)
    model.set_selections(["time", "[X]", "[Y]", "[Z]"])
    tcsim = Simulation(end=200, steps=2000)
    parameters = sampling.parameters_of(model)

    # the parameters drawn from lognormal distributions around their references
    draws = sampling.random(
        {pid: sampling.LogNormal(cv=0.03) for pid in parameters},
        50,
        seed=1234,
        model=model,
    )
    res_distrib_scan = simulator.run(model, Scan(tcsim, [draws]))

    # every parameter alone 10 % up and down
    local = sampling.local(parameters, delta=0.1, model=model)
    res_diff_scan = simulator.run(model, Scan(tcsim, [local]))

    # the local indices of the maximum of X: a run with an observable, which keeps
    # no timecourse, so the figures above come from the runs without observables
    res_local = simulator.run(
        model, Scan(tcsim, [local]), [Formula("x_max", "max([X])")]
    )
    normalized = sensitivity.local(res_local).index("normalized")
    print(normalized.to_pandas().round(3))

    # create figures
    plot_results(res_distrib_scan, "model_sensitivity_distribution.png")
    plot_results(res_diff_scan, "model_sensitivity_difference.png")


if __name__ == "__main__":
    run_sensitivity()
