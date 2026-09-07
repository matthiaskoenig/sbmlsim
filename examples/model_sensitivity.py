"""
Example shows basic model simulations and plotting.
"""

from sbmlsim.plot.serialization_matplotlib import plt
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.result import XResult
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.simulation.sensitivity import ModelSensitivity
from sbmlsim.simulator import SimulatorSerial


def plot_results(xres: XResult, filename: str) -> None:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    axes = (ax1, ax2, ax3, ax4)

    ax: plt.Axes
    for ax in (ax1, ax3):
        for sid, color in [
            ("[X]", "tab:blue"),
            ("[Y]", "tab:red"),
            ("[Z]", "tab:green"),
        ]:
            # mean line
            ax.plot(
                xres["time"],
                xres.dim_mean(sid).magnitude,
                color=color,
                label=sid,
            )
            # shaded areas
            # TODO

    for ax in (ax2, ax4):
        ax.plot(
            xres.dim_mean("[X]").magnitude,
            xres.dim_mean("[Y]").magnitude,
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


def run_sensitivity():
    """Parameter sensitivity simulations.

    :return:
    """
    simulator = SimulatorSerial(REPRESSILATOR_SBML)

    # parameter sensitivity
    tcsim = TimecourseSim(
        [
            Timecourse(start=0, end=200, steps=2000),
        ]
    )

    model = simulator.model_loaded

    distrib_scan = ModelSensitivity.distribution_sensitivity_scan(
        model=model, simulation=tcsim, cv=0.03, size=50
    )
    res_distrib_scan = simulator.run_scan(distrib_scan)

    diff_scan = ModelSensitivity.difference_sensitivity_scan(
        model=model, simulation=tcsim, difference=0.1
    )
    res_diff_scan = simulator.run_scan(diff_scan)

    # create figures
    plot_results(res_distrib_scan, "model_sensitivity_distribution.png")
    plot_results(res_diff_scan, "model_sensitivity_difference.png")


if __name__ == "__main__":
    run_sensitivity()
