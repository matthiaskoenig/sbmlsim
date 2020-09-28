"""
Example shows basic model simulations and plotting.
"""
from sbmlsim.plot.plotting_matplotlib import add_line, plt
from sbmlsim.simulation import Dimension, ScanSim, Timecourse, TimecourseSim
from sbmlsim.simulation.sensititvity import ModelSensitivity
from sbmlsim.simulator import SimulatorSerial as Simulator
from sbmlsim.test import MODEL_REPRESSILATOR


def plot_results(xres):
    unit_time = "s"
    unit_value = "dimensionless"
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    axes = (ax1, ax2, ax3, ax4)

    for ax in (ax1, ax3):
        add_line(
            ax=ax,
            xres=xres,
            xid="time",
            yid="X",
            label="X",
            xunit=unit_time,
            yunit=unit_value,
        )
        add_line(
            ax=ax,
            xres=xres,
            xid="time",
            yid="Y",
            label="Y",
            color="darkblue",
            xunit=unit_time,
            yunit=unit_value,
        )
        add_line(
            ax=ax,
            xres=xres,
            xid="time",
            yid="Z",
            label="Z",
            color="darkorange",
            xunit=unit_time,
            yunit=unit_value,
        )

    for ax in (ax2, ax4):
        add_line(
            ax=ax,
            xres=xres,
            xid="X",
            yid="Y",
            label="Y~X",
            color="black",
            xunit=unit_value,
            yunit=unit_value,
        )

    for ax in (ax3, ax4):
        ax.set_xscale("log")
        ax.set_yscale("log")

    for ax in (ax1, ax3):
        ax.set_xlabel(f"time [{unit_time}]")
    for ax in (ax2, ax4):
        ax.set_xlabel(f"value [{unit_value}]")

    for ax in axes:
        ax.set_ylabel(f"value [{unit_value}]")
        ax.legend()
    plt.show()


def run_sensitivity():
    """Parameter sensitivity simulations.

    :return:
    """
    simulator = Simulator(MODEL_REPRESSILATOR)

    # parameter sensitivity
    tcsim = TimecourseSim(
        [
            Timecourse(start=0, end=200, steps=2000),
        ]
    )
    distrib_scan = ModelSensitivity.distribution_sensitivity_scan(
        model=simulator.model, simulation=tcsim, cv=0.03, size=50
    )
    res_distrib_scan = simulator.run_scan(distrib_scan)

    diff_scan = ModelSensitivity.difference_sensitivity_scan(
        model=simulator.model, simulation=tcsim, difference=0.1
    )
    res_diff_scan = simulator.run_scan(diff_scan)

    # create figure
    plot_results(res_distrib_scan)
    plot_results(res_diff_scan)


if __name__ == "__main__":
    run_sensitivity()
