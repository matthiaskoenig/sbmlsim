"""
Example showing basic timecourse simulations and plotting.
"""
import os
from matplotlib import pyplot as plt

from sbmlsim.simulation_serial import SimulatorSerial as Simulator
from sbmlsim.timecourse import Timecourse, TimecourseSim, ensemble
from sbmlsim.tests.constants import MODEL_ACETAMINOPHEN
from sbmlsim.plotting_matplotlib import add_line


def run_timecourse_examples():
    """ Run various timecourses. """
    simulator = Simulator(MODEL_ACETAMINOPHEN)
    Q_ = simulator.ureg.Quantity

    # 1. simple timecourse simulation
    tc_sim = TimecourseSim(
        Timecourse(start=0, end=5*60, steps=1000,
                   changes={
                       '[apap_ext]': Q_(100, "mM"),
                    })
    )
    s = simulator.timecourses(tc_sim)

    # create figure
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    sids = ["apap_ext", "apap_sul_ext", "apap_mer_ext", "apap_cys_ext", "apap_glu_ext","apap_gsh_ext",]
    # species ids are amounts and [] concentrations
    ax1.set_ylabel("amounts [mmole]")
    ax2.set_ylabel("concentrations [mM]")
    for sid in sids:
        add_line(ax1, s, "time", sid, label=sid)
        add_line(ax2, s, "time", f"[{sid}]", label=sid)

    for ax in (ax1, ax2):
        ax.legend()
        ax.set_xlabel("time [min]")
    plt.show()


if __name__ == "__main__":
    run_timecourse_examples()
