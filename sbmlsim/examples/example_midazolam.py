"""
Example showing basic timecourse simulations and plotting.
"""
import os
from matplotlib import pyplot as plt

from sbmlsim.simulation_serial import SimulatorSerial as Simulator
from sbmlsim.timecourse import Timecourse, TimecourseSim, ensemble
from sbmlsim.tests.constants import MODEL_MIDAZOLAM, MODEL_MIDAZOLAM_BODY
from sbmlsim.plotting_matplotlib import add_line


def run_timecourse_liver():
    """ Run various timecourses. """
    simulator = Simulator(MODEL_MIDAZOLAM)
    Q_ = simulator.ureg.Quantity

    # 1. simple timecourse simulation
    tc_sim = TimecourseSim([
        Timecourse(start=0, end=100, steps=100,
                   changes={
                       '[mid_ext]': Q_(10, "mM"),
                       'MIDIM_Vmax': Q_(10, "mmole_per_min"),
                   }),
        Timecourse(start=0, end=100, steps=100,
               changes={
                    'MIDIM_Vmax': Q_(20, "mmole_per_min"),
               }),

        ]
    )
    s = simulator.timecourses(tc_sim)

    # create figure
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    # species ids are amounts
    ax1.set_ylabel("amounts [mmole]")
    for sid in ["mid_ext", "mid1oh_ext", "mid", "mid1oh"]:
        add_line(ax1, s, "time", sid, label=sid)
    # [species ids] are concentrations
    ax2.set_ylabel("concentrations [mM]")
    for sid in ["mid_ext", "mid1oh_ext", "mid", "mid1oh"]:
        add_line(ax2, s, "time", f"[{sid}]", label=sid)

    for ax in (ax1, ax2):
        ax.legend()
        ax.set_xlabel("time [min]")
    plt.show()


def run_timecourse_body():
    """ Run various timecourses. """
    simulator = Simulator(MODEL_MIDAZOLAM_BODY)
    Q_ = simulator.ureg.Quantity

    # 1. simple timecourse simulation
    tc_sim = TimecourseSim([
        Timecourse(start=0, end=300, steps=500,
                   changes={
                       'IVDOSE_mid': Q_(1, "mg"),
                       'Ka_mid': Q_(60, "per_hr"),
                   }),
        ]
    )
    s = simulator.timecourses(tc_sim)

    # create figure
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    # species ids are amounts
    ax1.set_ylabel("concentration [mM]")
    for sid in ["mid", "mid1oh"]:
        for cid in ["ve", "gu_blood", "li_blood"]:
            add_line(ax1, s, "time", f"C{cid}_{sid}", label=f"C{cid}_{sid}")

        add_line(ax2, s, "time", f"Cve_{sid}", label=f"Cve_{sid}")
        add_line(ax2, s, "time", f"Car_{sid}", label=f"Car_{sid}")

    for ax in (ax1, ax2):
        ax.legend()
        ax.set_xlabel("time [min]")
    plt.show()


if __name__ == "__main__":
    # run_timecourse_liver()
    run_timecourse_body()

