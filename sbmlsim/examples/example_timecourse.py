"""
Example showing basic timecourse simulations and plotting.
"""
import os
from matplotlib import pyplot as plt

from sbmlsim.model import load_model
# from sbmlsim.simulation_ray import SimulatorParallel as Simulator
from sbmlsim.simulation_serial import SimulatorSerial as Simulator

from sbmlsim.timecourse import Timecourse, TimecourseSim
from sbmlsim.tests.constants import MODEL_REPRESSILATOR


def run_timecourse_examples():
    """ Run various timecourses. """
    simulator = Simulator(MODEL_REPRESSILATOR)

    # 1. simple timecourse simulation
    print("*** simple timecourse ***")
    tc_sim = TimecourseSim(
        Timecourse(start=0, end=100, steps=100)
    )
    s1 = simulator.timecourses(tc_sim)
    print(tc_sim)


    # 2. timecourse with parameter changes
    print("*** parameter change ***")
    tc_sim = TimecourseSim(
        Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 200})
    )
    s2 = simulator.timecourses(tc_sim)
    print(tc_sim)

    # 3. combined timecourses
    print("*** combined timecourse ***")
    tc_sim = TimecourseSim([
            Timecourse(start=0, end=100, steps=100),
            Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 20}),
        ])
    s3 = simulator.timecourses(tc_sim)
    print(tc_sim)

    # 4. combined timecourses with model_change
    print("*** model change ***")
    tc_sim = TimecourseSim([
            Timecourse(start=0, end=100, steps=100),
            Timecourse(start=0, end=50, steps=100, model_changes={"boundary_condition": {"X": True}}),
            Timecourse(start=0, end=100, steps=100, model_changes={"boundary_condition": {"X": False}}),
        ])
    s4 = simulator.timecourses(tc_sim)
    print(tc_sim)

    # create figure
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    ax1.set_title("simple timecourse")
    ax2.set_title("parameter change")
    ax3.set_title("combined timecourse")
    ax4.set_title("model change")

    for s, ax in [(s1, ax1), (s2, ax2), (s3, ax3), (s4, ax4)]:
        df = s.frames[0]
        ax.plot(df.time, df.X, label="X")
        ax.plot(df.time, df.Y, label="Y")
        ax.plot(df.time, df.Z, label="Z")

    for ax in (ax1, ax2, ax3, ax4):
        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("concentration")
    plt.show()


if __name__ == "__main__":
    run_timecourse_examples()
