"""
Examples for model changes.

For instance clamping species to given formulas.
"""
import pandas as pd

from sbmlsim.model import ModelChange, RoadrunnerSBMLModel
from sbmlsim.plot.plotting_matplotlib import add_line, plt
from sbmlsim.result import XResult
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial as Simulator
from sbmlsim.test import MODEL_REPRESSILATOR


def run_model_change_example1():
    """Manually clamping species.

    :return:
    """
    r = RoadrunnerSBMLModel.load_roadrunner_model(MODEL_REPRESSILATOR)
    RoadrunnerSBMLModel.set_timecourse_selections(r)

    s1 = r.simulate(start=0, end=100, steps=500)
    s1 = pd.DataFrame(s1, columns=s1.colnames)

    ModelChange.clamp_species(r, "X", "10.0")
    RoadrunnerSBMLModel.set_timecourse_selections(r)
    s2 = r.simulate(start=0, end=100, steps=500)
    s2 = pd.DataFrame(s2, columns=s2.colnames)
    s2.time = s2.time + 100.0

    ModelChange.clamp_species(r, "X", False)
    RoadrunnerSBMLModel.set_timecourse_selections(r)
    s3 = r.simulate(start=0, end=100, steps=500)
    s3 = pd.DataFrame(s3, columns=s3.colnames)
    s3.time = s3.time + 200.0

    ModelChange.clamp_species(r, "X", True)
    RoadrunnerSBMLModel.set_timecourse_selections(r)
    s4 = r.simulate(start=0, end=100, steps=500)
    s4 = pd.DataFrame(s4, columns=s4.colnames)
    s4.time = s4.time + 300.0

    ModelChange.clamp_species(r, "X", False)
    RoadrunnerSBMLModel.set_timecourse_selections(r)
    s5 = r.simulate(start=0, end=100, steps=500)
    s5 = pd.DataFrame(s5, columns=s5.colnames)
    s5.time = s5.time + 400.0

    plt.plot(s1.time, s1.X, "o-")
    plt.plot(s2.time, s2.X, "o-")
    plt.plot(s3.time, s3.X, "o-")
    plt.plot(s4.time, s4.X, "o-")
    plt.plot(s5.time, s5.X, "o-")
    plt.show()


def run_model_clamp1():
    """Using Timecourse simulations for clamps."""
    simulator = Simulator(MODEL_REPRESSILATOR)

    # setting a species as boundary condition
    tcsim = TimecourseSim(
        [
            Timecourse(start=0, end=100, steps=100),
            Timecourse(
                start=0,
                end=300,
                steps=100,
                model_manipulations={ModelChange.CLAMP_SPECIES: {"X": True}},
            ),
            Timecourse(
                start=0,
                end=200,
                steps=100,
                model_manipulations={ModelChange.CLAMP_SPECIES: {"X": False}},
            ),
        ]
    )
    xres = simulator.run_timecourse(tcsim)

    # create figure
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    for sid in ["X", "Y", "Z"]:
        add_line(
            ax1,
            xres,
            xid="time",
            yid=sid,
            xunit="second",
            yunit="dimensionless",
            label=sid,
        )

    for ax in (ax1,):
        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("concentration")
    plt.show()


def run_model_clamp2():
    def plot_result(result: XResult, title: str = None) -> None:
        # create figure
        fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        fig.subplots_adjust(wspace=0.3, hspace=0.3)

        add_line(
            ax=ax1,
            xres=result,
            xid="time",
            yid="X",
            label="X",
            xunit="second",
            yunit="dimensionless",
        )
        add_line(
            ax=ax1,
            xres=result,
            xid="time",
            yid="Y",
            label="Y",
            xunit="second",
            yunit="dimensionless",
            color="darkblue",
        )

        if title:
            ax1.set_title(title)

        ax1.legend()
        plt.show()

    # reference simulation
    simulator = Simulator(MODEL_REPRESSILATOR)
    tcsim = TimecourseSim(
        [
            Timecourse(start=0, end=220, steps=300, changes={"X": 10}),
            # clamp simulation
            Timecourse(
                start=0,
                end=200,
                steps=200,
                model_manipulations={ModelChange.CLAMP_SPECIES: {"X": True}},
            ),
            # free simulation
            Timecourse(
                start=0,
                end=400,
                steps=400,
                model_manipulations={ModelChange.CLAMP_SPECIES: {"X": False}},
            ),
        ]
    )
    result = simulator.run_timecourse(tcsim)
    assert isinstance(result, XResult)
    plot_result(result, "clamp experiment (220-420)")


if __name__ == "__main__":
    run_model_change_example1()
    # run_model_clamp1()
    # run_model_clamp2()
