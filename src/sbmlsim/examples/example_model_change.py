"""
Examples for model changes.

For instance clamping species to given formulas.
"""

import pandas as pd

from sbmlsim.model import ModelChange, RoadrunnerSBMLModel
from sbmlsim.plot.serialization_matplotlib import plt
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.result import XResult


def run_model_change_example1():
    """Manually clamping species.

    :return:
    """
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    r = model.r
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
    simulator = SimulatorSerial(REPRESSILATOR_SBML)

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
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ax.set_xlabel("time")
    ax.set_ylabel("concentration")

    for sid in ["X", "Y", "Z"]:
        ax.plot(xres["time"], xres[f"[{sid}]"], label=sid)

    ax.legend()
    plt.show()


def run_model_clamp2():
    def plot_result(xres: XResult, title: str = None) -> None:
        """Plot the results with title."""
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        fig.subplots_adjust(wspace=0.3, hspace=0.3)

        t = xres["time"]
        ax.plot(t, xres["[X]"], label="X")
        ax.plot(t, xres["[Y]"], label="Y")

        if title:
            ax.set_title(title)

        ax.legend()
        plt.show()

    # reference simulation
    simulator = SimulatorSerial(REPRESSILATOR_SBML)
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
    xres = simulator.run_timecourse(tcsim)
    assert isinstance(xres, XResult)
    plot_result(xres, "clamp experiment (220-420)")


if __name__ == "__main__":
    run_model_change_example1()
    run_model_clamp1()
    run_model_clamp2()
