"""Example shows basic model simulations and plotting with scan."""

import numpy as np

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Dimension, ScanSim, Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.result import XResult


def run_scan0d() -> XResult:
    """Perform a parameter 0D scan, i.e., simple simulation"""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    simulator = SimulatorSerial(model)

    scan0d = ScanSim(
        simulation=TimecourseSim(
            [
                Timecourse(start=0, end=100, steps=100, changes={}),
                Timecourse(start=0, end=60, steps=100, changes={"[X]": 10}),
                Timecourse(start=0, end=60, steps=100, changes={"X": 10}),
            ]
        ),
        dimensions=[],
    )
    return simulator.run_scan(scan0d)


def run_scan1d() -> XResult:
    """Perform a 1D parameter scan.

    Scanning a single parameter.
    """
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    simulator = SimulatorSerial(model)

    scan1d = ScanSim(
        simulation=TimecourseSim(
            [
                Timecourse(start=0, end=100, steps=100, changes={}),
                Timecourse(start=0, end=60, steps=100, changes={"[X]": 10}),
                Timecourse(start=0, end=60, steps=100, changes={"X": 10}),
            ]
        ),
        dimensions=[
            Dimension(
                "dim1",
                changes={
                    "n": np.linspace(start=2, stop=10, num=8),
                },
            )
        ],
    )

    return simulator.run_scan(scan1d)


def run_scan2d() -> XResult:
    """Perform a parameter scan"""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    simulator = SimulatorSerial(model)

    scan2d = ScanSim(
        simulation=TimecourseSim(
            [
                Timecourse(start=0, end=100, steps=100, changes={}),
                Timecourse(start=0, end=60, steps=100, changes={"[X]": 10}),
                Timecourse(start=0, end=60, steps=100, changes={"X": 10}),
            ]
        ),
        dimensions=[
            Dimension(
                "dim1",
                changes={
                    "n": np.linspace(start=2, stop=10, num=8),
                },
            ),
            Dimension(
                "dim2",
                changes={
                    "Y": np.logspace(start=2, stop=2.5, num=4),
                },
            ),
        ],
    )
    return simulator.run_scan(scan2d)


def run_scan1d_distribution() -> XResult:
    """Perform a parameter scan by sampling from a distribution"""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    simulator = SimulatorSerial(model)

    scan1d = ScanSim(
        simulation=TimecourseSim(
            [
                Timecourse(start=0, end=100, steps=100, changes={}),
                Timecourse(start=0, end=60, steps=100, changes={"[X]": 10}),
                Timecourse(start=0, end=60, steps=100, changes={"X": 10}),
            ]
        ),
        dimensions=[
            Dimension(
                "dim1",
                changes={
                    "n": np.random.normal(loc=5.0, scale=0.2, size=50),
                },
            )
        ],
    )
    return simulator.run_scan(scan1d)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    column = "PX"

    # scan0d
    xres = run_scan0d()
    for key in ["PX", "PY", "PZ"]:
        plt.plot(xres.time, xres[key], label=key)
    plt.legend()
    plt.show()

    # scan1d
    xres = run_scan1d()
    xres.xds[column].plot()
    plt.show()

    # scan1d_distrib
    xres = run_scan1d_distribution()
    print(xres.xds)

    xres[column].plot()
    plt.show()

    da = xres[column]
    for k in range(xres.sizes["dim1"]):
        # individual timecourses
        plt.plot(da.coords["time"], da.isel(dim1=k))

    plt.plot(da.coords["time"], da.mean(dim="dim1"), color="black", linewidth=4.0)
    plt.plot(da.coords["time"], da.min(dim="dim1"), color="black", linewidth=2.0)
    plt.plot(da.coords["time"], da.max(dim="dim1"), color="black", linewidth=2.0)
    plt.show()

    """
    # scan2d
    xres = run_scan2d()
    xres[column].plot()
    plt.show()
    """
