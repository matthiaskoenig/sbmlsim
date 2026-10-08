"""Example shows basic model simulations and plotting with scan."""

import numpy as np

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.result import ScanResult
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator


def _simulation() -> Simulation:
    """Get a simulation with two changes during it, on a grid of 100 points."""
    return Simulation(
        end=220,
        changes=[Change(100, {"[X]": 10}), Change(160, {"X": 10})],
        steps=100,
    )


def run_scan0d() -> ScanResult:
    """Perform a parameter 0D scan, i.e., simple simulation."""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    return Simulator().run(model, Scan(_simulation()))


def run_scan1d() -> ScanResult:
    """Perform a 1D parameter scan.

    Scanning a single parameter.
    """
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    scan1d = Scan(
        simulation=_simulation(),
        dimensions=[
            Dimension("dim1", values={"n": np.linspace(start=2, stop=10, num=8)}),
        ],
    )
    return Simulator().run(model, scan1d)


def run_scan2d() -> ScanResult:
    """Perform a 2D parameter scan."""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    scan2d = Scan(
        simulation=_simulation(),
        dimensions=[
            Dimension("dim1", values={"n": np.linspace(start=2, stop=10, num=8)}),
            Dimension("dim2", values={"Y": np.logspace(start=2, stop=2.5, num=4)}),
        ],
    )
    return Simulator().run(model, scan2d)


def run_scan1d_distribution() -> ScanResult:
    """Perform a parameter scan by sampling from a distribution."""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    rng = np.random.default_rng(seed=1234)
    scan1d = Scan(
        simulation=_simulation(),
        dimensions=[
            Dimension("dim1", values={"n": rng.normal(loc=5.0, scale=0.2, size=50)}),
        ],
    )
    return Simulator().run(model, scan1d)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    column = "PX"

    # scan0d
    res = run_scan0d()
    fig, ax = plt.subplots()
    for key in ["PX", "PY", "PZ"]:
        ax.plot(res["time"], res[key], label=key)
    ax.set_xlabel(f"time [{res.units['time']}]")
    ax.set_ylabel("amount")
    ax.legend()
    fig.savefig("scan0d.png", bbox_inches="tight")
    plt.close(fig)

    # scan1d
    res = run_scan1d()
    print(res.ds)

    # scan1d_distrib
    res = run_scan1d_distribution()
    print(res.ds)

    da = res[column]
    fig, ax = plt.subplots()
    time = res["time"]
    for k in range(res.ds.sizes["dim1"]):
        # individual timecourses
        ax.plot(time, da.isel(dim1=k))

    ax.plot(time, da.mean(dim="dim1"), color="black", linewidth=4.0)
    ax.plot(time, da.min(dim="dim1"), color="black", linewidth=2.0)
    ax.plot(time, da.max(dim="dim1"), color="black", linewidth=2.0)
    ax.set_xlabel(f"time [{res.units['time']}]")
    ax.set_ylabel(f"{column} (mean, min and max in black)")
    fig.savefig("scan1d_distribution.png", bbox_inches="tight")
    plt.close(fig)
