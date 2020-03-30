"""
Example shows basic model simulations and plotting.
"""
import numpy as np

from sbmlsim.simulation import Timecourse, TimecourseSim, ScanSim, Dimension
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.result import XResult

from sbmlsim.tests.constants import MODEL_REPRESSILATOR


def run_scan0d() -> XResult:
    """Perform a parameter 0D scan, i.e., simple simulation"""
    simulator = SimulatorSerial(model=MODEL_REPRESSILATOR)
    Q_ = simulator.ureg.Quantity

    scan0d = ScanSim(
        simulation=TimecourseSim([
            Timecourse(start=0, end=100, steps=100, changes={}),
            Timecourse(start=0, end=60, steps=100, changes={'[X]': Q_(10, "dimensionless")}),
            Timecourse(start=0, end=60, steps=100, changes={'X': Q_(10, "dimensionless")}),
        ]),
        dimensions=[]
    )
    return simulator.run_scan(scan0d)


def run_scan1d() -> XResult:
    """Perform a parameter scan"""
    simulator = SimulatorSerial(model=MODEL_REPRESSILATOR)
    Q_ = simulator.ureg.Quantity

    scan1d = ScanSim(
        simulation=TimecourseSim([
            Timecourse(start=0, end=100, steps=100, changes={}),
            Timecourse(start=0, end=60, steps=100,
                       changes={'[X]': Q_(10, "dimensionless")}),
            Timecourse(start=0, end=60, steps=100,
                       changes={'X': Q_(10, "dimensionless")}),
        ]),
        dimensions=[
            Dimension("dim1", index=np.arange(8), changes={
                'n': Q_(np.linspace(start=2, stop=10, num=8), "dimensionless"),
            })
        ]
    )

    return simulator.run_scan(scan1d)


def run_parameter_scan2d() -> XResult:
    """Perform a parameter scan"""
    simulator = SimulatorSerial(model=MODEL_REPRESSILATOR)
    Q_ = simulator.ureg.Quantity

    scan2d = ScanSim(
        simulation=TimecourseSim([
            Timecourse(start=0, end=100, steps=100, changes={}),
            Timecourse(start=0, end=60, steps=100, changes={'[X]': Q_(10, "dimensionless")}),
            Timecourse(start=0, end=60, steps=100, changes={'X': Q_(10, "dimensionless")}),
        ]),
        dimensions=[
            Dimension("dim1", index=np.arange(8), changes={
                'n': Q_(np.linspace(start=2, stop=10, num=8), "dimensionless"),
            }),
            Dimension("dim2", index=np.arange(4), changes={
                'Y': Q_(np.linspace(start=10, stop=20, num=4), "dimensionless"),
            }),
        ]
    )
    return simulator.run_scan(scan2d)


def run_scan1d_distribution() -> XResult:
    """Perform a parameter scan by sampling from a distribution"""
    simulator = SimulatorSerial(model=MODEL_REPRESSILATOR)
    Q_ = simulator.ureg.Quantity

    scan1d = ScanSim(
        simulation=TimecourseSim([
            Timecourse(start=0, end=100, steps=100, changes={}),
            Timecourse(start=0, end=60, steps=100,
                       changes={'[X]': Q_(10, "dimensionless")}),
            Timecourse(start=0, end=60, steps=100,
                       changes={'X': Q_(10, "dimensionless")}),
        ]),
        dimensions=[
            Dimension("dim1", index=np.arange(50), changes={
                'n': Q_(np.random.normal(loc=5.0, scale=0.2, size=50), "dimensionless"),
            })
        ]
    )
    return simulator.run_scan(scan1d)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    column = 'PX'

    # scan0d
    xres = run_scan0d()

    for key in ['PX', 'PY', 'PZ']:
        plt.plot(xres.xds['time'], xres.xds[key], label=key)
    plt.legend()
    plt.show()

    # xres.xds['PX'].plot()
    # xres.xds['PY'].plot()
    # xres.xds['PZ'].plot()
    # plt.show()



    exit(0)
    # scan1d
    xres = run_scan1d()
    xres.xds[column].plot()
    plt.show()

    # scan1d_distrib
    xres = run_scan1d_distribution()
    xres.xds[column].plot()
    plt.show()
    da = xres.xds[column]
    for k in range(xres.xds.dims["dim1"]):
        # individual timecourses
        plt.plot(da.coords['time'], da.isel(dim1=k))

    plt.plot(da.coords['time'], da.mean(dim="dim1"), color="black", linewidth=4.0)
    plt.plot(da.coords['time'], da.min(dim="dim1"), color="black", linewidth=2.0)
    plt.plot(da.coords['time'], da.max(dim="dim1"), color="black", linewidth=2.0)
    plt.show()

    # scan2d
    xres = run_parameter_scan2d()
    xres.xds[column].plot()
    plt.show()
