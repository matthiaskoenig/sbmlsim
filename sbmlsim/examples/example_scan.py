"""
Example shows basic model simulations and plotting.
"""
import numpy as np

from sbmlsim.simulation import Timecourse, TimecourseSim, ScanSim, Dimension
from sbmlsim.simulator import SimulatorSerial, SimulatorParallel
from sbmlsim.units import Units
from sbmlsim.result import Result

from sbmlsim.tests.constants import MODEL_REPRESSILATOR


def run_parameter_scan0d():
    """Perform a parameter scan"""
    simulator = SimulatorSerial(model=MODEL_REPRESSILATOR)
    udict, ureg = Units.get_units_from_sbml(MODEL_REPRESSILATOR)
    Q_ = ureg.Quantity

    scan0d = ScanSim(
        simulation=TimecourseSim([
            Timecourse(start=0, end=100, steps=100, changes={}),
            Timecourse(start=0, end=60, steps=100, changes={'[X]': Q_(10, "dimensionless")}),
            Timecourse(start=0, end=60, steps=100, changes={'X': Q_(10, "dimensionless")}),
        ]),
        dimensions=[]
    )
    scan0d.normalize(udict=udict, ureg=ureg)
    dfs = simulator.run_scan(scan0d)
    result = Result.from_dfs(scan=scan0d, dfs=dfs)
    print(result)
    return result

def run_parameter_scan1d():
    """Perform a parameter scan"""
    simulator = SimulatorSerial(model=MODEL_REPRESSILATOR)
    udict, ureg = Units.get_units_from_sbml(MODEL_REPRESSILATOR)
    Q_ = ureg.Quantity

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
    scan1d.normalize(udict=udict, ureg=ureg)
    result = simulator.run_scan(scan1d)
    print(result)
    return result


def run_parameter_scan1d_distribution():
    """Perform a parameter scan"""
    simulator = SimulatorSerial(model=MODEL_REPRESSILATOR)
    udict, ureg = Units.get_units_from_sbml(MODEL_REPRESSILATOR)
    Q_ = ureg.Quantity

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
    scan1d.normalize(udict=udict, ureg=ureg)
    result = simulator.run_scan(scan1d)
    print(result)
    return result


def run_parameter_scan2d():
    """Perform a parameter scan"""
    simulator = SimulatorSerial(model=MODEL_REPRESSILATOR)
    udict, ureg = Units.get_units_from_sbml(MODEL_REPRESSILATOR)
    Q_ = ureg.Quantity

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
    scan2d.normalize(udict=udict, ureg=ureg)
    result = simulator.run_scan(scan2d)
    print(result)
    return result


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    column = 'PX'
    if False:
        results = run_parameter_scan0d()
        results[column].plot()
        plt.show()

    results = run_parameter_scan1d()
    results[column].plot()
    plt.show()

    results = run_parameter_scan1d_distribution()
    results[column].plot()
    plt.show()

    da = results[column]
    for k in range(results.dims["dim1"]):
        # individual timecourses
        plt.plot(da.coords['time'], da.isel(dim1=k))

    plt.plot(da.coords['time'], da.mean(dim="dim1"), color="black", linewidth=4.0)
    plt.plot(da.coords['time'], da.min(dim="dim1"), color="black", linewidth=2.0)
    plt.plot(da.coords['time'], da.max(dim="dim1"), color="black", linewidth=2.0)

    plt.show()

    # create mean and standard deviation

    if False:
        results = run_parameter_scan2d()
        results[column].plot()
        plt.show()
