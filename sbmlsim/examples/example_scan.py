"""
Example shows basic model simulations and plotting.
"""
import numpy as np

from sbmlsim.simulation import Timecourse, TimecourseSim, ParameterScan, ScanDimension
from sbmlsim.simulator import SimulatorSerial, SimulatorParallel
from sbmlsim.units import Units

from sbmlsim.tests.constants import MODEL_REPRESSILATOR


def run_parameter_scan():
    """Perform a parameter scan"""
    simulator = SimulatorSerial(model=MODEL_REPRESSILATOR)
    udict, ureg = Units.get_units_from_sbml(MODEL_REPRESSILATOR)
    Q_ = ureg.Quantity

    print("-" * 80)
    print("Parameter scan")
    print("-" * 80)
    scan2d = ParameterScan(
        simulation=TimecourseSim([
            Timecourse(start=0, end=100, steps=100, changes={}),
            Timecourse(start=0, end=60, steps=100, changes={'[X]': Q_(10, "dimensionless")}),
            Timecourse(start=0, end=60, steps=100, changes={'X': Q_(10, "dimensionless")}),
        ]),
        dimensions=[
            ScanDimension("dim1", index=range(8), changes={
                'n': Q_(np.linspace(start=2, stop=10, num=8), "dimensionless"),
            }),
            ScanDimension("dim2", index=range(4), changes={
                'Y': Q_(np.linspace(start=10, stop=20, num=4), "dimensionless"),
            }),
        ]
    )
    scan2d.normalize(udict=udict, ureg=ureg)
    results = simulator.run_scan(scan2d)
    return results


if __name__ == "__main__":
    results = run_parameter_scan()
    print(results)

