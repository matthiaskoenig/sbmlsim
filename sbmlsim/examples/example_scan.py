"""
Example shows basic model simulations and plotting.
"""
import numpy as np

from sbmlsim.simulation import Timecourse, TimecourseSim, ParameterScan
from sbmlsim.simulator import SimulatorSerial, SimulatorParallel
from sbmlsim.units import Units

from sbmlsim.tests.constants import MODEL_REPRESSILATOR


def run_parameter_scan(parallel=False):
    """Perform a parameter scan"""

    '''
    # simple timecourse simulation
    print("-" * 80)
    print("Simple timecourse")
    print("-" * 80)
    tcsim = TimecourseSim([
            Timecourse(start=0, end=100, steps=100, changes={'n': 2}),
            Timecourse(start=0, end=60, steps=100, changes={'[X]': 10}),
            Timecourse(start=0, end=60, steps=100, changes={'X': 10}),
    ]),
    if parallel:
        raise NotImplementedError
    else:
        simulator = SimulatorSerial(path=MODEL_REPRESSILATOR)
        results = simulator.timecourses(tcsim)
        print(results)
    '''
    if parallel:
        raise NotImplementedError
    else:
        simulator = SimulatorSerial(model=MODEL_REPRESSILATOR)

    udict, ureg = Units.get_units_from_sbml(MODEL_REPRESSILATOR)
    Q_ = ureg.Quantity

    print("-" * 80)
    print("Parameter scan")
    print("-" * 80)
    tcscan = ParameterScan(
        simulation=TimecourseSim([
            Timecourse(start=0, end=100, steps=100, changes={}),
            Timecourse(start=0, end=60, steps=100, changes={'[X]': Q_(10, "dimensionless")}),
            Timecourse(start=0, end=60, steps=100, changes={'X': Q_(10, "dimensionless")}),
        ]),
        scan={
            'n': Q_(np.linspace(start=2, stop=10, num=8), "dimensionless"),
            'Y': Q_(np.linspace(start=10, stop=20, num=4), "dimensionless"),
        }
    )
    tcscan.normalize(udict=udict, ureg=ureg)
    results = simulator.scan(tcscan)
    print(results)
    print(results.keys)
    print(results.vecs)
    print(results.indices)

    return results


if __name__ == "__main__":
    run_parameter_scan(parallel=False)

