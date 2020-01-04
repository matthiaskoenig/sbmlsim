"""
Example shows basic model simulations and plotting.
"""
import numpy as np

from sbmlsim.parametrization import ChangeSet
from sbmlsim.timecourse import Timecourse, TimecourseSim, TimecourseScan
from sbmlsim.simulation_ray import SimulatorParallel
from sbmlsim.simulation_serial import SimulatorSerial
from sbmlsim.result import Result
from sbmlsim.tests.constants import MODEL_REPRESSILATOR
from sbmlsim.units import Units

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
        simulator = SimulatorSerial(path=MODEL_REPRESSILATOR)

    udict, ureg = Units.get_units_from_sbml(MODEL_REPRESSILATOR)
    Q_ = ureg.Quantity

    print("-" * 80)
    print("Parameter scan")
    print("-" * 80)
    tcscan = TimecourseScan(
        tcsim=TimecourseSim([
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


def run_parameter_scan_old(parallel=False):
    """Perform a parameter scan"""

    # [2] value scan
    scan_changeset = ChangeSet.scan_changeset('n', values=np.linspace(start=2, stop=10, num=8))
    tcsims = ensemble(
        sim=TimecourseSim([
            Timecourse(start=0, end=100, steps=100, changes={}),
            Timecourse(start=0, end=60, steps=100, changes={'[X]': 10}),
            Timecourse(start=0, end=60, steps=100, changes={'X': 10}),
        ]),
        changeset=scan_changeset
    )
    print(tcsims[0])

    if parallel:
        simulator = SimulatorParallel(path=MODEL_REPRESSILATOR)
        results = simulator.timecourses(tcsims)
        assert isinstance(results, Result)

    else:
        simulator = SimulatorSerial(path=MODEL_REPRESSILATOR)
        results = simulator.timecourses(tcsims)
        assert isinstance(results, Result)

    return results


if __name__ == "__main__":
    run_parameter_scan(parallel=False)

    # run_parameter_scan_old(parallel=False)
    # run_parameter_scan_old(parallel=True)
