from typing import Dict, Callable

import numpy as np
from sbmlsim.data import DataSet
from sbmlsim.result import XResult
from sbmlsim.simulation import ScanSim, TimecourseSim, Timecourse, Dimension
from sbmlsim.simulator.simulation_ray import SimulatorParallel
from sbmlsim.tests import MODEL_MIDAZOLAM, MODEL_REPRESSILATOR


class DataGenerator:
    """
    DataGenerators allow to postprocess existing data. This can be a variety of operations.

    - Slicing: reduce the dimension of a given XResult, by slicing a subset on a given dimension
    - Cumulative processing: mean, sd, ...
    - Complex processing, such as pharmacokinetics calculation.
    """
    def __init__(self, f: Callable, xresults: Dict[str, XResult], dsets: Dict[str, DataSet]):
        self.xresults = xresults
        self.dsets = dsets
        self.f = f

    def process(self) -> XResult:
        return self.f(self.xresults)


def run_scan() -> XResult:
    simulator = SimulatorParallel(model=MODEL_MIDAZOLAM)
    Q_ = simulator.ureg.Quantity

    scan = ScanSim(
        simulation=TimecourseSim([
            Timecourse(start=0, end=1000, steps=200, changes={
            }),
        ]),
        dimensions=[
            Dimension("dim1", changes={
                'IVDOSE_mid': Q_(np.linspace(start=0, stop=100, num=10), "mg"),
            })
        ]
    )
    return simulator.run_scan(scan)


if __name__ == "__main__":

    xres = run_scan()
    print(xres)

    def f_last_timepoint(xresults: Dict[str, XResult]):
        """Get data at last timepoint

        :param xresults:
        :return:
        """
        results = {}
        for key, xres in xresults.items():
            results[key] = xres.xds[]


    dgen = DataGenerator(xresults={"res1": xres}, f=)