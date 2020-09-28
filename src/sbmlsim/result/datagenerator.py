from typing import Callable, Dict

import numpy as np

from sbmlsim.data import DataSet
from sbmlsim.result import XResult
from sbmlsim.simulation import Dimension, ScanSim, Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulator.simulation_ray import SimulatorParallel
from sbmlsim.test import MODEL_MIDAZOLAM, MODEL_REPRESSILATOR


class DataGeneratorFunction:
    def __call__(
        self, xresults: Dict[str, XResult], dsets: Dict[str, DataSet] = None
    ) -> Dict[str, XResult]:
        raise NotImplementedError


class DataGeneratorIndexingFunction(DataGeneratorFunction):
    def __init__(self, index: int, dimension: str = "_time"):
        self.index = index
        self.dimension = dimension

    def __call__(self, xresults: Dict[str, XResult], dsets=None) -> Dict[str, XResult]:
        """Reduces based on '_time' dimension with given index.

        :param xresults:
        :return:
        """
        results = {}
        for key, xres in xresults.items():
            xds_new = xres.xds.isel({self.dimension: self.index})
            xres_new = XResult(xdataset=xds_new, udict=xres.udict, ureg=xres.ureg)
            results[key] = xres_new

        return results


class DataGenerator:
    """
    DataGenerators allow to postprocess existing data. This can be a variety of operations.

    - Slicing: reduce the dimension of a given XResult, by slicing a subset on a given dimension
    - Cumulative processing: mean, sd, ...
    - Complex processing, such as pharmacokinetics calculation.
    """

    def __init__(
        self,
        f: DataGeneratorFunction,
        xresults: Dict[str, XResult],
        dsets: Dict[str, DataSet] = None,
    ):
        self.xresults = xresults
        self.dsets = dsets
        self.f = f

    def process(self) -> XResult:
        """Process the data generator. """
        return self.f(xresults=self.xresults, dsets=self.dsets)


def run_scan() -> XResult:
    simulator = SimulatorParallel(model=MODEL_MIDAZOLAM)
    # simulator = SimulatorSerial(model=MODEL_MIDAZOLAM)
    Q_ = simulator.ureg.Quantity

    scan = ScanSim(
        simulation=TimecourseSim(
            [
                Timecourse(start=0, end=1000, steps=200, changes={}),
            ]
        ),
        dimensions=[
            Dimension(
                "dim_dose",
                changes={
                    "IVDOSE_mid": Q_(np.linspace(start=0, stop=100, num=10), "mg"),
                },
            ),
            Dimension(
                "dim_bw",
                changes={
                    "BW": Q_(np.linspace(start=65, stop=100, num=5), "kg"),
                    # 'BW': Q_(np.random.normal(loc=75.0, scale=5.0, size=20), "kg"),
                },
            ),
        ],
    )
    simulator.set_timecourse_selections(
        selections=["time", "IVDOSE_mid", "BW", "[Cve_mid]"]
    )
    return simulator.run_scan(scan)


if __name__ == "__main__":

    xres = run_scan()
    # print(xres)

    dgen_first = DataGeneratorIndexingFunction(dimension="_time", index=0)
    dgen_last = DataGeneratorIndexingFunction(dimension="_time", index=-1)

    res_first = dgen_first(xresults={"res1": xres})
    res_last = dgen_last(xresults={"res1": xres})
    xres1 = res_first["res1"]
    print(xres1)

    from matplotlib import pyplot as plt

    # plt.plot(res_first['res1']["IVDOSE_mid"], res_last['res1']["Cve_mid"], 'o')
    # plt.show()
    x = (res_first["res1"]["IVDOSE_mid"]).mean(dim="dim_bw")
    y = (res_last["res1"]["[Cve_mid]"]).mean(dim="dim_bw")
    ystd = (res_last["res1"]["[Cve_mid]"]).std(dim="dim_bw")

    print("x", x)
    print("y", y)
    print("ystd", ystd)

    plt.errorbar(x=x, y=y, yerr=ystd, marker="o", linestyle="", color="black")
    plt.show()
