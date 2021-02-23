"""Example for DataGenerator functionality."""
import numpy as np

from sbmlsim.result import XResult
from sbmlsim.result.datagenerator import DataGeneratorIndexingFunction
from sbmlsim.simulation import Dimension, ScanSim, Timecourse, TimecourseSim
from sbmlsim.simulator.simulation_ray import SimulatorParallel
from sbmlsim.test import MODEL_MIDAZOLAM


def example_scan() -> XResult:
    """Run scan and return results."""
    simulator = SimulatorParallel(model=MODEL_MIDAZOLAM)
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


def example() -> None:
    """Run example for scan functionality."""
    xres = example_scan()
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


if __name__ == "__main__":
    example()
