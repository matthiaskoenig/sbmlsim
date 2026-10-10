"""Synthetic studies of a dosed one-compartment model for scalar fit mappings.

The studies compare the cmax of a PK observable and the concentration of the
model `tests.simulator.models.sbml_pk` with data which is computed once from
the model at its own parameters, so a fit at `TRUE_V` has the residuals 0. It
is a module of its own, because the workers of a parallel fit import the
experiment class of a problem.
"""

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from sbmlsim import Q
from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitData, FitMapping, FitSettings
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import PK, Change, Dimension, Observable, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from sbmlsim.units import UnitRegistry
from tests.simulator.models import sbml_pk

#: the volume of the model, the value the fits recover
TRUE_V = 10.0
DOSES = [50.0, 100.0, 200.0]
OBSERVABLES: dict[str, Observable] = {
    "pk": PK("pk", "[C]", dose="PODOSE", route="oral")
}


def dosed() -> Simulation:
    return Simulation(end=48, steps=480, changes=[Change(0, {"PODOSE": Q(100, "mg")})])


def dose_scan() -> Scan:
    return Scan(dosed(), [Dimension("dose", values={"PODOSE": Q(DOSES, "mg")})])


def _truth() -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """The cmax of every dose, the time and [C] of 100 mg, and the unit of cmax.

    The integrator settings are the ones of a fit with the default settings,
    so the simulation of a fit at `TRUE_V` is the one of the data.
    """
    settings = FitSettings()
    simulator = Simulator(
        absolute_tolerance=settings.absolute_tolerance,
        relative_tolerance=settings.relative_tolerance,
        variable_step_size=settings.variable_step_size,
        initial_time_step=settings.initial_time_step,
    )
    res = simulator.run(
        sbml_pk(), dose_scan(), list(OBSERVABLES.values()), keep=["pk.cmax", "[C]"]
    )
    return (
        res["pk.cmax"].values,
        res["time"].values,
        res["[C]"].values[1],
        res.units["pk.cmax"],
    )


CMAX, TIME, CONC, CMAX_UNIT = _truth()


def _dataset(df: pd.DataFrame, units: dict[str, str], ureg: UnitRegistry) -> DataSet:
    return DataSet.from_df(df, udict=units, ureg=ureg)


class PKStudy(SimulationExperiment):
    """The model, the observables and the tasks every study shares."""

    def models(self) -> dict[str, AbstractModel | Path]:
        return {"m": AbstractModel(source=sbml_pk())}

    def simulations(self) -> dict[str, Simulation | Scan]:
        return {"single": dosed(), "doses": dose_scan()}

    def observables(self) -> dict[str, Observable]:
        return dict(OBSERVABLES)

    def tasks(self) -> dict[str, Task]:
        return {
            "task_single": Task(model="m", simulation="single"),
            "task_doses": Task(model="m", simulation="doses"),
        }


class ScalarStudy(PKStudy):
    """The cmax of one group of 100 mg."""

    def datasets(self) -> dict[str, DataSet]:
        df = pd.DataFrame({"cmax": [CMAX[1]], "cmax_sd": [0.05 * CMAX[1]]})
        return {"tab": _dataset(df, {"cmax": CMAX_UNIT}, self.ureg)}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm_cmax": FitMapping(
                self,
                reference=FitData(
                    self, dataset="tab", xid=None, yid="cmax", yid_sd="cmax_sd"
                ),
                observable=FitData(self, task="task_single", xid=None, yid="pk.cmax"),
            )
        }


class RowsStudy(ScalarStudy):
    """Three individuals of 100 mg against the one simulated cmax."""

    def datasets(self) -> dict[str, DataSet]:
        df = pd.DataFrame({"cmax": CMAX[1] * np.array([0.9, 1.0, 1.2])})
        return {"tab": _dataset(df, {"cmax": CMAX_UNIT}, self.ureg)}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm_rows": FitMapping(
                self,
                reference=FitData(self, dataset="tab", xid=None, yid="cmax"),
                observable=FitData(self, task="task_single", xid=None, yid="pk.cmax"),
            )
        }


class IndividualsStudy(RowsStudy):
    """The individuals of `RowsStudy` under the key of the cmax of `ScalarStudy`."""

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {"fm_cmax": super().fit_mappings()["fm_rows"]}


class DoseStudy(PKStudy):
    """The cmax of three doses against the scan over the doses."""

    REF_DOSES: ClassVar[list[float]] = DOSES

    def datasets(self) -> dict[str, DataSet]:
        cmax = np.interp(self.REF_DOSES, DOSES, CMAX)
        df = pd.DataFrame(
            {"dose": self.REF_DOSES, "cmax": cmax, "cmax_sd": 0.05 * cmax}
        )
        return {"tab_doses": _dataset(df, {"dose": "mg", "cmax": CMAX_UNIT}, self.ureg)}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm_dose": FitMapping(
                self,
                reference=FitData(
                    self, dataset="tab_doses", xid="dose", yid="cmax", yid_sd="cmax_sd"
                ),
                observable=FitData(
                    self, task="task_doses", xid="dose.PODOSE", yid="pk.cmax"
                ),
            )
        }


class OneDoseStudy(DoseStudy):
    """The cmax of 100 mg against the one point of 100 mg of the scan."""

    REF_DOSES: ClassVar[list[float]] = [100.0]

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm_one": FitMapping(
                self,
                reference=FitData(
                    self, dataset="tab_doses", xid="dose", yid="cmax", yid_sd="cmax_sd"
                ),
                observable=FitData(
                    self,
                    task="task_doses",
                    xid="dose.PODOSE",
                    yid="pk.cmax",
                    sel={"dose": [1]},
                ),
            )
        }


class OutsideStudy(DoseStudy):
    """A dose of 400 mg, which the scan does not cover."""

    REF_DOSES: ClassVar[list[float]] = [50.0, 400.0]


class MixedStudy(ScalarStudy):
    """The timecourse of [C] and the cmax of one simulation of 100 mg."""

    def datasets(self) -> dict[str, DataSet]:
        sets = super().datasets()
        df = pd.DataFrame({"time": TIME[::40], "C": CONC[::40]})
        sets["tc"] = _dataset(df, {"time": "hr", "C": "mg/l"}, self.ureg)
        return sets

    def fit_mappings(self) -> dict[str, FitMapping]:
        mappings = super().fit_mappings()
        mappings["fm_tc"] = FitMapping(
            self,
            reference=FitData(self, dataset="tc", xid="time", yid="C"),
            observable=FitData(self, task="task_single", xid="time", yid="[C]"),
        )
        return mappings
