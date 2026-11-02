"""The problem of the PEtab v2 tutorial as a simulation experiment.

The [tutorial of PEtab v2](https://petab.readthedocs.io/en/latest/v2/tutorial/tutorial.html)
builds the parameter estimation problem of Boehm et al. 2014 (the dimerization
of STAT5A and STAT5B after a stimulation with erythropoietin) table by table.
`BoehmTutorial` is the same problem defined in python: the experiment
`epo_stimulation` applies the condition `epo_bolus`, `Epo_concentration =
1.25e-7`, at the time 0, three observables with normal noise whose standard
deviations are estimated, and the measurements from 0 to 240 minutes.
`to_petab` writes it as the tables of the tutorial, see `TUTORIAL`.

The model is the one of the benchmark collection with the concentration of
erythropoietin as the parameter `Epo_concentration`, which the tutorial
modified the model of BioModels (BIOMD0000000591) for; the data is the one of
the collection.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import (
    FitData,
    FitMapping,
    FitMappingCollection,
    FitParameter,
    FitSettings,
)
from sbmlsim.fit.objects import NoiseModel, ObservableModel
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import DataAtChangeType, ObjectiveType
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Scan, Simulation
from sbmlsim.task import Task

#: the directory of the files of the tutorial
DIRECTORY = Path(__file__).parent

#: the model, the one of the benchmark collection with `Epo_concentration`
MODEL_PATH = DIRECTORY / "model_Boehm_JProteomeRes2014.xml"

#: the measurements of the tutorial, the data of Boehm et al. 2014
MEASUREMENTS_PATH = DIRECTORY / "measurement_data.tsv"

#: the observables of the tutorial: the relative phosphorylation of STAT5A and
#: STAT5B and the relative abundance of STAT5A in percent, formulas of the
#: concentrations of the species
OBSERVABLES: dict[str, str] = {
    "pSTAT5A_rel": "100 * (2 * [pApA] + [pApB]) / (2 * [pApA] + [pApB] + [STAT5A])",
    "pSTAT5B_rel": "100 * (2 * [pBpB] + [pApB]) / (2 * [pBpB] + [pApB] + [STAT5B])",
    "rSTAT5A_rel": (
        "100 * ([STAT5A] + [pApB] + 2 * [pApA]) / "
        "(2 * [pApB] + 2 * [pApA] + [STAT5A] + [STAT5B] + 2 * [pBpB])"
    ),
}

#: the parameters the tutorial estimates, all within `1e-5` and `1e5`
KINETIC_PARAMETERS: tuple[str, ...] = (
    "Epo_degradation_BaF3",
    "k_exp_hetero",
    "k_exp_homo",
    "k_imp_hetero",
    "k_imp_homo",
    "k_phos",
)

#: the standard deviation of the noise of every observable, estimated
NOISE_PARAMETERS: dict[str, str] = {sid: f"sd_{sid}" for sid in OBSERVABLES}

#: the concentration of erythropoietin of the condition `epo_bolus`
EPO_CONCENTRATION = 1.25e-7

#: the tables of the tutorial which the export is compared with, see
#: `examples.petab.tutorial`
TUTORIAL: dict[str, Any] = {
    "experiments": [("epo_stimulation", 0.0, "Epo_concentration", EPO_CONCENTRATION)],
    "observables": {
        "pSTAT5A_rel": "100*(2*pApA + pApB) / (2*pApA + pApB + STAT5A)",
        "pSTAT5B_rel": "100*(2*pBpB + pApB) / (2*pBpB + pApB + STAT5B)",
        "rSTAT5A_rel": (
            "100*(STAT5A + pApB + 2*pApA) / "
            "(2 * pApB + 2* pApA + STAT5A + STAT5B + 2*pBpB)"
        ),
    },
    "noise_distribution": "normal",
    "estimated": (*KINETIC_PARAMETERS, *NOISE_PARAMETERS.values()),
    "bounds": (1e-5, 1e5),
    "fixed": {"ratio": 0.693},
    "measurements": 48,
}


class BoehmTutorial(SimulationExperiment):
    """The problem of the PEtab v2 tutorial, Boehm et al. 2014."""

    def datasets(self) -> dict[str, DataSet]:
        """Get the measurements of the observables, one column each."""
        measurements = pd.read_csv(MEASUREMENTS_PATH, sep="\t")
        wide = measurements.pivot(
            index="time", columns="observableId", values="measurement"
        ).reset_index()
        units = {"time": "min", **dict.fromkeys(OBSERVABLES, "dimensionless")}
        return {"measurement_data": DataSet.from_df(wide, udict=units, ureg=self.ureg)}

    def models(self) -> dict[str, AbstractModel | Path]:
        """Get the model with the standard deviations of the noise as parameters."""
        return {
            "boehm": AbstractModel(
                source=MODEL_PATH,
                parameters=dict.fromkeys(NOISE_PARAMETERS.values(), 1.0),
            )
        }

    def simulations(self) -> dict[str, Simulation | Scan]:
        """Get the experiment `epo_stimulation`, the condition `epo_bolus` at 0."""
        return {
            "epo_stimulation": Simulation(
                end=240,
                time_unit="min",
                preinit_changes={"Epo_concentration": EPO_CONCENTRATION},
            )
        }

    def tasks(self) -> dict[str, Task]:
        """Get the task of the model and the experiment."""
        return {
            "task_epo_stimulation": Task(model="boehm", simulation="epo_stimulation")
        }

    def fit_mappings(self) -> dict[str, FitMapping]:
        """Get the mapping of every observable with its normal noise.

        The noise formula is the placeholder of the tutorial, whose value at
        every measurement is the standard deviation of the observable.
        """
        size = len(self._datasets["measurement_data"]) if self._datasets else 16
        return {
            sid: FitMapping(
                self,
                reference=FitData(
                    self, dataset="measurement_data", xid="time", yid=sid
                ),
                observable=FitData(
                    self, task="task_epo_stimulation", xid="time", yid=sid
                ),
                observable_model=ObservableModel(formula=formula),
                noise=NoiseModel(
                    formula=f"{sid}_sigma",
                    placeholders=(f"{sid}_sigma",),
                    placeholder_values=((NOISE_PARAMETERS[sid],),) * size,
                ),
            )
            for sid, formula in OBSERVABLES.items()
        }


#: the settings of the problem: the maximum likelihood estimate, with the
#: measurement at the time of the change at 0 after it, as PEtab defines it
SETTINGS = FitSettings(
    objective=ObjectiveType.LIKELIHOOD,
    data_at_change=DataAtChangeType.AFTER,
)


def tutorial_problem(base_path: Path) -> OptimizationProblem:
    """Get the optimization problem of the tutorial.

    Args:
        base_path: the directory of the results of the experiment.

    Returns:
        The problem, with the nine parameters the tutorial estimates.
    """
    lower, upper = TUTORIAL["bounds"]
    parameters = [
        FitParameter(pid=pid, lower_bound=lower, upper_bound=upper)
        for pid in TUTORIAL["estimated"]
    ]
    collections = [
        FitMappingCollection(experiment=BoehmTutorial, mappings=list(OBSERVABLES))
    ]
    return OptimizationProblem(
        opid="boehm_tutorial",
        mapping_collections=collections,
        fit_parameters=parameters,
        base_path=base_path,
        data_path=DIRECTORY,
    )
