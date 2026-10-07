"""The simulation experiment of the neural ODE: the model against the data.

The model `lotka_volterra_neural_ode.xml` has the species `prey` and
`predator` with the rate rules `d prey/dt = prey_param` and
`d predator/dt = predator_param`, which the network sets; it is the model
`create_neural_ode` of `petab_sciml` writes. The data are the first four
seconds of the Lotka-Volterra system with noise, `data.tsv`.

The experiment simulates the model with the network, which the fit compiles
into its `base_path` as `COMPILED_MODEL`: the workers of a parallel fit load
the model from that file.
"""

from pathlib import Path

import pandas as pd

from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitData, FitMapping
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Simulation
from sbmlsim.task import Task

#: the directory of the example, with the model and the data
EXAMPLE_PATH = Path(__file__).parent

#: the model without the network
MODEL_PATH = EXAMPLE_PATH / "lotka_volterra_neural_ode.xml"

#: the model with the network, relative to the `base_path` of the problem
COMPILED_MODEL = "lotka_volterra_neural_ode_sciml.xml"

#: the species, which are observed and whose rates the network gives
SPECIES = ("prey", "predator")

#: end of the training data in seconds, the data after it is validation data
TRAINING_END = 4.0

#: end of the simulation in seconds, the end of the validation data
SIMULATION_END = 6.0


def mapping_id(species: str, validation: bool = False) -> str:
    """Get the id of the fit mapping of a species, of its validation data or not."""
    return f"{species}_validation" if validation else f"{species}_data"


class NeuralODE(SimulationExperiment):
    """The neural ODE simulated over the data of both species."""

    def models(self) -> dict[str, AbstractModel | Path]:
        return {
            "lv": AbstractModel(
                source=COMPILED_MODEL,
                base_path=self.base_path,
                language_type=AbstractModel.LanguageType.SBML,
            )
        }

    def datasets(self) -> dict[str, DataSet]:
        df = pd.read_csv(EXAMPLE_PATH / "data.tsv", sep="\t")
        datasets: dict[str, DataSet] = {}
        for species, rows in df.groupby("observable"):
            for validation in (False, True):
                part = rows[(rows["time"] > TRAINING_END) == validation]
                datasets[f"data_{mapping_id(str(species), validation)}"] = (
                    DataSet.from_df(
                        pd.DataFrame(
                            {
                                "time": part["time"].to_numpy(),
                                "time_unit": "second",
                                "value": part["value"].to_numpy(),
                                "value_unit": "dimensionless",
                            }
                        ),
                        ureg=self.ureg,
                    )
                )
        return datasets

    def simulations(self) -> dict[str, Simulation]:
        return {"sim": Simulation(end=SIMULATION_END, steps=150)}

    def tasks(self) -> dict[str, Task]:
        return {"task_sim": Task(model="lv", simulation="sim")}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            mapping_id(species, validation): FitMapping(
                self,
                reference=FitData(
                    self,
                    dataset=f"data_{mapping_id(species, validation)}",
                    xid="time",
                    yid="value",
                ),
                observable=FitData(self, task="task_sim", xid="time", yid=species),
            )
            for species in SPECIES
            for validation in (False, True)
        }
