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
from sbmlsim.simulation import AbstractSim, Timecourse, TimecourseSim
from sbmlsim.task import Task

#: the directory of the example, with the model and the data
EXAMPLE_PATH = Path(__file__).parent

#: the model without the network
MODEL_PATH = EXAMPLE_PATH / "lotka_volterra_neural_ode.xml"

#: the model with the network, relative to the `base_path` of the problem
COMPILED_MODEL = "lotka_volterra_neural_ode_sciml.xml"

#: the species, which are observed and whose rates the network gives
SPECIES = ("prey", "predator")


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
        return {
            str(species): DataSet.from_df(
                pd.DataFrame(
                    {
                        "time": rows["time"].to_numpy(),
                        "time_unit": "second",
                        "value": rows["value"].to_numpy(),
                        "value_unit": "dimensionless",
                    }
                ),
                ureg=self.ureg,
            )
            for species, rows in df.groupby("observable")
        }

    def simulations(self) -> dict[str, AbstractSim]:
        return {"sim": TimecourseSim([Timecourse(start=0.0, end=4.0, steps=100)])}

    def tasks(self) -> dict[str, Task]:
        return {"task_sim": Task(model="lv", simulation="sim")}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            f"{species}_data": FitMapping(
                self,
                reference=FitData(self, dataset=species, xid="time", yid="value"),
                observable=FitData(self, task="task_sim", xid="time", yid=species),
            )
            for species in SPECIES
        }
