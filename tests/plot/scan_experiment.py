"""A dosed one-compartment model over scans, for the figures over scan points."""

import numpy as np

from sbmlsim import Q
from sbmlsim.data import Data
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import PK, Change, Dimension, Observable, Scan, Simulation
from sbmlsim.task import Task
from tests.simulator.models import sbml_pk


def dosed(steps: int | None = 48) -> Simulation:
    return Simulation(
        end=24, steps=steps, changes=[Change(0, {"PODOSE": Q(100, "mg")})]
    )


def doses(n: int = 3) -> Dimension:
    values = [50.0, 100.0, 200.0] if n == 3 else np.linspace(10.0, 120.0, n).tolist()
    return Dimension("dose", values={"PODOSE": Q(values, "mg")})


class ScanFigures(SimulationExperiment):
    """Doses, many doses, doses times elimination rates, draws and a ragged scan."""

    def models(self) -> dict:
        return {"m": AbstractModel(source=sbml_pk())}

    def simulations(self) -> dict:
        return {
            "doses": Scan(dosed(), [doses()]),
            "many": Scan(dosed(), [doses(12)]),
            "grid2": Scan(
                dosed(),
                [doses(), Dimension("rate", values={"ke": np.array([0.1, 0.3])})],
            ),
            "draws": Scan(
                dosed(), [Dimension("draw", values={"ke": np.linspace(0.1, 0.4, 20)})]
            ),
            "dose_draws": Scan(
                dosed(),
                [doses(), Dimension("draw", values={"ke": np.linspace(0.1, 0.4, 8)})],
            ),
            "many2": Scan(
                dosed(),
                [doses(12), Dimension("rate", values={"ke": np.array([0.1, 0.3])})],
            ),
            "grid5": Scan(
                dosed(),
                [doses(), Dimension("rate", values={"ke": np.linspace(0.1, 0.5, 5)})],
            ),
            "ragged": Scan(dosed(steps=None), [doses()]),
        }

    def observables(self) -> dict[str, Observable]:
        return {"pk": PK("pk", "[C]", dose="PODOSE", route="oral")}

    def tasks(self) -> dict:
        return {
            f"task_{key}": Task(model="m", simulation=key) for key in self._simulations
        }

    def data(self) -> dict:
        self.add_selections_data(["time", "[C]"])
        return {
            "cmax_doses": Data("pk.cmax", task="task_doses"),
            "cmax_many": Data("pk.cmax", task="task_many"),
        }
