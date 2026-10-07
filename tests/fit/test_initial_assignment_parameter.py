"""A fit parameter which feeds an initial assignment reaches it."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit.objects import FitData, FitMapping, FitMappingCollection, FitParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Change, Simulation
from sbmlsim.task import Task
from tests.simulator.models import sbml

#: the file of the probe model, written by the fixture
MODEL_PATH: dict[str, Path] = {}


class IAExperiment(SimulationExperiment):
    """The probe model, whose species B has the initial assignment B = b0."""

    def models(self) -> dict[str, AbstractModel | Path]:
        return {"m": AbstractModel(source=MODEL_PATH["path"])}

    def datasets(self) -> dict[str, DataSet]:
        df = pd.DataFrame(
            {
                "time": [0.0, 0.5, 1.5],
                "time_unit": "s",
                "B": [0.0, 0.2, 0.5],
                "B_unit": "dimensionless",
            }
        )
        return {"d": DataSet.from_df(df, ureg=self.ureg)}

    def simulations(self) -> dict[str, Simulation]:
        return {"s": Simulation(end=2, changes=[Change(1, {"X": 0.0})])}

    def tasks(self) -> dict[str, Task]:
        return {"t": Task(model="m", simulation="s")}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm": FitMapping(
                self,
                reference=FitData(self, dataset="d", xid="time", yid="B"),
                observable=FitData(self, task="t", xid="time", yid="[B]"),
            )
        }


@pytest.fixture
def problem(tmp_path: Path) -> OptimizationProblem:
    """Get a problem which fits b0 of the probe model."""
    MODEL_PATH["path"] = tmp_path / "probe.xml"
    MODEL_PATH["path"].write_text(sbml())
    problem = OptimizationProblem(
        opid="ia",
        mapping_collections=[
            FitMappingCollection(experiment=IAExperiment, mappings=["fm"])
        ],
        fit_parameters=[
            FitParameter(
                pid="b0",
                lower_bound=0.0,
                upper_bound=2.0,
                start_value=1.0,
                unit="dimensionless",
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    return problem


def test_fit_parameter_reaches_initial_assignment(
    problem: OptimizationProblem,
) -> None:
    """The initial value of B follows the value of b0 the fit sets."""
    predictions = problem.predictions(np.array([0.0]))
    assert predictions[0][0] == pytest.approx(0.0)
    predictions = problem.predictions(np.array([1.5]))
    assert predictions[0][0] == pytest.approx(1.5)


def test_fit_simulates_at_the_data(problem: OptimizationProblem) -> None:
    """The plan of the fit outputs the times of the data, no interpolation."""
    assert problem.plans[0].times == (0.0, 0.5, 1.5)


class ShiftedExperiment(IAExperiment):
    """The data of the probe model on a shifted time axis."""

    def datasets(self) -> dict[str, DataSet]:
        df = pd.DataFrame(
            {
                "time": [-72.3, 0.5, 1.02],
                "time_unit": "s",
                "B": [0.0, 0.2, 0.5],
                "B_unit": "dimensionless",
            }
        )
        return {"d": DataSet.from_df(df, ureg=self.ureg)}

    def simulations(self) -> dict[str, Simulation]:
        return {"s": Simulation(end=73.32, time_shift=-72.3)}


def test_fit_with_a_time_shift(tmp_path: Path) -> None:
    """The data of a shifted simulation is found at its times, without rounding."""
    MODEL_PATH["path"] = tmp_path / "probe.xml"
    MODEL_PATH["path"].write_text(sbml())
    problem = OptimizationProblem(
        opid="shifted",
        mapping_collections=[
            FitMappingCollection(experiment=ShiftedExperiment, mappings=["fm"])
        ],
        fit_parameters=[
            FitParameter(
                pid="b0",
                lower_bound=0.0,
                upper_bound=2.0,
                start_value=1.0,
                unit="dimensionless",
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    predictions = problem.predictions(np.array([0.5]))
    assert predictions[0][0] == pytest.approx(0.5)
    assert predictions[0].shape == (3,)
