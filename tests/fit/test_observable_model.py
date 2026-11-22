"""An observable which is a formula of selections with placeholders."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbmlsim.data import DataSet
from sbmlsim.engine import Engine
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit.fit_core import fit_request, gaps_of, loaded_model
from sbmlsim.fit.objects import (
    FitData,
    FitMapping,
    FitMappingCollection,
    FitParameter,
    ObservableModel,
)
from sbmlsim.fit.observable_sbml import OBSERVABLE_PREFIX
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from tests.engines import require_core_fit
from tests.simulator.models import sbml


def test_symbols_are_the_selections_of_the_formula() -> None:
    """The symbols are the selections the formula reads, without placeholders."""
    observable = ObservableModel(
        "scale * [A] + offset",
        placeholders=("offset",),
        placeholder_values=((1.0,), ("k1 * 2",)),
    )
    assert observable.symbols == ("[A]", "k1", "scale")


def test_evaluate_on_arrays() -> None:
    """The formula is evaluated on the values of its symbols at the data."""
    observable = ObservableModel("scale * [A]")
    values = observable.evaluate(
        {"[A]": np.array([1.0, 2.0]), "scale": np.array([3.0, 3.0])}, size=2
    )
    np.testing.assert_allclose(values, [3.0, 6.0])


def test_evaluate_a_constant() -> None:
    """A formula without symbols has one value per measurement."""
    observable = ObservableModel("2.5")
    np.testing.assert_allclose(observable.evaluate({}, size=3), [2.5, 2.5, 2.5])


def test_placeholders_per_measurement() -> None:
    """Every measurement is evaluated with its own placeholder values."""
    observable = ObservableModel(
        "p1 * A + p2",
        placeholders=("p1", "p2"),
        placeholder_values=((2.0, 1.0), (3.0, "k")),
    )
    values = observable.evaluate(
        {"A": np.array([1.0, 1.0]), "k": np.array([5.0, 7.0])}, size=2
    )
    np.testing.assert_allclose(values, [3.0, 10.0])


def test_placeholders_require_a_value_per_measurement() -> None:
    """A measurement without a value for every placeholder is an error."""
    with pytest.raises(ValueError, match=re.escape("['p1', 'p2']")):
        ObservableModel(
            "p1 * A + p2",
            placeholders=("p1", "p2"),
            placeholder_values=((2.0,),),
        )


def test_placeholders_require_the_values_of_the_data() -> None:
    """The values of the placeholders are the values of the measurements."""
    observable = ObservableModel(
        "p1 * A", placeholders=("p1",), placeholder_values=((2.0,),)
    )
    with pytest.raises(ValueError, match="'1' measurements"):
        observable.evaluate({"A": np.array([1.0, 1.0])}, size=2)


def test_select_measurements() -> None:
    """A selection of the measurements keeps their placeholder values."""
    observable = ObservableModel(
        "p1 * A", placeholders=("p1",), placeholder_values=((1.0,), (2.0,), (3.0,))
    )
    selected = observable.select(np.array([True, False, True]))
    assert selected.placeholder_values == ((1.0,), (3.0,))


def test_invalid_formula() -> None:
    """A formula which is not math is an error which names it."""
    with pytest.raises(ValueError, match=re.escape("'A +'")):
        ObservableModel("A +")


#: the file of the probe model, written by the fixture
MODEL_PATH: dict[str, Path] = {}


class ObservableExperiment(SimulationExperiment):
    """The probe model, observed as `scale * [A] + offset`.

    The scale is a parameter of the model, the offset a placeholder with a
    value per measurement; one of them is the parameter `k1`.
    """

    def models(self) -> dict[str, AbstractModel | Path]:
        return {"m": AbstractModel(source=MODEL_PATH["path"])}

    def datasets(self) -> dict[str, DataSet]:
        df = pd.DataFrame(
            {
                "time": [0.0, 0.5, 1.5, 2.0],
                "time_unit": "s",
                "y": [1.0, np.nan, 0.5, 0.4],
                "y_unit": "dimensionless",
            }
        )
        return {"d": DataSet.from_df(df, ureg=self.ureg)}

    def simulations(self) -> dict[str, Simulation]:
        return {"s": Simulation(end=2)}

    def tasks(self) -> dict[str, Task]:
        return {"t": Task(model="m", simulation="s")}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm": FitMapping(
                self,
                reference=FitData(self, dataset="d", xid="time", yid="y"),
                observable=FitData(self, task="t", xid="time", yid="y_A"),
                observable_model=ObservableModel(
                    "f * [A] + offset",
                    placeholders=("offset",),
                    placeholder_values=((0.0,), (0.5,), (1.0,), ("k1",)),
                ),
            )
        }


@pytest.fixture
def problem(tmp_path: Path) -> OptimizationProblem:
    """Get a problem which fits `f` of the probe model with the observable."""
    MODEL_PATH["path"] = tmp_path / "probe.xml"
    MODEL_PATH["path"].write_text(sbml(), encoding="utf-8")
    problem = OptimizationProblem(
        opid="observable",
        mapping_collections=[
            FitMappingCollection(experiment=ObservableExperiment, mappings=["fm"])
        ],
        fit_parameters=[
            FitParameter(
                pid="f",
                lower_bound=0.1,
                upper_bound=10.0,
                start_value=2.0,
                unit="dimensionless",
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    return problem


def test_fit_evaluates_the_observable_model(problem: OptimizationProblem) -> None:
    """The prediction is the formula on the simulation at the data.

    The measurement at `0.5` has no value and is dropped with its placeholder
    value, the others keep theirs: `0.0`, `1.0` and `k1 = 0.8`.
    """
    simulator = Simulator(n_workers=1)
    model = simulator.load(MODEL_PATH["path"])
    model.set_selections(["time", "[A]"])
    result = simulator.simulate(
        model, Simulation(end=2, times=[0.0, 1.5, 2.0], preinit_changes={"f": 3.0})
    )
    expected = 3.0 * np.asarray(result["[A]"]) + np.array([0.0, 1.0, 0.8])

    predictions = problem.predictions(np.array([3.0]))[0]
    # the fit integrates with the tolerances of its settings
    np.testing.assert_allclose(predictions, expected, rtol=1e-4)


def test_fit_selects_the_symbols_of_the_observable(
    problem: OptimizationProblem,
) -> None:
    """The simulation selects what the observable reads, not its label."""
    assert set(problem.selections[0]) == {"time", "[A]", "f", "k1"}


def test_fit_observable_scales_with_parameter(problem: OptimizationProblem) -> None:
    """The observable reads the value of the fit parameter from the model."""
    offset = np.array([0.0, 1.0, 0.8])
    low = problem.predictions(np.array([1.0]))[0] - offset
    high = problem.predictions(np.array([4.0]))[0] - offset
    np.testing.assert_allclose(high, 4.0 * low)


# the observable in the fit of sbmlsim-core: it is written into the model of
# the fit as a parameter of an assignment rule, see `sbmlsim.fit.observable_sbml`


def test_an_observable_keeps_a_problem_out_of_no_fit_of_the_engine(
    problem: OptimizationProblem,
) -> None:
    """The observable of a mapping is a column of the model of the fit."""
    require_core_fit()
    assert problem.engine is Engine.CORE

    assert gaps_of(problem) == []


def test_a_placeholder_of_a_value_per_measurement_splits_the_curve(
    problem: OptimizationProblem,
) -> None:
    """A curve of the request reads one observable, so one row of the values.

    The three measurements which are kept have the offsets `0.0`, `1.0` and
    `k1`, three rows, so the mapping is three curves of one point each, in the
    order of the points: the residuals of the request are the residuals of the
    problem in their places.
    """
    require_core_fit()

    request = fit_request(problem)

    curves = request["curves"]
    assert len(curves) == 3
    assert [curve["times"] for curve in curves] == [[0.0], [1.5], [2.0]]
    assert [curve["id"] for curve in curves] == ["fm[0:1]", "fm[1:2]", "fm[2:3]"]
    columns = [curve["column"] for curve in curves]
    assert len(set(columns)) == 3
    assert all(column.startswith(OBSERVABLE_PREFIX) for column in columns)
    # the condition selects every one of them
    selections = request["conditions"][0]["request"]["selections"]
    assert set(columns) <= set(selections)


def test_the_residuals_of_the_engine_are_the_ones_of_the_problem(
    problem: OptimizationProblem,
) -> None:
    """The engine computes the residuals of the problem, point for point."""
    require_core_fit()
    core = loaded_model(problem).fit_problem(fit_request(problem), threads=1)

    for value in (2.0, 3.0, 0.5):
        x = problem.to_scale(np.array([value]))
        expected = np.asarray(problem.residuals(x), dtype=float)

        residuals = np.asarray(core.residuals(x), dtype=float)

        assert residuals.shape == expected.shape
        np.testing.assert_allclose(residuals, expected, rtol=1e-9, atol=1e-12)
