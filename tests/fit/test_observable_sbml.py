"""The observables of a fit as entities of the model the engine simulates.

A fit mapping which observes a formula of the selections is no column of a
simulation, so the engine could not fit it. The formula is written into the
model of the fit as a global parameter of an assignment rule
(`sbmlsim.fit.observable_sbml`), so the engine reads it as it reads the math
of the model and nothing of the math is evaluated twice.

What these tests check is that the parameter of the model carries the value
the observable of the problem computes, **bitwise**, for every form a
selection of roadrunner takes: a species in the math of SBML is its
concentration unless it has `hasOnlySubstanceUnits`, where it is its amount,
while the selection `S` is the amount of a species and `[S]` its
concentration, so the two differ by the size of the compartment of the
species, each the other way round. The probe model has a concentration
species (`A`), a substance-only species (`X`), a compartment of a size which
is not 1 (`C = 2`) and parameters, so one model covers every row of that
table.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sbmlsim.engine import Engine
from sbmlsim.fit.objects import ObservableModel
from sbmlsim.fit.observable_sbml import (
    HELPER_PREFIX,
    OBSERVABLE_PREFIX,
    Entities,
    add_observable,
    observable_id,
    observable_model,
    with_observables,
)
from sbmlsim.model import AbstractModel
from sbmlsim.model.model_sbml import load_model
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
from tests.engines import require
from tests.simulator.models import sbml

#: the formulas of a selection of every kind: the concentration and the
#: amount of a concentration species, of a substance-only species, a
#: parameter, a compartment and a parameter of an assignment rule
FORMULAS: list[str] = [
    "[A] * 2 + 1",
    "A * 2 + 1",
    "X * 2 + 1",
    "[X] * 2 + 1",
    "k1 * 2 + 1",
    "C * 2 + 1",
    "kk * 2 + 1",
    "[A] / C + X * k1",
]


@pytest.fixture
def model(tmp_path: Path) -> Any:
    """Get the probe model loaded in the engine core."""
    require(Engine.CORE)
    path = tmp_path / "probe.xml"
    path.write_text(sbml(), encoding="utf-8")
    return load_model(str(path), Engine.CORE, reload=True)


def observed(model: Any, entities: Entities, names: list[str]) -> dict[str, Any]:
    """Simulate the model of a fit and get its columns by name."""
    derived = observable_model(model, entities)
    selections = ["time", *names, "[A]", "A", "X", "[X]", "k1", "C", "kk"]
    derived.set_selections(selections)
    result = Simulator(n_workers=1, engine=Engine.CORE).simulate(
        derived, Simulation(end=2, steps=10)
    )
    return {name: np.asarray(result[name]) for name in selections}


@pytest.mark.parametrize("formula", FORMULAS)
def test_the_parameter_of_an_observable_is_the_observable_of_the_problem(
    model: Any, formula: str
) -> None:
    """The column of the model is what the observable computes, bitwise."""
    observable = ObservableModel(formula)
    name = observable_id("fm")
    entities = Entities()
    add_observable(name, observable, {}, model.symbols, entities)

    columns = observed(model, entities, [name])

    expected = observable.evaluate(
        {symbol: columns[symbol] for symbol in observable.symbols},
        size=columns[name].size,
    )
    np.testing.assert_array_equal(columns[name], expected)


def test_a_placeholder_is_a_parameter_of_its_value(model: Any) -> None:
    """A placeholder of a number is a constant parameter, one of a formula a rule."""
    number = ObservableModel("[A] * scale", placeholders=("scale",))
    formula = ObservableModel("[A] * scale", placeholders=("scale",))
    entities = Entities()
    add_observable(observable_id("n"), number, {"scale": 3.0}, model.symbols, entities)
    add_observable(
        observable_id("f"), formula, {"scale": "k1"}, model.symbols, entities
    )

    assert list(entities.constants.values()) == [3.0]
    assert len(entities.rules) == 1
    columns = observed(model, entities, [observable_id("n"), observable_id("f")])

    np.testing.assert_array_equal(columns[observable_id("n")], columns["[A]"] * 3.0)
    np.testing.assert_array_equal(
        columns[observable_id("f")], columns["[A]"] * columns["k1"]
    )


def test_the_model_of_a_fit_keeps_the_parameters_the_problem_adds(
    tmp_path: Path,
) -> None:
    """An observable which reads an added parameter compiles.

    A parameter of a fit which PEtab declares and the model does not have is
    added to the model (`AbstractModel(parameters=...)`), and `SBMLModel.sbml`
    is the SBML before them, so the model of the fit is loaded from the
    description of the model and not from its SBML alone.
    """
    require(Engine.CORE)
    path = tmp_path / "probe.xml"
    path.write_text(sbml(), encoding="utf-8")
    model = load_model(
        AbstractModel(source=str(path), parameters={"added": 4.0}),
        Engine.CORE,
        reload=True,
    )
    observable = ObservableModel("[A] * added")
    name = observable_id("fm")
    entities = Entities()
    add_observable(name, observable, {}, model.symbols, entities)

    columns = observed(model, entities, [name, "added"])

    np.testing.assert_array_equal(columns[name], columns["[A]"] * 4.0)


def test_an_observable_which_reads_a_species_the_model_has_not_is_refused(
    model: Any,
) -> None:
    """A concentration of no species of the model is no entity of the model."""
    with pytest.raises(ValueError, match="no species of the model"):
        add_observable(
            observable_id("fm"),
            ObservableModel("[nothing] * 2"),
            {},
            model.symbols,
            Entities(),
        )


def test_a_placeholder_without_a_value_is_refused(model: Any) -> None:
    """Every placeholder of an observable has a value in the model of the fit."""
    with pytest.raises(ValueError, match="without a value"):
        add_observable(
            observable_id("fm"),
            ObservableModel("[A] * scale", placeholders=("scale",)),
            {},
            model.symbols,
            Entities(),
        )


def test_an_id_of_an_entity_of_the_model_is_refused(model: Any) -> None:
    """An observable never replaces an entity of the model."""
    entities = Entities(constants={"k1": 1.0})
    with pytest.raises(ValueError, match="is an entity of the model"):
        with_observables(model.sbml, entities)


def test_a_model_without_observables_is_the_model_itself(model: Any) -> None:
    """Nothing is read, written or loaded for a problem without an observable."""
    assert with_observables(model.sbml, Entities()) == model.sbml
    assert observable_model(model, Entities()) is model


def test_the_names_of_the_entities_are_of_the_prefixes(model: Any) -> None:
    """Every entity the module adds carries a prefix, so none is an id of a model."""
    entities = Entities()
    add_observable(
        observable_id("fm"),
        ObservableModel("X * scale", placeholders=("scale",)),
        {"scale": 2.0},
        model.symbols,
        entities,
    )
    names = [*entities.constants, *entities.rules, *entities.observables]

    assert names
    assert all(name.startswith((OBSERVABLE_PREFIX, HELPER_PREFIX)) for name in names), (
        names
    )
