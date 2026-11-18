"""The model of an SBML model in the engine core."""

import logging
from typing import Any

import pytest

pytest.importorskip("sbmlsim_core")

from sbmlsim.engine import Engine, UnsupportedModelError, import_core
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel, load_model, model_core
from sbmlsim.model.model_core import (
    COMPILED,
    DEFAULT_SETTINGS,
    INTEGRATOR,
    CoreSBMLModel,
)
from sbmlsim.model.symbols import TargetKind
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.resources import HCTZ_SBML, REPRESSILATOR_SBML
from sbmlsim.simulator.plan import Assignment
from tests.simulator.models import ALGEBRAIC

#: a model whose assignment rule reads the delay csymbol, which the compiler
#: of the engine refuses
DELAYED = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="delayed">
    <listOfParameters>
      <parameter id="x" value="1" constant="false"/>
      <parameter id="y" value="2" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="x">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
      </rateRule>
      <assignmentRule variable="y">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply>
            <csymbol encoding="text"
              definitionURL="http://www.sbml.org/sbml/symbols/delay">delay</csymbol>
            <ci>x</ci>
            <cn>1</cn>
          </apply>
        </math>
      </assignmentRule>
    </listOfRules>
  </model>
</sbml>
"""


def per_state(model: CoreSBMLModel) -> list[float]:
    """Get the absolute tolerance of every state of a request of a model."""
    tolerance = model.request_settings()["absolute_tolerance"]
    assert isinstance(tolerance, dict), tolerance
    return [float(value) for value in tolerance["per_state"]]


def tolerance_vector(model: RoadrunnerSBMLModel) -> list[float]:
    """Get the absolute tolerance of every state of CVODE in roadrunner."""
    return [
        float(value)
        for value in model.r_loaded.getIntegrator().getAbsoluteToleranceVector()
    ]


@pytest.fixture
def hctz_core() -> CoreSBMLModel:
    """Get the HCTZ model in the engine core."""
    model = load_model(HCTZ_SBML, Engine.CORE)
    assert isinstance(model, CoreSBMLModel)
    return model


@pytest.fixture
def hctz_roadrunner() -> RoadrunnerSBMLModel:
    """Get the HCTZ model in roadrunner, the reference of the engine."""
    model = load_model(HCTZ_SBML, Engine.ROADRUNNER)
    assert isinstance(model, RoadrunnerSBMLModel)
    return model


def test_a_core_model_loads_and_reports_its_states(hctz_core: CoreSBMLModel) -> None:
    """The model is loaded by the engine and its states are entities of the model."""
    assert hctz_core.engine is Engine.CORE
    assert "time" not in hctz_core.state_ids()
    assert set(hctz_core.state_ids()) <= set(hctz_core.symbols.species) | set(
        hctz_core.symbols.rate_rules
    )


def test_the_states_are_the_states_of_roadrunner(
    hctz_core: CoreSBMLModel, hctz_roadrunner: RoadrunnerSBMLModel
) -> None:
    """The states are the ones of roadrunner in its order, which the tolerances follow."""
    assert hctz_core.state_ids() == hctz_roadrunner.state_ids()


def test_loaded_value_is_the_model_as_loaded(
    hctz_core: CoreSBMLModel, hctz_roadrunner: RoadrunnerSBMLModel
) -> None:
    """A value of the model as loaded is the one of roadrunner."""
    for sid in ("PODOSE_hctz", "Vli", "[Cve_hctz]"):
        assert hctz_core.loaded_value(sid) == pytest.approx(
            hctz_roadrunner.loaded_value(sid), rel=1e-12
        )
    with pytest.raises(KeyError):
        hctz_core.loaded_value("no_such_entity")


def test_value_is_the_value_as_loaded(hctz_core: CoreSBMLModel) -> None:
    """The engine is stateless, so a current value is the value as loaded."""
    assert hctz_core.value("Vli") == hctz_core.loaded_value("Vli")
    assert hctz_core.value("time") == 0.0
    with pytest.raises(RuntimeError, match="no_such_entity"):
        hctz_core.value("no_such_entity")


def test_has_selection_covers_the_default_selections(hctz_core: CoreSBMLModel) -> None:
    """Every default selection of the model is one the engine delivers."""
    assert all(hctz_core.has_selection(s) for s in hctz_core.default_selections())
    assert hctz_core.has_selection("time")
    assert not hctz_core.has_selection("no_such_entity")


def test_the_selections_are_set_and_checked(hctz_core: CoreSBMLModel) -> None:
    """The selections are the default ones, given ones are checked against the model."""
    assert hctz_core.selections == hctz_core.default_selections()
    assert not hctz_core.selections_given
    hctz_core.set_selections(["time", "[Cve_hctz]"])
    assert hctz_core.selections == ["time", "[Cve_hctz]"]
    assert hctz_core.selections_given
    with pytest.raises(ValueError, match="no_such_entity"):
        hctz_core.set_selections(["no_such_entity"])


def test_two_models_of_one_sbml_share_the_compiled_model() -> None:
    """A model is compiled and loaded once per SBML in a process."""
    a = load_model(AbstractModel(source=HCTZ_SBML), Engine.CORE)
    b = load_model(AbstractModel(source=HCTZ_SBML), Engine.CORE)
    assert isinstance(a, CoreSBMLModel)
    assert isinstance(b, CoreSBMLModel)
    assert a.loaded is b.loaded
    assert a is not b
    assert any(loaded is a.loaded for loaded in COMPILED.values())


def test_a_model_with_added_parameters_is_compiled_on_its_own() -> None:
    """The compiled model is the SBML with the added parameters, see `COMPILED`."""
    plain = load_model(AbstractModel(source=REPRESSILATOR_SBML), Engine.CORE)
    added = load_model(
        AbstractModel(source=REPRESSILATOR_SBML, parameters={"noise": 2.0}),
        Engine.CORE,
    )
    assert isinstance(plain, CoreSBMLModel)
    assert isinstance(added, CoreSBMLModel)
    assert plain.loaded is not added.loaded
    assert added.loaded_value("noise") == 2.0
    assert not plain.has_selection("noise")


@pytest.mark.parametrize(("sbml", "match"), [(DELAYED, "delay")])
def test_a_model_the_compiler_refuses_names_its_feature(sbml: str, match: str) -> None:
    """A model the compiler of the engine refuses raises with its message."""
    with pytest.raises(UnsupportedModelError, match=match):
        load_model(AbstractModel(source=sbml), Engine.CORE)


def test_a_model_with_an_algebraic_rule_loads() -> None:
    """An algebraic rule was refused until sbmlsim-core 0.10.0, which solves it.

    The matched unknown `y` is appended to the state vector as the algebraic
    suffix, so both parameters of the model are selections of it.
    """
    model = load_model(AbstractModel(source=ALGEBRAIC), Engine.CORE)
    assert isinstance(model, CoreSBMLModel)
    assert model.has_selection("x")
    assert model.has_selection("y")


def test_a_model_the_engine_cannot_load_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `LoadError` of the engine is an `UnsupportedModelError` as a `CompileError` is.

    Nothing of the bindings reaches a caller of `load_model`, which knows the
    errors of sbmlsim only.
    """
    core = import_core()

    class _Runtime:
        def load(self, wasm: bytes, description: dict[str, Any]) -> Any:
            raise core.LoadError("the module does not match its description")

    monkeypatch.setattr(model_core, "runtime", _Runtime)
    # an SBML of its own, so that the compiled model of another test is not read
    model = AbstractModel(source=REPRESSILATOR_SBML, parameters={"refused_load": 1.0})
    with pytest.raises(UnsupportedModelError, match="cannot load"):
        load_model(model, Engine.CORE)
    assert not any("refused_load" in sbml for sbml, _optimize, _jacobian in COMPILED), (
        "a model which was not loaded is no entry of the cache"
    )


def test_the_absolute_tolerance_is_a_vector_in_the_state_order(
    hctz_core: CoreSBMLModel, hctz_roadrunner: RoadrunnerSBMLModel
) -> None:
    """The tolerance of every state is the one roadrunner sets on CVODE."""
    tolerance = AbsoluteTolerance(amount=1e-10, concentration=1e-9, other=1e-8)
    hctz_core.set_integrator_settings(absolute_tolerance=tolerance)
    hctz_roadrunner.set_integrator_settings(absolute_tolerance=tolerance)
    tolerances = per_state(hctz_core)
    assert len(tolerances) == len(hctz_core.state_ids())
    assert tolerances == pytest.approx(tolerance_vector(hctz_roadrunner), rel=1e-12)
    assert len(set(tolerances)) > 1


def test_a_relative_tolerance_scales_every_state_as_in_roadrunner(
    hctz_core: CoreSBMLModel, hctz_roadrunner: RoadrunnerSBMLModel
) -> None:
    """A tolerance relative to the scale of a state reads the values as loaded."""
    tolerance = AbsoluteTolerance(relative=1e-8)
    hctz_core.set_integrator_settings(absolute_tolerance=tolerance)
    hctz_roadrunner.set_integrator_settings(absolute_tolerance=tolerance)
    assert per_state(hctz_core) == pytest.approx(
        tolerance_vector(hctz_roadrunner), rel=1e-12
    )


def test_a_float_absolute_tolerance_is_the_vector_of_roadrunner(
    hctz_core: CoreSBMLModel, hctz_roadrunner: RoadrunnerSBMLModel
) -> None:
    """A float absolute tolerance is the tolerance of every kind, as in roadrunner.

    The engine integrates with the tolerances of the states and not with one
    number for all of them, which the `scalar` of its contract would be: the
    tolerance of a species of a concentration is scaled by the volume of its
    compartment, so a float gives the vector roadrunner sets on CVODE.
    """
    hctz_core.set_integrator_settings(absolute_tolerance=1e-9)
    hctz_roadrunner.set_integrator_settings(absolute_tolerance=1e-9)
    assert hctz_core.absolute_tolerance == AbsoluteTolerance.of(1e-9)
    assert per_state(hctz_core) == pytest.approx(
        tolerance_vector(hctz_roadrunner), rel=1e-12
    )
    # the state of the integrator carries the tolerance of every engine, so
    # the key of a worker of a pool does not split equal models
    settings = dict(hctz_core.integrator_state()[1])
    assert settings["absolute_tolerance"] == AbsoluteTolerance.of(1e-9)


def test_the_settings_of_a_request_are_complete_and_accepted() -> None:
    """The settings of a request are the ones of the model and the engine takes them."""
    model = load_model(REPRESSILATOR_SBML, Engine.CORE)
    assert isinstance(model, CoreSBMLModel)
    model.set_integrator_settings(
        relative_tolerance=1e-10,
        maximum_num_steps=10000,
        maximum_time_step=0.5,
        initial_time_step=1e-3,
        stiff=False,
    )
    settings = model.request_settings()
    assert settings["relative_tolerance"] == 1e-10
    assert settings["maximum_num_steps"] == 10000
    assert settings["maximum_time_step"] == 0.5
    assert settings["initial_time_step"] == 1e-3
    assert settings["stiff"] is False
    output = model.loaded.instance().simulate(
        {
            "contract": "simulation/1",
            "start": 0.0,
            "end": 10.0,
            "output": {"uniform": {"steps": 5}},
            "preinit": [],
            "changes": [],
            "steady_state": None,
            "selections": ["time", "PX"],
            "seed": 1,
            "settings": settings,
        }
    )
    assert output.columns == ["time", "PX"]
    assert output.values.shape == (6, 2)


def test_a_setting_the_request_has_no_field_for_is_reported_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A setting of roadrunner which the request has no field for is ignored.

    The request has a field for `initial_time_step`, which COPASI and AMICI
    ignore, and none for `variable_step_size`, which its output states.
    """
    with caplog.at_level(logging.WARNING, logger="sbmlsim.model.model_core"):
        model = CoreSBMLModel(
            source=REPRESSILATOR_SBML,
            settings={"variable_step_size": True, "initial_time_step": 0.1},
        )
        model.set_integrator_settings(variable_step_size=False, relative_tolerance=1e-9)
    assert caplog.text.count("variable_step_size") == 1
    assert "initial_time_step" not in caplog.text
    assert "variable_step_size" not in model.request_settings()
    assert model.request_settings()["initial_time_step"] == 0.1
    assert model.request_settings()["relative_tolerance"] == 1e-9


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"no_such_setting": 1.0}, "no settings"),
        ({"absolute_tolerance": True}, "The absolute tolerance is a number"),
        ({"absolute_tolerance": 0.0}, "finite and positive"),
        ({"maximum_num_steps": 0}, "positive integer"),
        ({"maximum_num_steps": 1.5}, "positive integer"),
        ({"relative_tolerance": -1.0}, "is a number"),
        ({"stiff": 2}, "is a boolean"),
    ],
)
def test_a_setting_which_is_not_valid_raises(
    settings: dict[str, Any], match: str
) -> None:
    """A name which is no setting of roadrunner and a value which is not valid raise."""
    model = load_model(REPRESSILATOR_SBML, Engine.CORE)
    assert isinstance(model, CoreSBMLModel)
    before = model.request_settings()
    with pytest.raises(ValueError, match=match):
        model.set_integrator_settings(**{"maximum_time_step": 0.25, **settings})
    assert model.request_settings() == before


def test_the_state_of_the_integrator_is_complete_and_round_trips(
    hctz_core: CoreSBMLModel,
) -> None:
    """Every setting of a request is in the state, which is set again as it is."""
    tolerance = AbsoluteTolerance(other=1e-9)
    hctz_core.set_integrator_settings(
        absolute_tolerance=tolerance, relative_tolerance=1e-11, stiff=False
    )
    integrator, settings = hctz_core.integrator_state()
    assert integrator == INTEGRATOR
    assert dict(settings) == {
        **DEFAULT_SETTINGS,
        "absolute_tolerance": tolerance,
        "relative_tolerance": 1e-11,
        "stiff": False,
    }
    other = load_model(HCTZ_SBML, Engine.CORE)
    assert isinstance(other, CoreSBMLModel)
    # the state replaces the settings of a model which carries its own
    other.set_integrator_settings(relative_tolerance=1e-5, maximum_num_steps=100)
    other.set_integrator_state((integrator, settings))
    assert other.integrator_state() == (integrator, settings)
    assert other.request_settings() == hctz_core.request_settings()
    with pytest.raises(ValueError, match="cvode"):
        other.set_integrator_state(("lsoda", settings))


def test_the_pre_initialization_of_a_plan_is_recorded(hctz_core: CoreSBMLModel) -> None:
    """The engine is stateless: an initialization records what the executor sends."""
    assignments = (
        Assignment(target="PODOSE_hctz", kind=TargetKind.SPECIES_AMOUNT, value=2.0),
    )
    hctz_core.initialize(assignments)
    assert hctz_core.assignments == assignments
    assert hctz_core.loaded_value("PODOSE_hctz") == 0.0
    hctz_core.simulated()
    assert hctz_core.assignments == assignments
    hctz_core.reset()
    assert hctz_core.assignments == ()
