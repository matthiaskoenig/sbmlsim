"""Tests of the hybridization of a network and a model."""

import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import libsbml
import numpy as np
import pytest

from sbmlsim.fit.derived import DerivedChanges, _group_derived_changes
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.sciml import NetworkImportError
from sbmlsim.sciml.errors import NetworkHybridizationError
from sbmlsim.sciml.hybridization import (
    ALL_CONDITIONS,
    Hybridization,
    NetworkInput,
    NetworkPattern,
    entity_of,
)
from tests.sciml.hybrid import (
    MODEL_PATH,
    convolution,
    feed_forward,
    nominal_values,
    two_inputs,
    write_model,
)

PRE = NetworkPattern.PRE_INITIALIZATION
RHS = NetworkPattern.RHS
OBSERVABLE = NetworkPattern.OBSERVABLE


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Write the model of Lotka and Volterra."""
    return write_model(tmp_path / "lv.xml")


def _edited(path: Path, edit: Any) -> Path:
    """Write a copy of the model at `path` which `edit(model)` changed."""
    document = libsbml.readSBMLFromFile(str(path))
    edit(document.getModel())
    edited = path.with_name(f"edited_{path.name}")
    assert libsbml.writeSBMLToFile(document, str(edited))
    return edited


def _inputs(first: str = "prey", second: str = "predator") -> dict[str, NetworkInput]:
    return {
        "net1__input0__0": NetworkInput(formula=first),
        "net1__input0__1": NetworkInput(formula=second),
    }


def _hybridization(pattern: NetworkPattern | str = RHS, **kwargs: Any) -> Hybridization:
    compiled = NetworkPattern(pattern).is_compiled if pattern != "ode" else True
    arguments: dict[str, Any] = {
        "network": feed_forward(),
        "pattern": pattern,
        "model": "lv",
        "inputs": _inputs() if compiled else _inputs("alpha", "2 * k"),
        "outputs": {"net1__output0__0": "gamma"},
    }
    if not compiled:
        arguments["constants"] = {"k": 0.5}
    arguments.update(kwargs)
    return Hybridization(**arguments)


def _values(hybridization: Hybridization, **values: float) -> dict[str, float]:
    """Get the values of a fit: the elements at their nominal values, alpha."""
    return {**nominal_values(hybridization.network), "alpha": 1.3, **values}


# --- AN INPUT ---


def test_an_input_is_a_formula_or_arrays() -> None:
    """Exactly one of the three is given."""
    assert NetworkInput(formula="prey").all_formulas() == ["prey"]
    assert NetworkInput(formulas={"e1": "1.0", "e2": "k"}).all_formulas() == [
        "1.0",
        "k",
    ]
    assert NetworkInput(arrays={"e1": [1.0, 2.0]}).all_formulas() == []
    with pytest.raises(ValueError, match="but nothing is given"):
        NetworkInput()
    with pytest.raises(ValueError, match=r"but \['formula', 'arrays'\] is given"):
        NetworkInput(formula="prey", arrays={ALL_CONDITIONS: [1.0]})
    with pytest.raises(ValueError, match="formulas of an input name no condition"):
        NetworkInput(formulas={})
    with pytest.raises(ValueError, match="arrays of an input name no condition"):
        NetworkInput(arrays={})


@pytest.mark.parametrize("formula", ["", "prey +", "f(prey)"])
def test_the_formula_of_an_input_is_math(formula: str) -> None:
    """A formula which is not math is an error when the input is created."""
    with pytest.raises(ValueError, match="The formula"):
        NetworkInput(formula=formula)
    with pytest.raises(ValueError, match="The formula"):
        NetworkInput(formulas={"e1": formula})


def test_the_arrays_of_an_input() -> None:
    """The arrays have one shape and finite values, and are copies."""
    values = np.array([1.0, 2.0, 3.0])
    network_input = NetworkInput(arrays={"e1": values, "e2": [3, 2, 1]})
    values[0] = 5.0
    assert network_input.shape == (3,)
    np.testing.assert_array_equal(network_input.array_of("e1"), [1.0, 2.0, 3.0])
    stored = network_input.array_of("e1")
    assert stored is not None
    with pytest.raises(ValueError, match="read-only"):
        stored[0] = 5.0
    with pytest.raises(ValueError, match=r"one shape.*'e1': \(3,\).*'e2': \(2,\)"):
        NetworkInput(arrays={"e1": [1.0, 2.0, 3.0], "e2": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"condition 'e2'.*not finite"):
        NetworkInput(arrays={"e1": [1.0], "e2": [np.nan]})


def test_the_value_of_a_condition() -> None:
    """A condition which is not listed has the value of all conditions."""
    arrays = NetworkInput(arrays={ALL_CONDITIONS: [1.0], "e2": [2.0]})
    np.testing.assert_array_equal(arrays.array_of("e1"), [1.0])
    np.testing.assert_array_equal(arrays.array_of("e2"), [2.0])
    assert arrays.is_conditional
    assert arrays.formula_of("e1") is None
    assert NetworkInput(arrays={"e2": [2.0]}).array_of("e1") is None
    assert not NetworkInput(arrays={ALL_CONDITIONS: [1.0]}).is_conditional

    formulas = NetworkInput(formulas={"e1": "10.0", ALL_CONDITIONS: "k"})
    assert formulas.formula_of("e1") == "10.0"
    assert formulas.formula_of("e2") == "k"
    assert formulas.array_of("e1") is None
    assert NetworkInput(formulas={"e1": "10.0"}).formula_of("e2") is None
    assert NetworkInput(formula="k").formula_of("e2") == "k"
    assert not NetworkInput(formula="k").is_conditional


def test_the_equality_of_inputs() -> None:
    """Inputs are compared by their formulas and the elements of their arrays."""
    assert NetworkInput(formula="prey") == NetworkInput(formula="prey")
    assert NetworkInput(formula="prey") != NetworkInput(formula="predator")
    assert NetworkInput(formula="prey") != NetworkInput(formulas={"e1": "prey"})
    assert NetworkInput(arrays={"e1": [1.0, 2.0]}) == NetworkInput(
        arrays={"e1": np.array([1.0, 2.0])}
    )
    assert NetworkInput(arrays={"e1": [1.0, 2.0]}) != NetworkInput(
        arrays={"e1": [1.0, 3.0]}
    )
    assert NetworkInput(arrays={"e1": [1.0]}) != NetworkInput(arrays={"e2": [1.0]})
    assert NetworkInput(arrays={"e1": [1.0]}) != NetworkInput(formula="prey")
    assert NetworkInput(formula="prey") != "prey"


# --- THE HYBRIDIZATION AND ITS NETWORK ---


def test_a_hybridization() -> None:
    """The attributes are copies, the pattern is read from its value."""
    outputs = {"net1__output0__0": "gamma"}
    hybridization = _hybridization(pattern="rhs", outputs=outputs, frozen=[])
    outputs["net1__output0__0"] = "alpha"
    assert hybridization.pattern is RHS
    assert hybridization.outputs == {"net1__output0__0": "gamma"}
    assert hybridization.frozen == frozenset()
    assert hybridization.input_shapes() == [(2,)]
    assert hybridization.output_shapes() == [(1,)]
    assert isinstance(hybridization, DerivedChanges)
    with pytest.raises(TypeError, match="unhashable type: 'Hybridization'"):
        hash(hybridization)


def test_the_patterns() -> None:
    """The networks of two patterns are compiled into the model."""
    assert not PRE.is_compiled
    assert RHS.is_compiled
    assert OBSERVABLE.is_compiled
    with pytest.raises(NetworkHybridizationError, match=r"'net1'.*'ode' is not one of"):
        _hybridization(pattern="ode")


def test_the_entity_of_a_target() -> None:
    """The target of a concentration names its species."""
    assert entity_of("[prey]") == "prey"
    assert entity_of("prey") == "prey"


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"net1__input0__0": NetworkInput(formula="prey")}, r"\(1,\)\] do not fit"),
        (
            {
                "net1__input0__0": NetworkInput(formula="prey"),
                "net1__input0__2": NetworkInput(formula="prey"),
            },
            r"the input 0 has the shape \(3,\), but its elements \[\(1,\)\] are missing",
        ),
        ({}, "the input 0 of the forward pass is missing"),
        (
            {**_inputs(), "net1__input1__0": NetworkInput(formula="prey")},
            "is the input 1, but the forward pass has 1 inputs",
        ),
        ({"net1__in0__0": NetworkInput(formula="prey")}, "is not the id of an input"),
        ({"net1__input0": NetworkInput(formula="prey")}, "name the element"),
        (
            {"net1__input0__0": NetworkInput(arrays={"e1": [1.0, 2.0]})},
            r"is an element, but its arrays have the shape \(2,\)",
        ),
        (
            {
                "net1__input0": NetworkInput(arrays={"e1": [1.0, 2.0]}),
                "net1__input0__0": NetworkInput(formula="prey"),
            },
            r"the inputs \[0\] are given as an array and element by element",
        ),
        (
            {
                "net1__input0__0": NetworkInput(formula="prey"),
                "net1__input0__0_1": NetworkInput(formula="prey"),
            },
            "differ in their number of axes",
        ),
        ({"net1__input0__0": "prey"}, "is not a `NetworkInput`"),
        ({"net1__input0": NetworkInput(arrays={"e1": np.ones(3)})}, "do not fit"),
    ],
)
def test_inputs_which_do_not_fit_the_network(inputs: dict, message: str) -> None:
    """The inputs cover the inputs of the forward pass, the error names them."""
    with pytest.raises(NetworkHybridizationError, match=message) as excinfo:
        _hybridization(inputs=inputs)
    assert "net1" in str(excinfo.value)


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ({}, "no output has a target"),
        ({"net1__output0__1": "gamma"}, r"not an element of the output 0.*\(1,\)"),
        ({"net1__output0__0_0": "gamma"}, r"not an element of the output 0.*\(1,\)"),
        ({"net1__output1__0": "gamma"}, "is the output 1, but the forward pass has 1"),
        ({"net1__output0": "gamma"}, "names no element"),
        ({"net1__out0__0": "gamma"}, "is not the id of an output"),
        ({"net1__output0__0": ""}, "is not an id"),
        ({"net1__output0__0": "[]"}, "is not an id"),
        ({"net1__output0__0": 1.0}, "is not an id"),
    ],
)
def test_outputs_which_do_not_fit_the_network(outputs: dict, message: str) -> None:
    """The outputs are elements of the outputs of the network."""
    with pytest.raises(NetworkHybridizationError, match=message) as excinfo:
        _hybridization(outputs=outputs)
    assert "net1" in str(excinfo.value)


def test_two_outputs_with_one_target() -> None:
    """An entity is set by one output."""
    network = feed_forward(n_outputs=2)
    with pytest.raises(NetworkHybridizationError, match=r"both set 'prey'"):
        Hybridization(
            network=network,
            pattern=PRE,
            model="lv",
            inputs=_inputs("alpha", "beta"),
            outputs={"net1__output0__0": "prey", "net1__output0__1": "[prey]"},
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frozen": {"net1__layer9__weight__0_0"}}, "are not elements of the network"),
        ({"constants": {"k": np.inf}}, "the constant 'k' is 'inf'"),
        ({"constants": {"k": "a"}}, "the constant 'k' is 'a', not a number"),
        ({"constants": {"k": None}}, "the constant 'k' is 'None', not a number"),
        (
            {"frozen": "net1__layer1__bias__0"},
            "frozen elements are a collection of ids, not the id",
        ),
        ({"model": ""}, "is not an id"),
        ({"model": None}, "is not an id"),
    ],
)
def test_attributes_which_are_not_valid(kwargs: dict, message: str) -> None:
    """The error names the network and the attribute."""
    with pytest.raises(NetworkHybridizationError, match=message) as excinfo:
        _hybridization(**kwargs)
    assert "Network 'net1'" in str(excinfo.value)


def test_a_network_without_values() -> None:
    """A network is not evaluated with arrays which have no values."""
    network = feed_forward()
    parameters = {"layer1": dict(network.parameters["layer1"])}
    with pytest.raises(NetworkImportError, match=r"'layer2'.*has no values"):
        _hybridization(network=replace(network, parameters=parameters))


def test_a_network_which_is_compiled_is_evaluated_on_expressions() -> None:
    """A convolution runs before the simulation, not in the model."""
    inputs = {"net3__input0": NetworkInput(arrays={ALL_CONDITIONS: np.ones((1, 4, 4))})}
    outputs = {"net3__output0__0": "gamma"}
    hybridization = Hybridization(
        network=convolution(), pattern=PRE, model="lv", inputs=inputs, outputs=outputs
    )
    assert hybridization.output_shapes() == [(1,)]
    for pattern in (RHS, OBSERVABLE):
        with pytest.raises(
            NetworkHybridizationError, match=r"'net3'.*not evaluated on expressions"
        ):
            Hybridization(
                network=convolution(),
                pattern=pattern,
                model="lv",
                inputs=inputs,
                outputs=outputs,
            )


def test_a_network_which_is_compiled_has_one_formula_per_input() -> None:
    """A rule of a model does not depend on the condition."""
    inputs = {
        "net1__input0__0": NetworkInput(formulas={"e1": "prey", "e2": "predator"}),
        "net1__input0__1": NetworkInput(formula="predator"),
    }
    with pytest.raises(NetworkHybridizationError, match="one formula per input"):
        _hybridization(pattern=RHS, inputs=inputs)
    assert _hybridization(pattern=PRE, inputs=inputs).pattern is PRE


# --- THE DERIVED CHANGES ---


def test_the_derived_changes_are_the_forward_pass() -> None:
    """The outputs of the network at the inputs are the changes of the targets."""
    hybridization = _hybridization(pattern=PRE)
    network = hybridization.network
    changes = hybridization.derived_changes(_values(hybridization), condition="e1")
    (expected,) = network.forward(np.array([1.3, 1.0]))
    assert changes == {"gamma": pytest.approx(expected[0])}
    assert hybridization.targets() == {"gamma"}
    assert hybridization.symbols() == {"alpha", "k", *network.parameter_ids()}


def test_an_element_without_a_value() -> None:
    """An element which is estimated has a value, a frozen one is nominal."""
    frozen = "net1__layer1__bias__0"
    hybridization = _hybridization(pattern=PRE, frozen={frozen})
    values = _values(hybridization)
    del values[frozen]
    hybridization.derived_changes(values, condition="e1")
    del values["net1__layer2__bias__0"]
    with pytest.raises(
        NetworkHybridizationError,
        match=r"Network 'net1'.*\['net1__layer2__bias__0'\].*have no value",
    ):
        hybridization.derived_changes(values, condition="e1")


def test_the_derived_changes_use_the_values_of_the_elements() -> None:
    """An element which is not frozen has the value of the fit."""
    sid = "net1__layer2__bias__0"
    frozen = "net1__layer1__bias__0"
    hybridization = _hybridization(pattern=PRE, frozen={frozen})
    assert sid in hybridization.symbols()
    assert frozen not in hybridization.symbols()
    values = _values(hybridization)
    nominal = hybridization.derived_changes(values, condition="e1")["gamma"]
    shifted = hybridization.derived_changes(
        {**values, sid: values[sid] + 2.0}, condition="e1"
    )["gamma"]
    assert shifted == pytest.approx(nominal + 2.0)
    # a value of a frozen element is not read
    ignored = hybridization.derived_changes({**values, frozen: 100.0}, condition="e1")[
        "gamma"
    ]
    assert ignored == nominal
    # the network keeps its nominal values
    assert hybridization.derived_changes(values, "e1")["gamma"] == nominal


def test_the_values_have_precedence_over_the_constants() -> None:
    """A constant is the value of a symbol which the fit does not give."""
    hybridization = _hybridization(pattern=PRE)
    with_constant = hybridization.input_values({"alpha": 1.3}, condition="e1")
    np.testing.assert_allclose(with_constant[0], [1.3, 1.0])
    with_value = hybridization.input_values({"alpha": 1.3, "k": 2.0}, condition="e1")
    np.testing.assert_allclose(with_value[0], [1.3, 4.0])


def test_the_inputs_of_a_condition() -> None:
    """Formulas and arrays are selected by the condition of the simulation."""
    network = two_inputs()
    hybridization = Hybridization(
        network=network,
        pattern=PRE,
        model="lv",
        inputs={
            "net6__input0__0": NetworkInput(formulas={"e1": "10.0", "e2": "alpha"}),
            "net6__input1": NetworkInput(
                arrays={"e1": [1.0, 2.0, 3.0], "e2": [3.0, 2.0, 1.0]}
            ),
        },
        outputs={"net6__output0__0": "gamma"},
    )
    first, second = hybridization.input_values({"alpha": 1.3}, condition="e1")
    np.testing.assert_allclose(first, [10.0])
    np.testing.assert_allclose(second, [1.0, 2.0, 3.0])
    first, second = hybridization.input_values({"alpha": 1.3}, condition="e2")
    np.testing.assert_allclose(first, [1.3])
    np.testing.assert_allclose(second, [3.0, 2.0, 1.0])
    values = _values(hybridization)
    e1 = hybridization.derived_changes(values, condition="e1")["gamma"]
    e2 = hybridization.derived_changes(values, condition="e2")["gamma"]
    assert e1 == pytest.approx(network.forward(first * 0 + 10.0, second[::-1])[0][0])
    assert e1 != e2

    with pytest.raises(
        NetworkHybridizationError,
        match=r"'net6'.*'net6__input0__0' has no formula for the condition 'e3'",
    ):
        hybridization.derived_changes(values, condition="e3")


def test_an_array_without_the_condition() -> None:
    """An input without an array for a condition names the condition."""
    hybridization = Hybridization(
        network=two_inputs(),
        pattern=PRE,
        model="lv",
        inputs={
            "net6__input0__0": NetworkInput(formula="alpha"),
            "net6__input1": NetworkInput(arrays={"e1": [1.0, 2.0, 3.0]}),
        },
        outputs={"net6__output0__0": "gamma"},
    )
    with pytest.raises(
        NetworkHybridizationError,
        match=r"'net6__input1' has no array for the condition 'e2'.*\['e1'\]",
    ):
        hybridization.derived_changes(_values(hybridization), condition="e2")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({}, r"input 'net1__input0__0'.*uses \['alpha'\], which have no value"),
        ({"alpha": np.nan}, "which is not a finite number"),
        ({"alpha": np.array([1.0, 2.0])}, "which is not a finite number"),
    ],
)
def test_an_input_without_a_number(values: dict, message: str) -> None:
    """A symbol without a value and a value which is no number are errors."""
    with pytest.raises(NetworkHybridizationError, match=message) as excinfo:
        _hybridization(pattern=PRE).derived_changes(values, condition="e1")
    assert "Network 'net1'" in str(excinfo.value)


def test_an_input_which_is_not_defined() -> None:
    """A division by zero is an error which names the input."""
    hybridization = _hybridization(
        pattern=PRE, inputs=_inputs("alpha", "1 / k"), constants={"k": 0.0}
    )
    with pytest.raises(
        NetworkHybridizationError, match="Network 'net1': input 'net1__input0__1'"
    ):
        hybridization.derived_changes(_values(hybridization), condition="e1")


def test_an_output_which_is_not_finite() -> None:
    """An output which overflows is an error and not a change of the model."""
    hybridization = _hybridization(pattern=PRE, inputs=_inputs("alpha", "exp(alpha)"))
    with (
        pytest.raises(NetworkHybridizationError, match="input 'net1__input0__1'"),
        np.errstate(over="ignore"),
    ):
        hybridization.derived_changes(_values(hybridization, alpha=1e6), "e1")


def test_the_derived_changes_of_a_compiled_network() -> None:
    """The model evaluates the network, the fit sets the arrays of a condition.

    The hook reads the outputs and the frozen elements from the model, which
    must carry the values of the network.
    """
    assert _hybridization(pattern=RHS).symbols() == {"net1__output0__0"}
    assert _hybridization(pattern=RHS).targets() == frozenset()
    assert _hybridization(pattern=RHS).derived_changes({}, "e1") == {}

    frozen = _hybridization(pattern=RHS, frozen={"net1__layer1__bias__0"})
    assert frozen.symbols() == {"net1__output0__0", "net1__layer1__bias__0"}
    (bias,) = frozen._frozen_values[1]
    assert frozen.derived_changes({"net1__layer1__bias__0": bias}, "e1") == {}
    assert (
        frozen.derived_changes({"net1__layer1__bias__0": bias * (1 + 1e-15)}, "e1")
        == {}
    )
    with pytest.raises(NetworkHybridizationError, match=r"compile the network again"):
        frozen.derived_changes({"net1__layer1__bias__0": bias + 1e-6}, "e1")
    with pytest.raises(NetworkHybridizationError, match=r"have no value"):
        frozen.derived_changes({}, "e1")

    def hybridization(arrays: dict) -> Hybridization:
        return Hybridization(
            network=two_inputs(),
            pattern=RHS,
            model="lv",
            inputs={
                "net6__input0__0": NetworkInput(formula="prey"),
                "net6__input1": NetworkInput(arrays=arrays),
            },
            outputs={"net6__output0__0": "gamma"},
        )

    # one array for every condition is a part of the model
    constant = hybridization({ALL_CONDITIONS: [1.0, 2.0, 3.0]})
    assert constant.targets() == frozenset()
    assert constant.derived_changes({}, "e1") == {}

    conditional = hybridization({"e1": [1.0, 2.0, 3.0], "e2": [3.0, 2.0, 1.0]})
    ids = ["net6__input1__0", "net6__input1__1", "net6__input1__2"]
    assert conditional.targets() == set(ids)
    assert conditional.derived_changes({}, "e2") == dict(
        zip(ids, [3.0, 2.0, 1.0], strict=True)
    )
    with pytest.raises(NetworkHybridizationError, match="no array for the condition"):
        conditional.derived_changes({}, "e3")


def test_the_parameters_of_a_fit_are_checked() -> None:
    """A fit does not write what the network sets or holds constant."""
    frozen = "net1__layer1__bias__0"
    hybridization = _hybridization(pattern=PRE, frozen={frozen})
    hybridization.check_parameters(["alpha", "beta", "net1__layer1__bias__1"])
    with pytest.raises(NetworkHybridizationError, match=rf"\['{frozen}'\] are frozen"):
        hybridization.check_parameters(["alpha", frozen])
    with pytest.raises(NetworkHybridizationError, match=r"\['gamma'\] are set by"):
        hybridization.check_parameters(["alpha", "gamma"])
    with pytest.raises(NetworkHybridizationError, match=r"\['net1__output0__0'\]"):
        hybridization.check_parameters(["net1__output0__0"])


def test_a_hybridization_is_pickled() -> None:
    """A fit pickles its hybridizations for the workers, the copy is equal."""
    hybridization = Hybridization(
        network=two_inputs(),
        pattern=PRE,
        model="lv",
        inputs={
            "net6__input0__0": NetworkInput(formula="alpha"),
            "net6__input1": NetworkInput(arrays={"e1": [1.0, 2.0, 3.0]}),
        },
        outputs={"net6__output0__0": "gamma"},
        frozen={"net6__layer1__bias__0"},
        constants={"k": 1.0},
    )
    copy = pickle.loads(pickle.dumps(hybridization))
    assert copy == hybridization
    values = _values(hybridization)
    assert copy.derived_changes(values, "e1") == hybridization.derived_changes(
        values, "e1"
    )
    assert copy != _hybridization()


def test_a_concentration_and_its_species_are_one_entity() -> None:
    """The fit refuses two targets of one species, the change sets the selection."""
    concentration = _hybridization(pattern=PRE, outputs={"net1__output0__0": "[prey]"})
    assert concentration.targets() == {"[prey]"}
    assert set(concentration.derived_changes(_values(concentration), "e1")) == {
        "[prey]"
    }
    species = Hybridization(
        network=feed_forward(sid="net2"),
        pattern=PRE,
        model="lv",
        inputs={
            "net2__input0__0": NetworkInput(formula="alpha"),
            "net2__input0__1": NetworkInput(formula="beta"),
        },
        outputs={"net2__output0__0": "prey"},
    )
    model = RoadrunnerSBMLModel(source=MODEL_PATH)
    with pytest.raises(ValueError, match="two hybridizations of the model 'lv' set"):
        _group_derived_changes("'p':", [species, concentration], [], model, {})
    with pytest.raises(ValueError, match=r"sets '\[prey\]', which the first time"):
        _group_derived_changes("'p':", [concentration], [], model, {"prey": 1.0})


# --- THE HYBRIDIZATION AND ITS MODEL ---


def test_a_hybridization_fits_its_model(model_path: Path) -> None:
    """The three patterns are valid for the model."""
    _hybridization(pattern=RHS).validate(model_path)
    _hybridization(pattern=PRE).validate(model_path)
    _hybridization(pattern=PRE, outputs={"net1__output0__0": "[prey]"}).validate(
        model_path
    )
    _hybridization(pattern=OBSERVABLE, outputs={"net1__output0__0": "y"}).validate(
        model_path
    )


@pytest.mark.parametrize(
    ("pattern", "kwargs", "message"),
    [
        (RHS, {"outputs": {"net1__output0__0": "kappa"}}, "'kappa'.*is not an entity"),
        (PRE, {"outputs": {"net1__output0__0": "kappa"}}, "'kappa'.*is not an entity"),
        (
            OBSERVABLE,
            {"outputs": {"net1__output0__0": "gamma"}},
            "the model 'lv.xml' has an entity 'gamma'",
        ),
        (
            OBSERVABLE,
            {"outputs": {"net1__output0__0": "[y]"}},
            "is the symbol of an observable and not a concentration",
        ),
        (RHS, {"outputs": {"net1__output0__0": "[prey]"}}, "is a concentration"),
        (PRE, {"outputs": {"net1__output0__0": "[gamma]"}}, "is a concentration"),
        (PRE, {"constants": {"k": 0.5, "alpha": 1.0}}, "'alpha' is an entity"),
        (RHS, {"inputs": _inputs("prey", "gamma")}, r"uses \['gamma'\], which the out"),
        (PRE, {"inputs": _inputs("alpha", "prey")}, "'prey', which is a species"),
        (PRE, {"inputs": _inputs("alpha", "time")}, "'time', which is the time"),
    ],
)
def test_a_hybridization_which_does_not_fit_its_model(
    model_path: Path, pattern: NetworkPattern, kwargs: dict, message: str
) -> None:
    """The error names the network and the target or the input."""
    hybridization = _hybridization(pattern=pattern, **kwargs)
    with pytest.raises(NetworkHybridizationError, match=message) as excinfo:
        hybridization.validate(model_path)
    assert "Network 'net1'" in str(excinfo.value)


def _assignment_rule(model: libsbml.Model) -> None:
    model.getParameter("gamma").setConstant(False)
    rule = model.createAssignmentRule()
    rule.setVariable("gamma")
    rule.setMath(libsbml.parseL3Formula("2 * alpha"))


def _rate_rule(model: libsbml.Model) -> None:
    model.getParameter("beta").setConstant(False)
    rule = model.createRateRule()
    rule.setVariable("beta")
    rule.setMath(libsbml.parseL3Formula("0.1"))


def _event(model: libsbml.Model) -> None:
    for sid in ("beta", "gamma"):
        model.getParameter(sid).setConstant(False)
    event = model.createEvent()
    event.setUseValuesFromTriggerTime(True)
    trigger = event.createTrigger()
    trigger.setInitialValue(False)
    trigger.setPersistent(True)
    trigger.setMath(libsbml.parseL3Formula("time > 1"))
    for sid in ("beta", "gamma"):
        assignment = event.createEventAssignment()
        assignment.setVariable(sid)
        assignment.setMath(libsbml.parseL3Formula("2"))


def _species_reference(model: libsbml.Model) -> None:
    model.getReaction("v1").getProduct(0).setId("sr1")


@pytest.mark.parametrize(
    ("pattern", "target", "edit", "message"),
    [
        (RHS, "prey", None, "'prey' of 'net1__output0__0' is a species, but"),
        (RHS, "default", None, "'default' of 'net1__output0__0' is a compartment"),
        (RHS, "v1", None, "'v1' of 'net1__output0__0' is a reaction"),
        (RHS, "gamma", _assignment_rule, "'gamma'.*is set by a rule"),
        (RHS, "gamma", _event, "'gamma'.*is changed by an event"),
        (PRE, "v1", None, "'v1' of 'net1__output0__0' is a reaction, but"),
        (PRE, "sr1", _species_reference, "'sr1'.*is a species reference"),
        (PRE, "gamma", _assignment_rule, "'gamma'.*is set by an assignment rule"),
        (OBSERVABLE, "2y", None, "'2y' of 'net1__output0__0' is not a valid SBML id"),
        (OBSERVABLE, "y z", None, "'y z' of 'net1__output0__0' is not a valid SBML"),
    ],
)
def test_a_target_which_the_model_cannot_take(
    model_path: Path, pattern: NetworkPattern, target: str, edit: Any, message: str
) -> None:
    """The error names the network, the target and why."""
    path = _edited(model_path, edit) if edit else model_path
    hybridization = _hybridization(
        pattern=pattern, outputs={"net1__output0__0": target}
    )
    with pytest.raises(NetworkHybridizationError, match=message) as excinfo:
        hybridization.validate(path)
    assert "Network 'net1'" in str(excinfo.value)


def test_a_target_which_the_model_can_take(model_path: Path) -> None:
    """A network before the simulation sets the initial value of a rate rule."""
    for target in ("beta", "default", "prey"):
        _hybridization(pattern=PRE, outputs={"net1__output0__0": target}).validate(
            _edited(model_path, _rate_rule)
        )
    _hybridization(pattern=PRE, outputs={"net1__output0__0": "gamma"}).validate(
        _edited(model_path, _event)
    )


@pytest.mark.parametrize(
    ("symbol", "edit", "message"),
    [
        ("v1", None, "'v1', which is a reaction"),
        ("sr1", _species_reference, "'sr1', which is a species reference"),
        ("beta", _rate_rule, "'beta', which is set by a rule"),
        ("beta", _event, "'beta', which is changed by an event"),
    ],
)
def test_an_input_which_varies(
    model_path: Path, symbol: str, edit: Any, message: str
) -> None:
    """An input before the simulation is a constant of the simulation."""
    path = _edited(model_path, edit) if edit else model_path
    hybridization = _hybridization(pattern=PRE, inputs=_inputs("alpha", symbol))
    with pytest.raises(NetworkHybridizationError, match=message) as excinfo:
        hybridization.validate(path)
    assert "Network 'net1'" in str(excinfo.value)


def test_an_input_which_a_rule_sets(tmp_path: Path, model_path: Path) -> None:
    """An entity with a rule is not a constant of a simulation."""
    document = libsbml.readSBMLFromFile(str(model_path))
    model = document.getModel()
    model.getParameter("alpha").setConstant(False)
    rule = model.createAssignmentRule()
    rule.setVariable("alpha")
    rule.setMath(libsbml.parseL3Formula("2 * prey"))
    path = tmp_path / "rule.xml"
    libsbml.writeSBMLToFile(document, str(path))
    with pytest.raises(
        NetworkHybridizationError, match="'alpha', which is set by a rule"
    ):
        _hybridization(pattern=PRE).validate(path)


def test_a_model_which_cannot_be_read(tmp_path: Path) -> None:
    """A file which is missing or no model names the network and the file."""
    with pytest.raises(NetworkHybridizationError, match=r"'net1'.*does not exist"):
        _hybridization().validate(tmp_path / "missing.xml")
    path = tmp_path / "empty.xml"
    path.write_text("<sbml/>")
    with pytest.raises(NetworkHybridizationError, match=r"'net1'.*holds no SBML model"):
        _hybridization().validate(path)
